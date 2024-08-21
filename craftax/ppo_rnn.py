import argparse
import os
import sys
from math import ceil, sqrt
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
import jaxpruner
import numpy as np
import optax
import time

from flax.training import orbax_utils
from matplotlib import pyplot as plt, animation
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

from craftax.craftax import craftax_state
from craftax.environment_base.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    VideoPlotWrapper,
    ReduceActionSpaceWrapper,
    CurriculumWrapper
)
from craftax.logz.batch_logging import create_log_dict, batch_log, reset_batch_logs


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_ENV_STEPS"] // config["NUM_ENVS"] // config['UPDATES_PER_VIZ']
    )

    config["NUM_LOG_STEPS"] = config["NUM_UPDATES"] * config["UPDATES_PER_VIZ"]

    # HACK: We have to use the original formula for num_updates for LR annealing,
    # modifying it breaks training due to its effect on LR scheduling
    config['NUM_UPDATES_FOR_LR_ANNEALING'] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_ENV_STEPS"] // config["NUM_ENVS"]
    )

    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_ENV_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # Define static params, modify based on command line flags and pass to env object to hold during runtime
    # We modify static params here because there's a number of core game logic functions that take static params
    # And don't take the normal "params" blob
    static_params = craftax_state.StaticEnvParams()
    if config['REWARD_FUNCTION'] == 'vanilla':
        static_params.reward_func = 'vanilla'


    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
        )

        env = CraftaxClassicSymbolicEnv()
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
        )

        env = CraftaxClassicPixelsEnv()
        is_symbolic = False
    elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

        env = CraftaxSymbolicEnv(static_params)
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

        env = CraftaxPixelsEnv(static_params)
        is_symbolic = False
    else:
        raise ValueError(f"Unknown env: {config['ENV_NAME']}")
    env_params = env.default_params

    # Restrict action space
    if not config['FULL_ACTION_SPACE']:
        env = ReduceActionSpaceWrapper(env)

    # Env version to log videos, use only for occasional visualization as plotting is expensive/slow
    # TODO why do I need to put this wrapper early in the stack? It can't just layer on top
    env_viz = VideoPlotWrapper(env, config['OUTPUT_PATH'], config['FRAMES_PER_FILE'], not config['NO_VIDEOS'])

    env = LogWrapper(env)

    if not os.path.isdir(config['OUTPUT_PATH']):
        os.makedirs(config['OUTPUT_PATH'])

    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
        env_viz = OptimisticResetVecEnvWrapper(
            env_viz,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])
        env_viz = AutoResetEnvWrapper(env_viz)
        env_viz = BatchEnvWrapper(env_viz, num_envs=config["NUM_ENVS"])

    env = CurriculumWrapper(env, num_envs=config["NUM_ENVS"],
                            num_steps=config["NUM_LOG_STEPS"],
                            use_curriculum=config["CURRICULUM"],
                            predators=config["PREDATORS"])


    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES_FOR_LR_ANNEALING"]
        )
        return config["LR"] * frac

    def train(rng):

        jax.debug.print('Training')
        # INIT NETWORK
        if config['FULL_ACTION_SPACE']:
            action_space_size = env.action_space(env_params).n
        else:
            action_space_size = 17
        network = ActorCriticRNN(action_space_size, config=config)
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        sparsity_distribution = functools.partial(
            jaxpruner.sparsity_distributions.uniform, sparsity=config["SPARSITY"])
        if config["ALG"] == "MagnitudePruning":
            pruner = jaxpruner.MagnitudePruning(sparsity_distribution_fn=sparsity_distribution)
        elif config["ALG"] == "RandomPruning":
            pruner = jaxpruner.RandomPruning(sparsity_distribution_fn=sparsity_distribution)
        elif config["ALG"] == "SaliencyPruning":
            pruner = jaxpruner.SaliencyPruning(sparsity_distribution_fn=sparsity_distribution)
        elif config["ALG"] == "SET":
            pruner = jaxpruner.SET(sparsity_distribution_fn=sparsity_distribution)
        elif config["ALG"] == "RigL":
            pruner = jaxpruner.RigL(sparsity_distribution_fn=sparsity_distribution)
        else:
            pruner = jaxpruner.NoPruning()
        tx = pruner.wrap_optax(tx)

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, log_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value, action, log_prob = (
                    value.squeeze(0),
                    action.squeeze(0),
                    log_prob.squeeze(0),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info, = env.step(
                    _rng, env_state, action, update_step, env_params
                )

                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_ENV_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done = carry
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - next_done) * gae
                    )
                    return (gae, value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            metric = jax.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            to_log = metric

            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, to_log, update_step)

            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )

            return runner_state, metric

            # Version of _env_step that calls the video plotting wrapper

        def _env_step_viz(runner_state, unused):
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            rng, _rng = jax.random.split(rng)

            # SELECT ACTION
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            obsv, env_state, reward, done, info = env_viz.step(
                _rng, env_state, action, env_params
            )
            # HACK: Add hstate to info for future logging
            info['hidden_state'] = hstate
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info
            )
            runner_state = (
                train_state,
                env_state,
                obsv,
                done,
                hstate,
                rng,
                update_step,
            )
            return runner_state, transition

            # Do one "step" of logging, writing the result to a file.
            # Several steps can be run in series using --logging_steps_per_viz to do long rollouts without hitting memory limits

        def _logging_step(runner_state, unused, logging_threads):
            # Visualization rollouts
            runner_state, traj_batch = jax.lax.scan(
                _env_step_viz, runner_state, None, config['STEPS_PER_VIZ']
            )

            # Finally, log data associated with the visualization runs
            update_step = runner_state[-1]
            hidden_states = traj_batch.info['hidden_state']
            actions = traj_batch.info['action']
            healths = traj_batch.info['health']
            foods = traj_batch.info['food']
            drinks = traj_batch.info['drink']
            energies = traj_batch.info['energy']
            dones = traj_batch.info['done']
            is_sleepings = traj_batch.info['is_sleeping']
            is_restings = traj_batch.info['is_resting']
            player_position_xs = traj_batch.info['player_position_x']
            player_position_ys = traj_batch.info['player_position_y']
            recovers = traj_batch.info['recover']
            hungers = traj_batch.info['hunger']
            thirsts = traj_batch.info['thirst']
            fatigues = traj_batch.info['fatigue']
            light_levels = traj_batch.info['light_level']
            dist_to_melees = traj_batch.info['dist_to_melee_l1']
            melee_on_screen = traj_batch.info['melee_on_screen']
            dist_to_passives = traj_batch.info['dist_to_passive_l1']
            passive_on_screen = traj_batch.info['passive_on_screen']
            dist_to_ranged = traj_batch.info['dist_to_ranged_l1']
            ranged_on_screen = traj_batch.info['ranged_on_screen']
            num_melee_nearby = traj_batch.info['num_melee_nearby']
            num_passives_nearby = traj_batch.info['num_passives_nearby']
            num_ranged_nearby = traj_batch.info['num_ranged_nearby']
            epi_ids = traj_batch.info['episode_id']
            traj_batch.info['hidden_state'] = None

            # Callback function for logging hidden states
            def write_rnn_hstate(hstate, scalars, increment=0):
                # We save to temp files and then append to the target file since numpy apparently cannot write files in append mode for some reason
                for i in range(logging_threads):
                    out_filename_hstates = os.path.join(config['OUTPUT_PATH'], 'hstates_{}_{}.csv'.format(increment, i))
                    temp_filename = os.path.join(config['OUTPUT_PATH'], 'temp.csv')
                    np.savetxt(temp_filename,
                               hstate[:, i, :], delimiter=',')
                    temp_file = open(temp_filename, 'r')
                    out_file_hstates = open(out_filename_hstates, 'a+')
                    out_file_hstates.write(temp_file.read())
                    out_file_hstates.close()
                    temp_file.close()
                    # Then do the same thing for the scalars
                    out_filename_scalars = os.path.join(config['OUTPUT_PATH'], 'scalars_{}_{}.csv'.format(increment, i))
                    np.savetxt(temp_filename,
                               scalars[:, i, :], delimiter=',', fmt='%f',
                               header='action,health,food,drink,energy,done,is_sleeping,is_resting,player_position_x,'
                                      'player_position_y,recover,hunger,thirst,fatigue,light_level,dist_to_melee_l1,'
                                      'melee_on_screen,dist_to_passive_l1,passive_on_screen,dist_to_ranged_l1,'
                                      'ranged_on_screen,num_melee_nearby,num_passives_nearby,num_ranged_nearby,episode_id'
                               )
                    temp_file = open(temp_filename, 'r')
                    out_file_scalars = open(out_filename_scalars, 'a+')
                    out_file_scalars.write(temp_file.read())
                    temp_file.close()
                    out_file_scalars.close()
                    print('Writing log file', out_filename_hstates)

            # Reshape logging arrays and concat
            new_shape = actions.shape + (1,)
            actions = actions.reshape(new_shape)
            healths = healths.reshape(new_shape)
            foods = foods.reshape(new_shape)
            drinks = drinks.reshape(new_shape)
            energies = energies.reshape(new_shape)
            dones = dones.reshape(new_shape)
            is_sleepings = is_sleepings.reshape(new_shape)
            is_restings = is_restings.reshape(new_shape)
            player_position_xs = player_position_xs.reshape(new_shape)
            player_position_ys = player_position_ys.reshape(new_shape)
            recovers = recovers.reshape(new_shape)
            hungers = hungers.reshape(new_shape)
            thirsts = thirsts.reshape(new_shape)
            fatigues = fatigues.reshape(new_shape)
            light_levels = light_levels.reshape(new_shape)
            dist_to_melees = dist_to_melees.reshape(new_shape)
            melee_on_screen = melee_on_screen.reshape(new_shape)
            dist_to_passives = dist_to_passives.reshape(new_shape)
            passive_on_screen = passive_on_screen.reshape(new_shape)
            dist_to_ranged = dist_to_ranged.reshape(new_shape)
            ranged_on_screen = ranged_on_screen.reshape(new_shape)
            num_melee_nearby = num_melee_nearby.reshape(new_shape)
            num_passives_nearby = num_passives_nearby.reshape(new_shape)
            num_ranged_nearby = num_ranged_nearby.reshape(new_shape)
            epi_ids = epi_ids.reshape(new_shape)

            log_array = jnp.concatenate([actions, healths, foods, drinks, energies, dones, is_sleepings, is_restings,
                                         player_position_xs, player_position_ys, recovers, hungers, thirsts, fatigues,
                                         light_levels, dist_to_melees,melee_on_screen, dist_to_passives, passive_on_screen,
                                         dist_to_ranged, ranged_on_screen, num_melee_nearby, num_passives_nearby,
                                         num_ranged_nearby, epi_ids
                                         ], axis=2)
            jax.debug.callback(write_rnn_hstate, hidden_states, log_array, update_step)

            return runner_state, None

        # Func to interleave update steps and plotting
        def _update_plot(runner_state, unused):
            # First, update
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["UPDATES_PER_VIZ"]
            )

            # Log model weights
            def save_weights_callback(weights_flat, iter):
                weight_filename = os.path.join(config['OUTPUT_PATH'], 'weights_{}.csv'.format(iter))
                weight_file = open(weight_filename, 'w')
                for weights_set in weights_flat:
                    if len(weights_set.shape) == 1:
                        continue
                    np.savetxt(weight_file, np.transpose(weights_set), delimiter=',', fmt='%f')
                print('Saving weights in file', weight_filename)

            weights_flat = jax.tree.flatten(runner_state[0].params)
            jax.debug.callback(save_weights_callback, weights_flat[0], runner_state[-1])

            # Can we save the environment state and resume training later?
            #runner_state_copy = runner_state

            # Then do iterations of logging
            runner_state, empty = jax.lax.scan(
                partial(_logging_step, logging_threads = config["LOGGING_THREADS_PER_VIZ"]), runner_state, None, config['LOGGING_STEPS_PER_VIZ']
            )

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            log_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )

        # Copy initial runner state for final validation runs
        initial_runner_state = runner_state

        runner_state, metric = jax.lax.scan(
            _update_plot, runner_state, None, config["NUM_UPDATES"]
        )
        # Do validation rollouts with a fixed random seed
        # Generate rng from validation-specific random seed

        val_rng_key = jax.random.PRNGKey(config["VALIDATION_SEED"])

        rng, _rng = jax.random.split(val_rng_key)

        #RE-INIT FOR VAL RUNS
        obsv, log_state = env.reset(_rng, env_params)

        # init_hstate = ScannedRNN.initialize_carry(
        #     config["NUM_ENVS"], config["LAYER_SIZE"]
        # )

        val_runner_state = (
            runner_state[0],
            log_state,
            obsv,
            jnp.ones((config["NUM_ENVS"]), dtype=bool),
            runner_state[4],
            rng,
            config['VALIDATION_STEP_OFFSET'] + runner_state[-1],
        )

        # Do validation logging iterations
        # TODO separate command line argument for validation logging step count?
        val_runner_state, empty = jax.lax.scan(
            partial(_logging_step, logging_threads = config["LOGGING_THREADS_PER_VIZ_VAL"]), val_runner_state, None, config['LOGGING_STEPS_PER_VIZ_VAL']
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


def run_ppo():

    run = wandb.init(project="sparsity")
    config = wandb.config
    reset_batch_logs()

    else:
        wandb.init(mode="disabled")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree_map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":

    sweep_configuration = {
        "name": "predators_sweep",
        "method": "grid",
        "metric": {"name": "episode_return", "goal": "maximize"},
        "parameters": {
            "ENV_NAME": {
                "values": ["Craftax-Symbolic-v1"]
            },
            "ALG" : {
                "values": ["NoPruning"]
            },
            "PREDATORS": {
                "values": [False]
            },
            "SPARSITY" : {
                "values": [None]
            },
            "NUM_ENVS": {
                "values": [1024] # 1024
            },
            "TOTAL_TIMESTEPS": {
                "values": [1e9]
            },
            "LR": {
                "values": [2e-4]
            },
            "NUM_ENV_STEPS": {
                "values": [64]
            },
            "UPDATE_EPOCHS": {
                "values": [4]
            },
            "NUM_MINIBATCHES": {
                "values": [8]
            },
            "GAMMA": {
                "values": [0.99]
            },
            "GAE_LAMBDA": {
                "values": [0.8]
            },
            "CLIP_EPS": {
                "values": [0.2]
            },
            "ENT_COEF": {
                "values": [0.01]
            },
            "VF_COEF": {
                "values": [0.5]
            },
            "MAX_GRAD_NORM": {
                "values": [1.0]
            },
            "ACTIVATION": {
                "values": ["tanh"]
            },
            "ANNEAL_LR": {
                "values": [True]
            },
            "DEBUG": {
                "values": [True]
            },
            "JIT": {
                "values": [True]
            },
            "SEED": {
                "values": [np.random.randint(2 ** 31)]
            },
            "USE_WANDB": {
                "values": [True]
            },
            "SAVE_POLICY": {
                "values": [True]
            },
            "NUM_REPEATS": {
                "values": [1]
            },
            "LAYER_SIZE": {
                "values": [512]
            },
            "WANDB_PROJECT": {
                "values": [None]  # Adjust with actual project name
            },
            "WANDB_ENTITY": {
                "values": [None]  # Adjust with actual entity name
            },
            "USE_OPTIMISTIC_RESETS": {
                "values": [True]
            },
            "OPTIMISTIC_RESET_RATIO": {
                "values": [16]
            },
            "UPDATES_PER_VIZ": {
                "values": [1024]
            },
            "STEPS_PER_VIZ": {
                "values": [1024]
            },
            "LOGGING_STEPS_PER_VIZ": {
                "values": [8]
            },
            "LOGGING_STEPS_PER_VIZ_VAL": {
                "values": [8]
            },
            "OUTPUT_PATH": {
                "values": ['./output/']
            },
            "FRAMES_PER_FILE": {
                "values": [512]
            },
            "NO_VIDEOS": {
                "values": [True]
            },
            "FULL_ACTION_SPACE": {
                "values": [False]
            },
            "REWARD_FUNCTION": {
                "values": ['foraging']
            },
            "VALIDATION_SEED": {
                "values": [777]
            },
            "VALIDATION_STEP_OFFSET": {
                "values": [0]
            },
            "LOGGING_THREADS_PER_VIZ": {
                "values": [1]
            },
            "LOGGING_THREADS_PER_VIZ_VAL": {
                "values": [1]
            },
            "CURRICULUM": {
                "values": [False]
            },
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="predators_sweep")
    wandb.agent(sweep_id=sweep_id, function=run_ppo)
