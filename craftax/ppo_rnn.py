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
from ml_collections import ConfigDict

from craftax.craftax import craftax_state
from craftax.environment_base.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
<<<<<<< HEAD
    VideoPlotWrapper,
    ReduceActionSpaceWrapper, AppendActionToObsWrapper,
    CurriculumWrapper
=======
    VideoPlotWrapper, ReduceActionSpaceWrapper, AppendActionToObsWrapper,
>>>>>>> foraging
)
from craftax.logz.batch_logging import create_log_dict, batch_log, reset_batch_logs

def parse_args():
    parser = argparse.ArgumentParser(description="Run sparsity PPO.")
    
    # Add arguments for each parameter you want to override
    parser.add_argument("--RUN_NAME", type=str, default="default_run", help="Name of the run")
    parser.add_argument("--ENV_NAME", type=str, default="Craftax-Symbolic-v1", help="Environment name")
    parser.add_argument("--SPARSE_ALG", type=str, default="magnitude", help="Sparsity algorithm")
    parser.add_argument("--GPU_ID", type=int, default=0, help="GPU ID")
    parser.add_argument("--PREDATORS", type=bool, default=True, help="Use predators")
    parser.add_argument("--SPARSITY", type=float, default=0.4, help="Sparsity value")
    parser.add_argument("--NUM_ENVS", type=int, default=1024, help="Number of environments")
    parser.add_argument("--TOTAL_TIMESTEPS", type=float, default=1e9, help="Total timesteps")
    parser.add_argument("--LR", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--NUM_ENV_STEPS", type=int, default=64, help="Number of environment steps")
    parser.add_argument("--UPDATE_EPOCHS", type=int, default=4, help="Number of update epochs")
    parser.add_argument("--NUM_MINIBATCHES", type=int, default=8, help="Number of minibatches")
    parser.add_argument("--GAMMA", type=float, default=0.99, help="Gamma value")
    parser.add_argument("--GAE_LAMBDA", type=float, default=0.8, help="GAE Lambda")
    parser.add_argument("--CLIP_EPS", type=float, default=0.2, help="Clip epsilon")
    parser.add_argument("--ENT_COEF", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--VF_COEF", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--AUX_COEF", type=float, default=0.1, help="Auxiliary coefficient")
    parser.add_argument("--MAX_GRAD_NORM", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--ACTIVATION", type=str, default="tanh", help="Activation function")
    parser.add_argument("--ANNEAL_LR", type=bool, default=True, help="Whether to anneal LR")
    parser.add_argument("--DEBUG", type=bool, default=True, help="Enable debug mode")
    parser.add_argument("--JIT", type=bool, default=True, help="Use JIT compilation")
    parser.add_argument("--ACTION_IN_OBS", type=bool, default=False, help="Include action in observation")
    parser.add_argument("--SEED", type=int, default=np.random.randint(2 ** 31), help="Random seed")
    parser.add_argument("--USE_WANDB", type=bool, default=True, help="Use WandB for logging")
    parser.add_argument("--SAVE_POLICY", type=bool, default=True, help="Save the policy")
    parser.add_argument("--NUM_REPEATS", type=int, default=1, help="Number of repeats")
    parser.add_argument("--LAYER_SIZE", type=int, default=512, help="Layer size")
    parser.add_argument("--WANDB_PROJECT", type=str, default="sparsity_project", help="WandB project name")
    parser.add_argument("--WANDB_ENTITY", type=str, default=None, help="WandB entity name")
    parser.add_argument("--USE_OPTIMISTIC_RESETS", type=bool, default=True, help="Use optimistic resets")
    parser.add_argument("--OPTIMISTIC_RESET_RATIO", type=int, default=16, help="Optimistic reset ratio")
    parser.add_argument("--UPDATES_PER_VIZ", type=int, default=1024, help="Updates per visualization")
    parser.add_argument("--STEPS_PER_VIZ", type=int, default=1024, help="Steps per visualization")
    parser.add_argument("--LOGGING_STEPS_PER_VIZ", type=int, default=8, help="Logging steps per viz")
    parser.add_argument("--LOGGING_STEPS_PER_VIZ_VAL", type=int, default=8, help="Logging steps per viz validation")
    parser.add_argument("--OUTPUT_PATH", type=str, default='./output/', help="Output path")
    parser.add_argument("--FRAMES_PER_FILE", type=int, default=512, help="Frames per file")
    parser.add_argument("--NO_VIDEOS", type=bool, default=True, help="Disable video recording")
    parser.add_argument("--FULL_ACTION_SPACE", type=bool, default=False, help="Use full action space")
    parser.add_argument("--REWARD_FUNCTION", type=str, default='foraging', help="Reward function")
    parser.add_argument("--VALIDATION_SEED", type=int, default=777, help="Validation seed")
    parser.add_argument("--VALIDATION_STEP_OFFSET", type=int, default=0, help="Validation step offset")
    parser.add_argument("--LOGGING_THREADS_PER_VIZ", type=int, default=1, help="Logging threads per viz")
    parser.add_argument("--LOGGING_THREADS_PER_VIZ_VAL", type=int, default=1, help="Logging threads per viz validation")
    parser.add_argument("--CURRICULUM", type=bool, default=False, help="Use curriculum learning")

    return parser.parse_args()

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

        aux = nn.Dense(
<<<<<<< HEAD
        self.config["LAYER_SIZE"],
        kernel_init=orthogonal(2),
        bias_init=constant(0.0),
=======
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
>>>>>>> foraging
        )(embedding)
        aux = nn.relu(aux)
        aux = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(aux)
        aux = nn.relu(aux)
        aux = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            aux
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1), aux


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    deltas_to_start: jnp.ndarray


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
    if config['FEATURELESS_WORLD']:
        static_params.featureless_world = True


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

    if config['ACTION_IN_OBS']:
        env = AppendActionToObsWrapper(env)

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

        # INIT NETWORK
        if config['FULL_ACTION_SPACE']:
            action_space_size = env.action_space(env_params).n
        else:
            action_space_size = 17
        network = ActorCriticRNN(action_space_size, config=config)
        rng, _rng = jax.random.split(rng)
        # We have to do this here because I can't figure out how to wrap the observation_space function (it's not defined in Gymnax, seemingly)
        if config['ACTION_IN_OBS']:
            obs_shape = env.observation_space(env_params).shape[:-1] + (env.observation_space(env_params).shape[-1] + 1,)
        else:
            obs_shape = env.observation_space(env_params).shape
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *obs_shape)
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

        sparsity_config = ConfigDict()
        sparsity_config.sparsity = config["SPARSITY"]
        sparsity_config.algorithm = config["SPARSE_ALG"]
        sparsity_config.dist_type = "erk"
        sparsity_config.update_start_step = 10000 * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
        sparsity_config.update_end_step = 10000 * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
        sparsity_config = sparsity_config.unlock()
        sparse_updater = jaxpruner.create_updater_from_config(sparsity_config)
        tx = sparse_updater.wrap_optax(tx)

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
                hstate, pi, value, aux = network.apply(train_state.params, hstate, ac_in)
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

                # Compute distance to origin for aux loss
                starting_pos = env_state.env_state.player_starting_position[env_state.env_state.player_level]
                # dists_to_start = jnp.linalg.norm(env_state.player_position - starting_pos, ord=1, axis=-1)
                deltas_to_start = env_state.env_state.player_position - starting_pos

                transition = Transition(
                    last_done, action, value, reward, log_prob, last_obs, info, deltas_to_start
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

            train_state = runner_state[0]

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
            _, _, last_val, _ = network.apply(train_state.params, hstate, ac_in)
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
                        _, pi, value, aux = network.apply(
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

                        # Calculate auxiliary loss (predict distance to origin)
                        # Simple L2
                        aux_loss = jnp.square(aux - traj_batch.deltas_to_start).mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config["AUX_COEF"] * aux_loss
                        )

                        return total_loss, (value_loss, loss_actor, entropy, aux_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    updated_params = sparse_updater.pre_forward_update(
                        train_state.params, train_state.opt_state
                    )

                    total_loss, grads = grad_fn(
                        updated_params, init_hstate, traj_batch, advantages, targets
                    )

                    train_state = train_state.apply_gradients(grads=grads)

                    post_grad_params = sparse_updater.post_gradient_update(train_state.params, train_state.opt_state)

                    return train_state.replace(params=post_grad_params), total_loss

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

            # TODO figure out how to log loss data, it's not syncronized with env steps so this is annoying
            #traj_batch.info['total_loss'] = loss_info[0].mean()
            #traj_batch.info['aux_loss'] = loss_info[1][-1].mean()

            metric = jax.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            metric["total_sparsity"] = jaxpruner.utils.summarize_sparsity(
                train_state.params, only_total_sparsity=True
            )["_total_sparsity"]

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
            hstate, pi, value, aux = network.apply(train_state.params, hstate, ac_in)
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

            # Compute distance to origin for aux loss
            starting_pos = env_state.env_state.player_starting_position[env_state.env_state.player_level]
            deltas_to_start = env_state.env_state.player_position - starting_pos

            # Add hstate and other non-env metrics to info so they can be logged
            info['hidden_state'] = hstate
            info['pred_delta'] = aux
            info['delta'] = deltas_to_start

            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info, deltas_to_start,
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
            # Null this for memory savings
            traj_batch.info['hidden_state'] = None

            # Add new logging fields here
            fields_to_log = ['health','food','drink','energy','done','is_sleeping','is_resting','player_position_x',
                                      'player_position_y','recover','hunger','thirst','fatigue','light_level','dist_to_melee_l1',
                                      'melee_on_screen','dist_to_passive_l1','passive_on_screen','dist_to_ranged_l1',
                                      'ranged_on_screen','num_melee_nearby','num_passives_nearby','num_ranged_nearby','delta',
                                      'pred_delta','episode_id']

            # Callback function for logging hidden states
            def write_rnn_hstate(hstate, scalars, increment=0):
<<<<<<< HEAD

                run_out_path = os.path.join(config['OUTPUT_PATH'], wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
=======
>>>>>>> foraging
                # Assemble header for the scalar file(s)
                scalar_file_header = 'action'
                for key in fields_to_log:
                    scalar_file_header += ',' + key

                # We save to temp files and then append to the target file since numpy apparently cannot write files in append mode for some reason
                for i in range(logging_threads):
                    out_filename_hstates = os.path.join(run_out_path, 'hstates_{}_{}.csv'.format(increment, i))
                    temp_filename = os.path.join(run_out_path, 'temp.csv')
                    np.savetxt(temp_filename,
                               hstate[:, i, :], delimiter=',')
                    temp_file = open(temp_filename, 'r')
                    out_file_hstates = open(out_filename_hstates, 'a+')
                    out_file_hstates.write(temp_file.read())
                    out_file_hstates.close()
                    temp_file.close()
                    # Then do the same thing for the scalars
                    out_filename_scalars = os.path.join(run_out_path, 'scalars_{}_{}.csv'.format(increment, i))
                    np.savetxt(temp_filename,
                               scalars[:, i, :], delimiter=',', fmt='%f',
                               header=scalar_file_header
                               )
                    temp_file = open(temp_filename, 'r')
                    out_file_scalars = open(out_filename_scalars, 'a+')
                    out_file_scalars.write(temp_file.read())
                    temp_file.close()
                    out_file_scalars.close()
                    print('Writing log file', out_filename_hstates)


            # Add the specified field to the logging array
            # Also assembles the header for the log file itself
            def add_field_to_log_array(info_dict, log_array, field_key):
                field_value = info_dict[field_key]
                if len(field_value.shape) < 3:
                    new_shape = field_value.shape + (1,)
                    field_value = field_value.reshape(new_shape)
                else:
                    field_value = field_value.squeeze()
                log_array = jnp.concatenate([log_array, field_value], axis=2)

                return log_array

            # Assemble logging variable array
            log_array = traj_batch.info['action'].reshape(traj_batch.info['action'].shape + (1,))
            # Yes this is a for loop in the JAX code but this stuff was getting done in serial before anyway and it's cheap operations
            for field_to_log in fields_to_log:
                log_array = add_field_to_log_array(traj_batch.info, log_array, field_to_log)

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
                run_out_path = os.path.join(config['OUTPUT_PATH'], wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                weight_filename = os.path.join(run_out_path, 'weights_{}.csv'.format(iter))
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


def run_ppo(config):

    reset_batch_logs()

    if not config["JIT"]:
        jax.config.update("jax_disable_jit", True)
        print('JIT disabled')

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config), device=jax.devices()[config['GPU_ID']])
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

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
<<<<<<< HEAD
=======
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--total_timesteps", type=int, default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--aux_coef", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument('--updates_per_viz', type=int, default=1024)
    parser.add_argument('--steps_per_viz', type=int, default=1024)
    parser.add_argument('--logging_steps_per_viz', type=int, default=8)
    parser.add_argument('--logging_steps_per_viz_val', type=int, default=8)
    parser.add_argument('--output_path', type=str, default='./output/')
    parser.add_argument('--frames_per_file', type=int, default=512)
    parser.add_argument('--no_videos', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--full_action_space', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reward_function', type=str, default='foraging')
    parser.add_argument('--validation_seed', type=int, default=777)
    parser.add_argument('--validation_step_offset', type=int, default=0)
    parser.add_argument('--logging_threads_per_viz',type=int, default=1)
    parser.add_argument('--logging_threads_per_viz_val', type=int, default=1)
    parser.add_argument('--action_in_obs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--featureless_world', action=argparse.BooleanOptionalAction, default=False)
>>>>>>> foraging

    args = parse_args()

    wandb.init(
        project=args.WANDB_PROJECT,
        entity=args.WANDB_ENTITY,
        config={
            "ENV_NAME": args.ENV_NAME,
            "SPARSE_ALG": args.SPARSE_ALG,
            "GPU_ID": args.GPU_ID,
            "PREDATORS": args.PREDATORS,
            "SPARSITY": args.SPARSITY,
            "NUM_ENVS": args.NUM_ENVS,
            "TOTAL_TIMESTEPS": args.TOTAL_TIMESTEPS,
            "LR": args.LR,
            "NUM_ENV_STEPS": args.NUM_ENV_STEPS,
            "UPDATE_EPOCHS": args.UPDATE_EPOCHS,
            "NUM_MINIBATCHES": args.NUM_MINIBATCHES,
            "GAMMA": args.GAMMA,
            "GAE_LAMBDA": args.GAE_LAMBDA,
            "CLIP_EPS": args.CLIP_EPS,
            "ENT_COEF": args.ENT_COEF,
            "VF_COEF": args.VF_COEF,
            "AUX_COEF": args.AUX_COEF,
            "MAX_GRAD_NORM": args.MAX_GRAD_NORM,
            "ACTIVATION": args.ACTIVATION,
            "ANNEAL_LR": args.ANNEAL_LR,
            "DEBUG": args.DEBUG,
            "JIT": args.JIT,
            "ACTION_IN_OBS": args.ACTION_IN_OBS,
            "SEED": args.SEED,
            "USE_WANDB": args.USE_WANDB,
            "SAVE_POLICY": args.SAVE_POLICY,
            "NUM_REPEATS": args.NUM_REPEATS,
            "LAYER_SIZE": args.LAYER_SIZE,
            "USE_OPTIMISTIC_RESETS": args.USE_OPTIMISTIC_RESETS,
            "OPTIMISTIC_RESET_RATIO": args.OPTIMISTIC_RESET_RATIO,
            "UPDATES_PER_VIZ": args.UPDATES_PER_VIZ,
            "STEPS_PER_VIZ": args.STEPS_PER_VIZ,
            "LOGGING_STEPS_PER_VIZ": args.LOGGING_STEPS_PER_VIZ,
            "LOGGING_STEPS_PER_VIZ_VAL": args.LOGGING_STEPS_PER_VIZ_VAL,
            "OUTPUT_PATH": args.OUTPUT_PATH,
            "FRAMES_PER_FILE": args.FRAMES_PER_FILE,
            "NO_VIDEOS": args.NO_VIDEOS,
            "FULL_ACTION_SPACE": args.FULL_ACTION_SPACE,
            "REWARD_FUNCTION": args.REWARD_FUNCTION,
            "VALIDATION_SEED": args.VALIDATION_SEED,
            "VALIDATION_STEP_OFFSET": args.VALIDATION_STEP_OFFSET,
            "LOGGING_THREADS_PER_VIZ": args.LOGGING_THREADS_PER_VIZ,
            "LOGGING_THREADS_PER_VIZ_VAL": args.LOGGING_THREADS_PER_VIZ_VAL,
            "CURRICULUM": args.CURRICULUM
        },
        name=args.RUN_NAME
    )

    # Run PPO or your main training function
    run_ppo(wandb.config)