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
import chex

from flax.training import orbax_utils
from matplotlib import pyplot as plt, animation
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict, Any
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
    VideoPlotWrapper,
    ReduceActionSpaceWrapper, AppendActionToObsWrapper, AppendActionToObsWrapper,
    CurriculumWrapper
)
from craftax.logz.batch_logging import create_log_dict, batch_log, reset_batch_logs
from craftax.models import BatchRenorm


def parse_args():
    parser = argparse.ArgumentParser(description="Run PQN LSTM.")
    parser.add_argument("--prune_step", type=int, default=20000, help="Step to prune")
    parser.add_argument('--featureless_world', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_name", type=str, default="default_run", help="Name of the run")
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1", help="Environment name")
    parser.add_argument("--sparse_alg", type=str, default="magnitude", help="options, magnitude, no_prune, saliency, random")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--predators", type=bool, default=True, help="Use predators")
    parser.add_argument("--max_cows", type=int, default=72, help="Maximum number of cows that can exist at a time")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
    parser.add_argument("--total_timesteps", type=float, default=3e9, help="Total timesteps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_env_steps", type=int, default=64, help="Number of environment steps")
    parser.add_argument("--update_epochs", type=int, default=4, help="Number of update epochs")
    parser.add_argument("--num_minibatches", type=int, default=8, help="Number of minibatches")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma value")
    parser.add_argument("--aux_coef", type=float, default=0.1, help="Auxiliary coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--activation", type=str, default="tanh", help="Activation function")
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--action_in_obs', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=np.random.randint(2 ** 31), help="Random seed")
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_policy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num_repeats", type=int, default=1, help="Number of repeats")
    parser.add_argument("--layer_size", type=int, default=512, help="Layer size")
    parser.add_argument("--wandb_project", type=str, default="sparsity_project", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity name")
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16, help="Optimistic reset ratio")
    parser.add_argument("--updates_per_viz", type=int, default=2048, help="Updates per visualization")
    parser.add_argument("--steps_per_viz", type=int, default=1024, help="Steps per visualization")
    parser.add_argument("--logging_steps_per_viz", type=int, default=8, help="Logging steps per viz")
    parser.add_argument("--logging_steps_per_viz_val", type=int, default=8, help="Logging steps per viz validation")
    parser.add_argument("--output_path", type=str, default='./output/', help="Output path")
    parser.add_argument("--frames_per_file", type=int, default=512, help="Frames per file")
    parser.add_argument('--no_videos', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--full_action_space', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reward_function", type=str, default='foraging', help="Reward function")
    parser.add_argument("--validation_seed", type=int, default=777, help="Validation seed")
    parser.add_argument("--validation_step_offset", type=int, default=0, help="Validation step offset")
    parser.add_argument("--logging_threads_per_viz", type=int, default=1, help="Logging threads per viz")
    parser.add_argument("--logging_threads_per_viz_val", type=int, default=1, help="Logging threads per viz validation")
    parser.add_argument("--curriculum", type=bool, default=False, help="Use curriculum learning")
    parser.add_argument("--map_size", type=int, default=96, help="The side length for the map")
    parser.add_argument("--directional_vision", action=argparse.BooleanOptionalAction, default=False, help="Turn on directional vision cones")
    parser.add_argument("--EPS_START", type=float, default=0.1, help="Initial epsilon")
    parser.add_argument("--EPS_FINISH", type=float, default=0.005, help="Final epsilon")
    parser.add_argument("--EPS_DECAY", type=float, default=0.2, help="Epsilon decay")
    parser.add_argument("--TOTAL_TIMESTEPS_DECAY", type=int, default=1e9, help="Total timesteps for decay")
    parser.add_argument("--NUM_STEPS", type=int, default=8, help="steps per environment in each update")
    parser.add_argument("--LR_LINEAR_DECAY", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--REW_SCALE", type=float, default=1.0, help="Reward scale")
    parser.add_argument("--Q_LAMBDA", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--LAMBDA", type=float, default=0, help="Lambda value")
    parser.add_argument("--LOG_ACHIEVEMENTS", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--WANDB_LOG_INTERVAL", type=int, default=100, help="WandB log interval")
    parser.add_argument("--TEST_DURING_TRAINING", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--TEST_INTERVAL", type=float, default=0.01, help="Test interval in terms of total updates")
    parser.add_argument("--TEST_NUM_ENVS", type=int, default=512, help="Number of environments to test")
    parser.add_argument("--TEST_NUM_STEPS", type=int, default=10000, help="Number of steps to test")
    parser.add_argument("--EPS_TEST", type=float, default=0.00, help="For greedy policy")
    parser.add_argument("--NUM_EPOCHS", type=int, default=1, help="Number of epochs")
    parser.add_argument("--WANDB_MODE", type=str, default="online", help="WandB mode")
    parser.add_argument("--NORM_TYPE", type=str, default="layer_norm", help="layer_norm or batch_norm")
    parser.add_argument("--MEMORY_WINDOW", type=int, default=0, help="steps of previous episode added in the rnn training horizon")
    parser.add_argument("--NUM_RNN_LAYERS", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("--NUM_LAYERS", type=int, default=1, help="Number of layers")
    return parser.parse_args()


class ScannedRNN(nn.Module):

    @partial(
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
        hidden_size = rnn_state[0].shape[-1]
        init_rnn_state = self.initialize_carry(hidden_size, *resets.shape)
        rnn_state = jax.tree_util.tree_map(
            lambda init, old: jnp.where(resets[:, np.newaxis], init, old),
            init_rnn_state,
            rnn_state,
        )

        new_rnn_state, y = nn.OptimizedLSTMCell(hidden_size)(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_size, *batch_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.OptimizedLSTMCell(hidden_size, parent=None).initialize_carry(
            jax.random.PRNGKey(0), (*batch_size, hidden_size)
        )

class RNNQNetwork(nn.Module):
    action_dim: int
    layer_size: int
    hidden_size: int = 512
    num_layers: int = 4
    num_rnn_layers: int = 1
    norm_input: bool = False
    norm_type: str = "layer_norm"
    dueling: bool = False
    add_last_action: bool = False

    @nn.compact
    def __call__(self, hidden, x, done, last_action, train: bool = False):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x

        if self.norm_input:
            x = BatchRenorm(use_running_average=not train)(x)
        else:
            # dummy normalize input in any case for global compatibility
            x_dummy = BatchRenorm(use_running_average=not train)(x)

        for l in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = normalize(x)
            x = nn.relu(x)

        # add last action to the input of the rnn
        if self.add_last_action:
            last_action = jax.nn.one_hot(last_action, self.action_dim)
            x = jnp.concatenate([x, last_action], axis=-1)

        new_hidden = []
        for i in range(self.num_rnn_layers):
            rnn_in = (x, done)
            hidden_aux, x = ScannedRNN()(hidden[i], rnn_in)
            new_hidden.append(hidden_aux)

        q_vals = nn.Dense(self.action_dim)(x)

        aux = nn.Dense(
            self.layer_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(x)
        aux = nn.relu(aux)
        aux = nn.Dense(
            self.layer_size,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(aux)
        aux = nn.relu(aux)
        aux = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            aux
        )

        return new_hidden, q_vals, aux

    def initialize_carry(self, *batch_size):
        return [
            ScannedRNN.initialize_carry(self.hidden_size, *batch_size)
            for _ in range(self.num_rnn_layers)
        ]

@chex.dataclass(frozen=True)
class Transition:
    last_hs: chex.Array
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    last_done: chex.Array
    last_action: chex.Array
    next_obs: chex.Array
    q_val: chex.Array
    info: jnp.ndarray
    deltas_to_start: jnp.ndarray


class CustomTrainState(TrainState):
    batch_stats: Any
    timesteps: int = 0
    n_updates: int = 0
    grad_steps: int = 0

def make_train(config):
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"] // config['UPDATES_PER_VIZ']
    )

    config["NUM_UPDATES_DECAY"] = (
            config["TOTAL_TIMESTEPS_DECAY"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    assert (config["NUM_STEPS"] * config["NUM_ENVS"]) % config[
        "NUM_MINIBATCHES"
    ] == 0, "NUM_MINIBATCHES must divide NUM_STEPS*NUM_ENVS"

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
    if config['PREDATORS']:
        static_params.predators = True
    static_params.map_size = (config['MAP_SIZE'],config['MAP_SIZE'])
    static_params.directional_vision = config['DIRECTIONAL_VISION']

    static_params.max_passive_mobs = config['MAX_COWS']

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

    # epsilon-greedy exploration
    def eps_greedy_exploration(rng, q_vals, eps):
        rng_a, rng_e = jax.random.split(
            rng
        )  # a key for sampling random actions and one for picking
        greedy_actions = jnp.argmax(q_vals, axis=-1)
        chosed_actions = jnp.where(
            jax.random.uniform(rng_e, greedy_actions.shape)
            < eps,  # pick the actions that should be random
            jax.random.randint(
                rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
            ),  # sample random actions,
            greedy_actions,
        )
        return chosed_actions

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

    def train(rng):

        # INIT NETWORK
        if config['FULL_ACTION_SPACE']:
            action_space_size = env.action_space(env_params).n
        else:
            action_space_size = 17

        if "pixel" in config["ENV_NAME"]:
            is_symbolic = False
        else:
            is_symbolic = True

        network = RNNQNetwork(
            action_dim=action_space_size,
            hidden_size=config.get("HIDDEN_SIZE", 128),
            num_layers=config.get("NUM_LAYERS", 2),
            num_rnn_layers=config.get("NUM_RNN_LAYERS", 1),
            norm_type=config["NORM_TYPE"],
            norm_input=config.get("NORM_INPUT", False),
            add_last_action=config.get("ADD_LAST_ACTION", False),
            layer_size=config["LAYER_SIZE"],
        )


        original_rng = rng[0]

        eps_scheduler = optax.linear_schedule(
            config["EPS_START"],
            config["EPS_FINISH"],
            (config["EPS_DECAY"]) * config["NUM_UPDATES_DECAY"],
        )

        lr_scheduler = optax.linear_schedule(
            init_value=config["LR"],
            end_value=1e-20,
            transition_steps=(config["NUM_UPDATES_DECAY"])
                             * config["NUM_MINIBATCHES"]
                             * config["NUM_EPOCHS"],
        )
        lr = lr_scheduler if config.get("LR_LINEAR_DECAY", False) else config["LR"]

        def create_agent(rng):
            init_x = (
                jnp.zeros(
                    (1, 1, *env.observation_space(env_params).shape)
                ),  # (time_step, batch_size, obs_size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
                jnp.zeros((1, 1)),  # (time_step, batch size)
            )  # (obs, dones, last_actions)
            init_hs = network.initialize_carry(1)  # (batch_size, hidden_dim)
            network_variables = network.init(rng, init_hs, *init_x, train=False)
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.radam(learning_rate=lr),
            )

            train_state = CustomTrainState.create(
                apply_fn=network.apply,
                params=network_variables["params"],
                batch_stats=network_variables["batch_stats"],
                tx=tx,
            )
            return train_state

        rng, _rng = jax.random.split(rng)
        train_state = create_agent(rng)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            train_state, memory_transitions, expl_state, test_metrics, rng = (
                runner_state
            )

            # SAMPLE PHASE
            def _step_env(carry, _):
                hs, last_obs, last_done, last_action, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)

                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

                new_hs, q_vals, aux = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(
                    axis=0
                )  # (num_envs, num_actions) remove the time dim

                _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
                eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
                new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

                new_obs, new_env_state, reward, new_done, info = env.step(
                    rng_s, env_state, new_action, env_params
                )

                # Compute distance to origin for aux loss
                starting_pos = env_state.env_state.player_starting_position[env_state.env_state.player_level]
                # dists_to_start = jnp.linalg.norm(env_state.player_position - starting_pos, ord=1, axis=-1)
                deltas_to_start = env_state.env_state.player_position - starting_pos

                transition = Transition(
                    last_hs=hs,
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    last_done=last_done,
                    last_action=last_action,
                    q_val=q_vals,
                    next_obs=new_obs,
                    info=info,
                    deltas_to_start=deltas_to_start,
                )
                return (new_hs, new_obs, new_done, new_action, new_env_state, rng), (
                    transition,
                    info,
                )

            # step the env
            rng, _rng = jax.random.split(rng)
            (*expl_state, rng), (transitions, infos) = jax.lax.scan(
                _step_env,
                (*expl_state, _rng),
                None,
                config["NUM_STEPS"],
            )
            expl_state = tuple(expl_state)

            train_state = train_state.replace(
                timesteps=train_state.timesteps
                          + config["NUM_STEPS"] * config["NUM_ENVS"]
            )  # update timesteps count

            # insert the transitions into the memory
            memory_transitions = jax.tree_util.tree_map(
                lambda x, y: jnp.concatenate([x[config["NUM_STEPS"]:], y], axis=0),
                memory_transitions,
                transitions,
            )

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch):
                    # minibatch shape: num_steps, batch_size, ...
                    # with batch_size = num_envs/num_minibatches

                    train_state, rng = carry
                    hs = jax.tree_util.tree_map(lambda x: x[0],
                                                minibatch.last_hs)  # hs of oldest step (batch_size, hidden_size)
                    agent_in = (
                        minibatch.obs,
                        minibatch.last_done,
                        minibatch.last_action,
                    )

                    def _compute_targets(last_q, q_vals, reward, done):
                        def _get_target(lambda_returns_and_next_q, rew_q_done):
                            reward, q, done = rew_q_done
                            lambda_returns, next_q = lambda_returns_and_next_q
                            target_bootstrap = (
                                    reward + config["GAMMA"] * (1 - done) * next_q
                            )
                            delta = lambda_returns - next_q
                            lambda_returns = (
                                    target_bootstrap
                                    + config["GAMMA"] * config["LAMBDA"] * delta
                            )
                            lambda_returns = (1 - done) * lambda_returns + done * reward
                            next_q = jnp.max(q, axis=-1)
                            return (lambda_returns, next_q), lambda_returns

                        lambda_returns = (
                                reward[-1] + config["GAMMA"] * (1 - done[-1]) * last_q
                        )
                        last_q = jnp.max(q_vals[-1], axis=-1)
                        _, targets = jax.lax.scan(
                            _get_target,
                            (lambda_returns, last_q),
                            jax.tree_util.tree_map(lambda x: x[:-1], (reward, q_vals, done)),
                            reverse=True,
                        )
                        targets = jnp.concatenate([targets, lambda_returns[np.newaxis]])
                        return targets

                    def _loss_fn(params):
                        (_, q_vals, aux), updates = partial(
                            network.apply, train=True, mutable=["batch_stats"]
                        )(
                            {"params": params, "batch_stats": train_state.batch_stats},
                            hs,
                            *agent_in,
                        )  # (num_steps, batch_size, num_actions)

                        # lambda returns are computed using NUM_STEPS as the horizon, and optimizing from t=0 to NUM_STEPS-1
                        target_q_vals = jax.lax.stop_gradient(q_vals)
                        last_q = target_q_vals[-1].max(axis=-1)
                        target = _compute_targets(
                            last_q,  # q_vals at t=NUM_STEPS-1
                            target_q_vals[:-1],
                            minibatch.reward[:-1],
                            minibatch.done[:-1],
                        ).reshape(
                            -1
                        )  # (num_steps-1*batch_size,)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(
                            axis=-1
                        )  # (num_steps, num_agents, batch_size,)
                        chosen_action_qvals = chosen_action_qvals[:-1].reshape(
                            -1
                        )  # (num_steps-1*batch_size,)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        # Calculate auxiliary loss (predict distance to origin)
                        # Simple L2
                        aux_loss = jnp.square(aux - minibatch.deltas_to_start).mean()

                        total_loss = loss + config["AUX_COEF"] * aux_loss

                        return total_loss, (updates, chosen_action_qvals, loss, aux_loss)

                    (total_loss, (updates, qvals, critic_loss, aux_loss)), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    train_state = train_state.replace(
                        grad_steps=train_state.grad_steps + 1,
                        batch_stats=updates["batch_stats"],
                    )
                    return (train_state, rng), (total_loss, qvals, critic_loss, aux_loss)

                def preprocess_transition(x, rng):
                    # x: (num_steps, num_envs, ...)
                    x = jax.random.permutation(
                        rng, x, axis=1
                    )  # shuffle the transitions
                    x = x.reshape(
                        x.shape[0], config["NUM_MINIBATCHES"], -1, *x.shape[2:]
                    )  # num_steps, minibatches, batch_size/num_minbatches,
                    x = jnp.swapaxes(
                        x, 0, 1
                    )  # (minibatches, num_steps, batch_size/num_minbatches, ...)
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng),
                    memory_transitions,
                )  # num_minibatches, num_steps+memory_window, batch_size/num_minbatches, ...

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (total_loss, qvals, critic_loss, aux_loss) = jax.lax.scan(
                    _learn_phase, (train_state, rng), minibatches
                )

                return (train_state, rng), (total_loss, qvals, critic_loss, aux_loss)

            rng, _rng = jax.random.split(rng)
            (train_state, rng), (total_loss, qvals, critic_loss, aux_loss) = jax.lax.scan(
                _learn_epoch, (train_state, rng), None, config["NUM_EPOCHS"]
            )

            train_state = train_state.replace(n_updates=train_state.n_updates + 1)
            metrics = {
                "env_step": train_state.timesteps,
                "update_steps": train_state.n_updates,
                "grad_steps": train_state.grad_steps,
                "total_loss": total_loss.mean(),
                "aux_loss": aux_loss.mean(),
                "td_loss": critic_loss.mean(),
                "qvals": qvals.mean(),
                "eps:": eps_scheduler(train_state.n_updates),
            }
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                          / infos["returned_episode"].sum(),
                infos,
            )
            metrics.update(done_infos)

            if config.get("TEST_DURING_TRAINING", False):
                rng, _rng = jax.random.split(rng)
                test_metrics = jax.lax.cond(
                    train_state.n_updates
                    % int(config["NUM_UPDATES"] * config["TEST_INTERVAL"])
                    == 0,
                    lambda _: get_test_metrics(train_state, _rng),
                    lambda _: test_metrics,
                    operand=None,
                )
                metrics.update({f"test/{k}": v for k, v in test_metrics.items()})

            # remove achievement metrics if not logging them
            if not config.get("LOG_ACHIEVEMENTS", False):
                metrics = {
                    k: v for k, v in metrics.items() if "achievement" not in k.lower()
                }

            # report on wandb if required
            if config["WANDB_MODE"] != "disabled":

                def callback(metrics, original_rng):
                    to_log = create_log_dict(metrics, config)
                    metrics.update({k: v for k, v in to_log.items()})
                    batch_log(metrics["update_steps"], metrics, config)

                    for k,v in metrics.items():
                        print(f"{k}: {v}")

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (
                train_state,
                memory_transitions,
                tuple(expl_state),
                test_metrics,
                rng,
            )

            return runner_state, None

        def get_test_metrics(train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _greedy_env_step(step_state, _):
                hs, last_obs, last_done, last_action, env_state, rng = step_state
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
                _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
                _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
                new_hs, q_vals, aux = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    hs,
                    _obs,
                    _done,
                    _last_action,
                    train=False,
                )  # (num_envs, hidden_size), (1, num_envs, num_actions)
                q_vals = q_vals.squeeze(
                    axis=0
                )  # (num_envs, num_actions) remove the time dim
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(rng_a, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                step_state = (new_hs, new_obs, new_done, new_action, new_env_state, rng)
                return step_state, info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)
            init_done = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=bool)
            init_action = jnp.zeros((config["TEST_NUM_ENVS"]), dtype=int)
            init_hs = network.initialize_carry(
                config["TEST_NUM_ENVS"]
            )  # (n_envs, hs_size)
            step_state = (
                init_hs,
                init_obs,
                init_done,
                init_action,
                env_state,
                _rng,
            )
            step_state, infos = jax.lax.scan(
                _greedy_env_step, step_state, None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                          / infos["returned_episode"].sum(),
                infos,
            )
            return done_infos

        rng, _rng = jax.random.split(rng)
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        obs, env_state = env.reset(_rng, env_params)
        init_dones = jnp.zeros((config["NUM_ENVS"]), dtype=bool)
        init_action = jnp.zeros((config["NUM_ENVS"]), dtype=int)
        init_hs = network.initialize_carry(config["NUM_ENVS"])

        expl_state = (init_hs, obs, init_dones, init_action, env_state)

        # step randomly to have the initial memory window
        def _random_step(carry, _):
            hs, last_obs, last_done, last_action, env_state, rng = carry
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
            _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
            _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)
            new_hs, q_vals, aux = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                hs,
                _obs,
                _done,
                _last_action,
                train=False,
            )  # (num_envs, hidden_size), (1, num_envs, num_actions)
            q_vals = q_vals.squeeze(
                axis=0
            )  # (num_envs, num_actions) remove the time dim
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], 1.0)  # random actions
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)
            new_obs, new_env_state, reward, new_done, info = env.step(
                rng_s, env_state, new_action, env_params
            )

            # Compute distance to origin for aux loss
            starting_pos = env_state.env_state.player_starting_position[env_state.env_state.player_level]
            # dists_to_start = jnp.linalg.norm(env_state.player_position - starting_pos, ord=1, axis=-1)
            deltas_to_start = env_state.env_state.player_position - starting_pos

            transition = Transition(
                last_hs=hs,
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                q_val=q_vals,
                info=info,
                deltas_to_start=deltas_to_start,
                next_obs=new_obs,
            )
            return (
                new_hs,
                new_obs,
                new_done,
                new_action,
                new_env_state,
                rng,
            ), transition

        def _env_step_viz(runner_state, unused):
            # train_state, expl_state, test_metrics, rng = runner_state
            train_state, memory_transitions, expl_state, test_metrics, rng = runner_state
            hs, last_obs, last_done, last_action, env_state = expl_state
            rng, rng_a  = jax.random.split(rng)

            _obs = last_obs[np.newaxis]  # (1 (dummy time), num_envs, obs_size)
            _done = last_done[np.newaxis]  # (1 (dummy time), num_envs)
            _last_action = last_action[np.newaxis]  # (1 (dummy time), num_envs)

            # select action using epsilon greedy
            new_hs, q_vals, aux = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                hs,
                _obs,
                _done,
                _last_action,
                train=False,
            )  # (num_envs, hidden_size), (1, num_envs, num_actions)

            # different eps for each env
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
            q_vals = q_vals.squeeze()
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

            # step env
            rng, rng_s  = jax.random.split(rng)
            new_obs, new_env_state, reward, new_done, info = env_viz.step(
                rng_s, env_state, new_action, env_params
            )

            # Compute distance to origin for aux loss
            starting_pos = env_state.env_state.player_starting_position[env_state.env_state.player_level]
            deltas_to_start = env_state.env_state.player_position - starting_pos

            # use the values in new_action to get the q_vals
            q_vals_action_taken = jnp.take_along_axis(q_vals, jnp.expand_dims(new_action, axis=-1), axis=-1).squeeze(axis=-1)

            info['value'] = q_vals_action_taken
            info['hidden_state'] = new_hs
            info['pred_delta'] = aux
            info['delta'] = deltas_to_start

            transition = Transition(
                last_hs=hs,
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=new_done,
                last_done=last_done,
                last_action=last_action,
                q_val=q_vals,
                info=info,
                deltas_to_start=deltas_to_start,
                next_obs=new_obs,
            )

            return runner_state, transition

        def _logging_step(runner_state, unused, logging_threads):

            runner_state, minibatch = jax.lax.scan(
                _env_step_viz, runner_state, None, config['STEPS_PER_VIZ']
            )

            hidden_states = minibatch.info['hidden_state']


            # Null this for memory savings
            minibatch.info['hidden_state'] = None

            # Add new logging fields here
            fields_to_log = ['health', 'food', 'drink', 'energy', 'done', 'is_sleeping', 'is_resting',
                             'player_position_x',
                             'player_position_y', 'recover', 'hunger', 'thirst', 'fatigue', 'light_level',
                             'dist_to_melee_l1',
                             'melee_on_screen', 'dist_to_passive_l1', 'passive_on_screen', 'dist_to_ranged_l1',
                             'ranged_on_screen', 'num_melee_nearby', 'num_passives_nearby', 'num_ranged_nearby',
                             'delta',
                             'pred_delta', 'num_monsters_killed', 'has_sword', 'has_pick', 'held_iron', 'value',
                             'episode_id']

            # Callback function for logging the scalars
            def write_rnn_hstate(hstate, scalars, increment=0):

                cell_state, hidden_state = hstate[0]

                header_field_names = ['health', 'food', 'drink', 'energy', 'done', 'is_sleeping', 'is_resting',
                                      'player_position_x',
                                      'player_position_y', 'recover', 'hunger', 'thirst', 'fatigue', 'light_level',
                                      'dist_to_melee_l1',
                                      'melee_on_screen', 'dist_to_passive_l1', 'passive_on_screen', 'dist_to_ranged_l1',
                                      'ranged_on_screen', 'num_melee_nearby', 'num_passives_nearby',
                                      'num_ranged_nearby', 'delta_x',
                                      'delta_y', 'pred_delta_x', 'pred_delta_y', 'num_monsters_killed', 'has_sword',
                                      'has_pick', 'held_iron', 'value', 'episode_id']

                run_out_path = os.path.join(config['OUTPUT_PATH'], wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                # Assemble header for the scalar file(s)
                scalar_file_header = 'action'
                for key in header_field_names:
                    scalar_file_header += ',' + key

                # We save to temp files and then append to the target file since numpy apparently cannot write files in append mode for some reason
                for i in range(logging_threads):
                    out_filename_hstates = os.path.join(run_out_path, 'hstates_{}_{}.csv'.format(increment, i))
                    temp_filename = os.path.join(run_out_path, 'temp.csv')
                    np.savetxt(temp_filename,
                               cell_state[:, i, :], delimiter=',')
                    temp_file = open(temp_filename, 'r')
                    out_file_hstates = open(out_filename_hstates, 'a+')
                    out_file_hstates.write(temp_file.read())
                    out_file_hstates.close()
                    temp_file.close()
                    # Then do the same thing for the scalars
                    out_filename_scalars = os.path.join(run_out_path, 'scalars_{}_{}.csv'.format(increment, i))
                    temp_filename = os.path.join(run_out_path, 'temp.csv')
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
            log_array = minibatch.info['action'].reshape(minibatch.info['action'].shape + (1,))

            # Yes this is a for loop in the JAX code but this stuff was getting done in serial before anyway and it's cheap operations
            for field_to_log in fields_to_log:
                log_array = add_field_to_log_array(minibatch.info, log_array, field_to_log)

            jax.debug.callback(write_rnn_hstate, hidden_states, log_array, runner_state[0].n_updates)

            return runner_state, None

        # Func to interleave update steps and plotting
        def _update_plot(runner_state, unused):
            # First, update
            runner_state, metrics = jax.lax.scan(
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
            jax.debug.callback(save_weights_callback, weights_flat[0], runner_state[0].n_updates)

            # Can we save the environment state and resume training later?
            # runner_state_copy = runner_state

            # Then do iterations of logging
            runner_state, empty = jax.lax.scan(
                partial(_logging_step, logging_threads=config["LOGGING_THREADS_PER_VIZ"]), runner_state, None,
                config['LOGGING_STEPS_PER_VIZ']
            )

            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        (*expl_state, rng), memory_transitions = jax.lax.scan(
            _random_step,
            (*expl_state, _rng),
            None,
            config["MEMORY_WINDOW"] + config["NUM_STEPS"],
        )
        expl_state = tuple(expl_state)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, memory_transitions, expl_state, test_metrics, _rng)

        # runner_state, metrics = jax.lax.scan(
        #     _update_step, runner_state, None, config["NUM_UPDATES"]
        # )

        runner_state, metrics = jax.lax.scan(
            _update_plot, runner_state, None, config["NUM_UPDATES"]
        )

        return {"runner_state": runner_state, "metrics": metrics}

    return train

def run_pqn(config):

    reset_batch_logs()

    if not config["JIT"]:
        jax.config.update("jax_disable_jit", True)
        print("JIT disabled")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config), device=jax.devices()[config["GPU_ID"]])
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

    args = parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={key.upper(): value for key, value in vars(args).items()},
        name=args.run_name,
    )

    run_pqn(wandb.config)
