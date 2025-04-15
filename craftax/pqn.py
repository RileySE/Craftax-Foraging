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
from craftax.logz import Logger, Timer
from craftax.models import BatchRenorm


def parse_args():
    parser = argparse.ArgumentParser(description="Run PQN.")
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
    parser.add_argument("--EPS_START", type=float, default=1, help="Initial epsilon")
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
    return parser.parse_args()

class CNN(nn.Module):

    norm_type: str = "layer_norm"

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        elif self.norm_type == "batch_norm":
            normalize = lambda x: nn.BatchNorm(use_running_average=not train)(x)
        else:
            normalize = lambda x: x
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=nn.initializers.he_normal(),
        )(x)
        x = normalize(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=nn.initializers.he_normal())(x)
        x = normalize(x)
        x = nn.relu(x)
        return x


class QNetwork(nn.Module):
    action_dim: int
    layer_size: int
    norm_type: str = "layer_norm"
    norm_input: bool = False
    num_layers: int = 4
    hidden_size: int = 512
    is_symbolic: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool):
        if self.is_symbolic:
            """
            Symbolic observation uses the same PQN_craftax architecture.
            """
            if self.norm_input:
                x = BatchRenorm(use_running_average=not train)(x)
            else:
                # dummy normalize input for global compatibility
                x_dummy = BatchRenorm(use_running_average=not train)(x)

            if self.norm_type == "layer_norm":
                normalize = lambda x: nn.LayerNorm()(x)
            elif self.norm_type == "batch_norm":
                normalize = lambda x: BatchRenorm(use_running_average=not train)(x)
            else:
                normalize = lambda x: x

            for l in range(self.num_layers):
                x = nn.Dense(self.hidden_size)(x)
                x = normalize(x)
                x = nn.relu(x)
        else:

            """
            Pixel observation uses the PQN_atari architecture.
            """

            x = jnp.transpose(x, (0, 2, 3, 1))
            if self.norm_input:
                x = nn.BatchNorm(use_running_average=not train)(x)
            else:
                # dummy normalize input for global compatibility
                x_dummy = nn.BatchNorm(use_running_average=not train)(x)
                x = x / 255.0
            x = CNN(norm_type=self.norm_type)(x, train)
        q_val = nn.Dense(self.action_dim)(x)

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

        return q_val, aux


@chex.dataclass(frozen=True)
class Transition:
    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
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

        network = QNetwork(action_dim=action_space_size, is_symbolic=is_symbolic, layer_size=config["LAYER_SIZE"],)

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
            init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
            network_variables = network.init(rng, init_x, train=False)
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        new_obs, log_state = env.reset(_rng, env_params)

        # TRAINING LOOP
        def _update_step(runner_state, unused):

            # train_state, env_state, last_obs, test_metrics, rng, = runner_state
            train_state, expl_state, test_metrics, rng = runner_state

            # SAMPLE PHASE
            def _step_env(carry, _):
                last_obs, env_state, rng = carry
                rng, rng_a, rng_s = jax.random.split(rng, 3)
                q_vals, aux = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )

                # different eps for each env
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
                    obs=last_obs,
                    action=new_action,
                    reward=config.get("REW_SCALE", 1) * reward,
                    done=new_done,
                    next_obs=new_obs,
                    q_val=q_vals,
                    info=info,
                    deltas_to_start=deltas_to_start,
                )

                # return (train_state, new_env_state, new_obs, test_metrics, update_step, rng), (transition, info)
                return (new_obs, new_env_state, rng), (transition, info)

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

            last_q,_ = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                transitions.next_obs[-1],
                train=False,
            )
            last_q = jnp.max(last_q, axis=-1)

            def _get_target(lambda_returns_and_next_q, transition):
                lambda_returns, next_q = lambda_returns_and_next_q
                target_bootstrap = (
                        transition.reward + config["GAMMA"] * (1 - transition.done) * next_q
                )
                delta = lambda_returns - next_q
                lambda_returns = (
                        target_bootstrap + config["GAMMA"] * config["LAMBDA"] * delta
                )
                lambda_returns = (
                                         1 - transition.done
                                 ) * lambda_returns + transition.done * transition.reward
                next_q = jnp.max(transition.q_val, axis=-1)
                return (lambda_returns, next_q), lambda_returns

            last_q = last_q * (1 - transitions.done[-1])
            lambda_returns = transitions.reward[-1] + config["GAMMA"] * last_q
            _, targets = jax.lax.scan(
                _get_target,
                (lambda_returns, last_q),
                jax.tree_util.tree_map(lambda x: x[:-1], transitions),
                reverse=True,
            )
            lambda_targets = jnp.concatenate((targets, lambda_returns[np.newaxis]))

            # NETWORKS UPDATE
            def _learn_epoch(carry, _):
                train_state, rng = carry

                def _learn_phase(carry, minibatch_and_target):

                    train_state, rng = carry
                    minibatch, target = minibatch_and_target

                    def _loss_fn(params):

                        if config.get("Q_LAMBDA", False):
                            (q_vals, aux), updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                minibatch.obs,
                                train=True,
                                mutable=["batch_stats"],
                            )
                        else:
                            # if not using q_lambda, re-pass the next_obs through the network to compute target
                            (all_q_vals, aux), updates = network.apply(
                                {
                                    "params": params,
                                    "batch_stats": train_state.batch_stats,
                                },
                                jnp.concatenate((minibatch.obs, minibatch.next_obs)),
                                train=True,
                                mutable=["batch_stats"],
                            )
                            q_vals, q_next = jnp.split(all_q_vals, 2)
                            q_next = jax.lax.stop_gradient(q_next)
                            q_next = jnp.max(q_next, axis=-1)  # (batch_size,)
                            target = (
                                    minibatch.reward
                                    + (1 - minibatch.done) * config["GAMMA"] * q_next
                            )

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            jnp.expand_dims(minibatch.action, axis=-1),
                            axis=-1,
                        ).squeeze(axis=-1)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        # Calculate auxiliary loss (predict distance to origin)
                        # Simple L2
                        aux_loss = jnp.square(aux - jnp.concatenate((minibatch.deltas_to_start, minibatch.deltas_to_start))).mean()

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
                    x = x.reshape(
                        -1, *x.shape[2:]
                    )  # num_steps*num_envs (batch_size), ...
                    x = jax.random.permutation(rng, x)  # shuffle the transitions
                    x = x.reshape(
                        config["NUM_MINIBATCHES"], -1, *x.shape[1:]
                    )  # num_mini_updates, batch_size/num_mini_updates, ...
                    return x

                rng, _rng = jax.random.split(rng)
                minibatches = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), transitions
                )  # num_actors*num_envs (batch_size), ...
                targets = jax.tree_util.tree_map(
                    lambda x: preprocess_transition(x, _rng), lambda_targets
                )

                rng, _rng = jax.random.split(rng)
                (train_state, rng), (total_loss, qvals, critic_loss, aux_loss) = jax.lax.scan(
                    _learn_phase, (train_state, rng), (minibatches, targets)
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
                    if config.get("WANDB_LOG_ALL_SEEDS", False):
                        metrics.update(
                            {
                                f"rng{int(original_rng)}/{k}": v
                                for k, v in metrics.items()
                            }
                        )
                    wandb.log(metrics, step=metrics["update_steps"])

                jax.debug.callback(callback, metrics, original_rng)

            runner_state = (train_state, tuple(expl_state), test_metrics, rng)

            return runner_state, metrics

        def get_test_metrics(train_state, rng):

            if not config.get("TEST_DURING_TRAINING", False):
                return None

            def _env_step(carry, _):
                env_state, last_obs, rng = carry
                rng, _rng = jax.random.split(rng)
                q_vals = network.apply(
                    {
                        "params": train_state.params,
                        "batch_stats": train_state.batch_stats,
                    },
                    last_obs,
                    train=False,
                )
                eps = jnp.full(config["TEST_NUM_ENVS"], config["EPS_TEST"])
                new_action = jax.vmap(eps_greedy_exploration)(
                    jax.random.split(_rng, config["TEST_NUM_ENVS"]), q_vals, eps
                )
                new_obs, new_env_state, reward, new_done, info = test_env.step(
                    _rng, env_state, new_action, env_params
                )
                return (new_env_state, new_obs, rng), info

            rng, _rng = jax.random.split(rng)
            init_obs, env_state = test_env.reset(_rng, env_params)

            _, infos = jax.lax.scan(
                _env_step, (env_state, init_obs, _rng), None, config["TEST_NUM_STEPS"]
            )
            # return mean of done infos
            done_infos = jax.tree_util.tree_map(
                lambda x: (x * infos["returned_episode"]).sum()
                          / infos["returned_episode"].sum(),
                infos,
            )
            return done_infos

        def _env_step_viz(runner_state, unused):
            train_state, expl_state, test_metrics, rng = runner_state
            rng, rng_a  = jax.random.split(rng)

            last_obs = expl_state[0]
            env_state = expl_state[1]

            # select action using epsilon greedy
            (q_vals, aux) = network.apply(
                {
                    "params": train_state.params,
                    "batch_stats": train_state.batch_stats,
                },
                last_obs,
                train=False,
            )

            # different eps for each env
            _rngs = jax.random.split(rng_a, config["NUM_ENVS"])
            eps = jnp.full(config["NUM_ENVS"], eps_scheduler(train_state.n_updates))
            new_action = jax.vmap(eps_greedy_exploration)(_rngs, q_vals, eps)

            # step env
            rng, rng_s  = jax.random.split(rng)
            new_obs, new_env_state, reward, new_done, info = env_viz.step(
                rng_s, env_state, new_action, env_params
            )

            # Compute distance to origin for aux loss
            starting_pos = env_state.env_state.player_starting_position[env_state.env_state.player_level]
            deltas_to_start = env_state.env_state.player_position - starting_pos

            print("q_vals shape", q_vals.shape)
            print("new_action shape", new_action.shape)

            # use the values in new_action to get the q_vals
            q_vals_action_taken = jnp.take_along_axis(q_vals, jnp.expand_dims(new_action, axis=-1), axis=-1).squeeze(axis=-1)
            print("q_vals_action_taken shape", q_vals_action_taken.shape)

            info['value'] = q_vals_action_taken
            info['pred_delta'] = aux
            info['delta'] = deltas_to_start

            print("info['value'] shape", info['value'].shape)
            print("info['pred_delta'] shape", info['pred_delta'].shape)
            print("info['delta'] shape", info['delta'].shape)

            transition = Transition(
                obs=last_obs,
                action=new_action,
                reward=config.get("REW_SCALE", 1) * reward,
                done=new_done,
                next_obs=new_obs,
                q_val=q_vals,
                info=info,
                deltas_to_start=deltas_to_start,
            )

            return runner_state, transition

        def _logging_step(runner_state, unused, logging_threads):
            runner_state, minibatch = jax.lax.scan(
                _env_step_viz, runner_state, None, config['STEPS_PER_VIZ']
            )

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
            def write_scalars(scalars, increment=0):

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
            log_array = minibatch.info['action'].reshape(minibatch.info['action'].shape + (1,))

            # Yes this is a for loop in the JAX code but this stuff was getting done in serial before anyway and it's cheap operations
            for field_to_log in fields_to_log:
                log_array = add_field_to_log_array(minibatch.info, log_array, field_to_log)

            jax.debug.callback(write_scalars, log_array, runner_state[0].n_updates)

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
        test_metrics = get_test_metrics(train_state, _rng)

        rng, _rng = jax.random.split(rng)
        expl_state = env.reset(_rng, env_params)

        # train
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, expl_state, test_metrics, _rng)

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
