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
from jax import lax
import jax.nn as jnn

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

from craftax import craftax_state
from environment_base.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    VideoPlotWrapper,
    ReduceActionSpaceWrapper, AppendActionToObsWrapper, AppendActionToObsWrapper,
    CurriculumWrapper
)
from logz.batch_logging import create_log_dict, batch_log, reset_batch_logs


def parse_args():
    parser = argparse.ArgumentParser(description="Run sparsity PPO.")
    parser.add_argument("--prune_step", type=int, default=20000, help="Step to prune")
    parser.add_argument('--featureless_world', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run_name", type=str, default="default_run", help="Name of the run")
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1", help="Environment name")
    parser.add_argument("--sparse_alg", type=str, default="magnitude", help="options, magnitude, no_prune, saliency, random")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--predators", type=bool, default=True, help="Use predators")
    parser.add_argument("--sparsity", type=float, default=0.4, help="Sparsity value")
    parser.add_argument("--max_cows", type=int, default=72, help="Maximum number of cows that can exist at a time")
    parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments")
    parser.add_argument("--total_timesteps", type=float, default=3e9, help="Total timesteps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_env_steps", type=int, default=64, help="Number of environment steps")
    parser.add_argument("--update_epochs", type=int, default=4, help="Number of update epochs")
    parser.add_argument("--num_minibatches", type=int, default=8, help="Number of minibatches")
    parser.add_argument("--gamma", type=float, default=0.99, help="Gamma value")
    parser.add_argument("--gae_lambda", type=float, default=0.8, help="GAE Lambda")
    parser.add_argument("--clip_eps", type=float, default=0.2, help="Clip epsilon")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient")
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
    return parser.parse_args()

NUM_CAN = 3
N = 12

class ScannedRNN(nn.Module):
    """
    GRU layer that now consumes                        ┌─────────────┐
      • the previous RNN hidden state  hₜ₋₁            │ carry        │
      • the standard network input      xₜ   = ins     │ rnn_state    │
      • a reset flag                    dₜ   = resets  ├─────────────┤
      • the three CAN states,           cₜ             │              │
                                                       └─────────────┘
    """
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
        ins, resets, can_states = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        gru_input = jnp.concatenate([ins, can_states], axis=-1)  # (B, 512+3·144)
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, gru_input)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

def direction_preference_grid(N: int) -> jnp.ndarray:
    """
    Build the (N², 2) table of unit vectors
    (0,1), (1,0), (0,‑1), (‑1,0) in a 2 × 2 tiling pattern.
    """
    # Pattern for a 2×2 tile, shape (2,2,2)
    tile = jnp.array([
        [[0.,  1.],  # North
         [1.,  0.]], # East
        [[0., -1.],  # South
         [-1., 0.]]  # West
    ])
    # Repeat it until we get an (N,N,2) array, then flatten to (N²,2)
    reps = (N // 2, N // 2, 1)
    return jnp.tile(tile, reps).reshape(-1, 2)      # (N*N, 2)

def rhombus_coords(N: int) -> jnp.ndarray:                   # (N², 2)
    i = jnp.arange(N)
    j = jnp.arange(N)
    ii, jj = jnp.meshgrid(i, j, indexing="ij")               # shape (N,N)

    # eqn (28) from the paper …  (cells are in [0, 1)  ×  [0, √3/2) )
    x = (ii + 0.5 * jj + 0.75) / N
    y = (jnp.sqrt(3) / 2) * (jj + 0.5) / N
    return jnp.stack([x, y], axis=-1).reshape(-1, 2)

"""def make_dist_matrix(n_rows: int, n_cols: int):
    i = jnp.arange(n_rows, dtype=jnp.float32).reshape(n_rows, 1)  # shape (n_rows, 1)
    j = jnp.arange(n_cols, dtype=jnp.float32).reshape(1, n_cols)  # shape (1, n_cols)
    M = jnp.abs(i - j) / 512.0                                    # element‑wise formula
    return M
"""

def make_dist_matrix(
    N: int,
    *,
    a: float   = 1.0,      # centre‑surround gain
    sigma: float = 4.5,    # width of the excitatory bump   (in “r” units below)
    l: float   = 1.5,      # shift strength (cf. equation (18))
    dtype=jnp.float32
) -> jnp.ndarray:          # (N², N²)
    P = N * N                                      # total number of neurons

    pos   = rhombus_coords(N)                      # (P, 2)
    e     = direction_preference_grid(N)           # (P, 2)

    # ------------------------------------------------------------------
    # base (unshifted) displacement  …  (P, P, 2)
    # Δ = r_i  –  r_j  –  (l/N) · e_j     ––– eqn (18)
    # ------------------------------------------------------------------
    diff = (pos[:, None, :]                       # (P,1,2)
            - pos[None, :, :]                     # (1,P,2)
            - (l / N) * e[None, :, :])            # (1,P,2)

    # nine slip‑vectors that realise the triangular torus tiling
    shifts = jnp.asarray([
        [ 0.0,  0.0],
        [-0.5,  jnp.sqrt(3)/2],  [-0.5, -jnp.sqrt(3)/2],
        [ 0.5,  jnp.sqrt(3)/2],  [ 0.5, -jnp.sqrt(3)/2],
        [-1.0,  0.0],            [ 1.0,  0.0],
        [-1.5, -jnp.sqrt(3)/2],  [ 1.5,  jnp.sqrt(3)/2],
    ], dtype=dtype)                                # (9, 2)

    # ------------------------------------------------------------------
    # for every pair (i,j) take the *minimum* distance over the 9 shifts
    # ------------------------------------------------------------------
    d2   = jnp.sum(                                # (9,P,P)
        (diff[None, :, :, :] + shifts[:, None, None, :]) ** 2,
        axis=-1
    )
    d2_min    = d2.min(axis=0)               # (P,P)   – actual distance
    r2    = d2_min * (N*N)                                  # scale exactly as in the
                                                  # reference implementation

    # ------------------------------------------------------------------
    # centre‑surround profile  W₀(r)  (eqn (15))
    # ------------------------------------------------------------------
    W = a * jnp.exp(-r2 / (2.0 * sigma**2)) - a    # (P,P)

    return W.astype(dtype)

class VelocityMatMul(nn.Module):
    """Batched mat‑mul:  v  ↦  M·v  with a fixed N*N×N*N matrix.

    * Input  shape:  (T, B, 2)   – because we keep the same scan
      convention as `ScannedRNN` (time axis 0, batch axis 1, features 2).
    * Output shape:  (T, B, N*N)
    * No parameters are created; `M` lives in the non‑trainable
      `constants` collection so it is saved/restored with checkpoints
      but never updated by the optimiser.

      Recurrent layer
        hₜ   ∈ ℝ^{N²}   (stored in `carry`)
        vₜ   ∈ ℝ²        (fed in every step)

        outₜ = hₜ·Mᵀ + (1 + α vₜ·dir_prefᵀ)
        hₜ₊₁ = outₜ
    """
    N: int = N
    alpha: float = 0.1

    dt: float = 0.05
    tau: float = 1.0
    steps: int = 50

    @functools.partial(
        nn.scan,
        variable_broadcast=("constants",),       # nothing in "params", but keeps the signature identical
        variable_axes={"intermediates": 0}, # stack W along the time axis
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, velocity):

        vel, dones = velocity
        # --------------------------------------------------
        # carry is unused – we just echo it back
        # --------------------------------------------------

        # Create / retrieve the fixed matrix once.
        # Shape (512, 2); dtype=float32  (match your other layers).
        M = self.variable(
            "constants", # Ensures that the matrix is only created once.
            "fixed_M",
            lambda: make_dist_matrix(self.N)   # (N*N, 2)
        )

        dir_pref = self.variable(
            "constants",
            "dir_pref",
            lambda: direction_preference_grid(self.N) # (N*N, 2)
        )

        can_state = carry
        can_state = jnp.where(
            dones[:, np.newaxis],
            self.initialize_carry(vel.shape[0], N),
            can_state,
        )
        # --------------------------------------------------------
        # 2.  Core computation each time step
        # --------------------------------------------------------
        

        dot = jnp.matmul(vel, dir_pref.value.T) # (B, 2) @ (2, N*N) → (B, N*N)
        W = 1.0 + self.alpha * dot

        self.sow("intermediates", "W", W)

        def step(_, s):
            rec_input = jnp.matmul(s, M.value.T)
            ds = (-s + jnn.relu(rec_input + W)) / self.tau
            return s + self.dt * ds
        
        can_state = lax.fori_loop(0, self.steps, step, can_state)

        #mv_term = jnp.matmul(carry, M.value.T)  # (B, N*N) @ (N*N, N*N) → (B, N*N)

        out = can_state # (B, N*N) + (B, N*N) → (B, N*N) 
        new_carry = out
      
        # I need to add more of these VelocityMatMul layers to the network
        # I need to make the recurrency so that the GRU has the carries and input as well.

        return new_carry, out
    
    @staticmethod
    def initialize_carry(batch_size: int,
                         N: int,
                         dtype = jnp.float32) -> jnp.ndarray:
        
        state = jax.random.uniform(
            jax.random.PRNGKey(0),
            shape=(batch_size, N * N),
            minval=0.0,
            maxval=0.01,
            dtype=dtype
        )
        return state.at[:, 0].set(1.0)

class ActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        rnn_state, can_states = hidden          # can_states ≡ (s1, s2, s3)
        obs, dones = x
        #print(can_states[0].shape)

        embedding = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)



        # ──────────────────────────────────────────────────────────────
        # 2) flatten & strip ANY old time‐axes from the CAN states
        # ──────────────────────────────────────────────────────────────
        #    each cs might be (B, F) *or* (1, B, F); we force (B, F):
        flat_cs = []
        for cs in can_states:
            # collapse all leading dims except the last
            new_shape = (cs.shape[-2], cs.shape[-1])
            flat_cs.append(cs.reshape(new_shape))
        # now can_batched is (B, 3*144)

        def pool4(x, N=12, block=6):
            B = x.shape[0]
            grid   = x.reshape(B, N, N)                            # (B,12,12)
            blocks = grid.reshape(B, N//block, block, N//block, block)  # (B,2,6,2,6)
            small  = blocks.mean(axis=(2,4))                       # (B,2,2)
            return small.reshape(B, -1)                            # (B,4)

        pooled_cs = [ pool4(cs) for cs in flat_cs ]              # three (B,4)
        can_pooled = jnp.concatenate(pooled_cs, axis=-1)         # (B, 12)

        # ──────────────────────────────────────────────────────────────
        # 3) broadcast that *once* over your RNN time‐axis
        # ──────────────────────────────────────────────────────────────
        T, B, _ = embedding.shape
        can_flat = jnp.broadcast_to(
            can_pooled[None, ...],            # (1, B, 12)
            (T, B, can_pooled.shape[-1])      # → (T, B, 12)
        )
        
        #print(embedding.shape)
        #print(can_flat.shape)
        rnn_in = (embedding, dones, can_flat)
        new_rnn_state, embedding = ScannedRNN()(rnn_state, rnn_in)

        

        #########################################################
        # 3. Velocity heads  
        #########################################################
        new_can_states = []
        vel_embeds = []
        vel_list = []
        can_states = list(can_states)

        for i in range(NUM_CAN):
            vel_raw = nn.Dense(
                2,
                kernel_init=orthogonal(1.0),     # keep gain moderate
                bias_init=constant(0.0),
                name=f"velocity_head{i+1}",
            )(embedding)
            
            velocity = nn.tanh(vel_raw)
            vel_list.append(velocity)
            vel_in = (velocity, dones)
            new_can_state, vel_embed = VelocityMatMul()(can_states[i], vel_in)
            
            new_can_states.append(new_can_state)
            vel_embeds.append(vel_embed)
        
        self.sow("intermediates", "can_states", new_can_states[0])
        
        new_can_states = tuple(new_can_states)

        #print(f"Velocity embed - shape: {vel_embed.shape}. Type: {type(vel_embed)}")

        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        #print(f"Actor mean - shape: {actor_mean.shape}. Type: {type(actor_mean)}")
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
        self.config["LAYER_SIZE"],
        kernel_init=orthogonal(2),
        bias_init=constant(0.0),
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
        new_hidden = (new_rnn_state, new_can_states)

        return new_hidden, pi, jnp.squeeze(critic, axis=-1), aux, vel_list


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
    if config['PREDATORS']:
        static_params.predators = True
    static_params.map_size = (config['MAP_SIZE'],config['MAP_SIZE'])
    static_params.directional_vision = config['DIRECTIONAL_VISION']

    static_params.max_passive_mobs = config['MAX_COWS']

    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
        )

        env = CraftaxClassicSymbolicEnv()
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
        )

        env = CraftaxClassicPixelsEnv()
        is_symbolic = False
    elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

        env = CraftaxSymbolicEnv(static_params)
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

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
        init_rnn_state = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        init_can_states = tuple(
            VelocityMatMul.initialize_carry(
                config["NUM_ENVS"], N
            )
            for _ in range(NUM_CAN)
        )
        init_hstate = (init_rnn_state, init_can_states)
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
        sparsity_config.update_start_step = config["PRUNE_STEP"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
        sparsity_config.update_end_step = config["PRUNE_STEP"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
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
        init_rnn_state = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        init_can_states = tuple(
            VelocityMatMul.initialize_carry(
                config["NUM_ENVS"], N
            )
            for _ in range(NUM_CAN)
        )
        init_hstate = (init_rnn_state, init_can_states)

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
                hstate, pi, value, aux, vel_list = network.apply(train_state.params, hstate, ac_in)
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
                     _rng, env_state, action, env_params
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
            _, _, last_val, _, _ = network.apply(train_state.params, hstate, ac_in)
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
                    rnn_state, can_states = init_hstate
                    init_hstate = (rnn_state[0], tuple(cs[0] for cs in can_states))

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value, aux, vel_list = network.apply(
                            params, init_hstate, (traj_batch.obs, traj_batch.done)
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

                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
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

            #init_hstate = initial_hstate[None, :]  # TBH
            init_hstate = jax.tree_util.tree_map(lambda x: x[None, ...], initial_hstate)

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

            metric = jax.tree_util.tree_map(
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
            hstate, pi, value, aux, vel_list = network.apply(train_state.params, hstate, ac_in)
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
            info['value'] = value
            info['hidden_state'] = hstate[0] # can also add the can state here if we want to log that
            info['can_states'] = jnp.concatenate(hstate[1], axis=-1)  # Shape: (B, 4 * N*N)
            info['pred_delta'] = aux
            info['delta'] = deltas_to_start
            info['entropy'] = pi.entropy().squeeze(0)
            info['log_prob'] = log_prob
            info['vel_list'] = jnp.concatenate(vel_list, axis=-1)
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
                                      'pred_delta', 'num_monsters_killed', 'has_sword', 'has_pick', 'held_iron', 'value',
                                      'entropy', 'log_prob', 'episode_id', 'can_states', 'vel_list']

            # Callback function for logging hidden states
            def write_rnn_hstate(hstate, scalars, increment=0):


                header_field_names = ['health','food','drink','energy','done','is_sleeping','is_resting','player_position_x',
                                      'player_position_y','recover','hunger','thirst','fatigue','light_level','dist_to_melee_l1',
                                      'melee_on_screen','dist_to_passive_l1','passive_on_screen','dist_to_ranged_l1',
                                      'ranged_on_screen','num_melee_nearby','num_passives_nearby','num_ranged_nearby','delta_x',
                                      'delta_y', 'pred_delta_x', 'pred_delta_y', 'num_monsters_killed', 'has_sword',
                                      'has_pick', 'held_iron', 'value', 'entropy', 'log_prob', 'episode_id']
                
                header_field_names += [f'cs_{can_num}_{i}' for can_num in range(NUM_CAN) for i in range(N*N)]
                header_field_names += [
                        name for i in range(NUM_CAN) for name in (f'vx_{i}', f'vy_{i}')
                    ]
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
            def save_weights_callback(weights, iter):
                weights_flat = jax.tree_util.tree_flatten(weights)
                run_out_path = os.path.join(config['OUTPUT_PATH'], wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                weight_filename = os.path.join(run_out_path, 'weights_{}.csv'.format(iter))
                weight_file = open(weight_filename, 'w')
                weights_params = weights['params']

                def save_weight_dict(curr_value, key_string=''):
                    if type(curr_value) != dict:
                        np.savetxt(weight_file, np.transpose(curr_value), delimiter=',', fmt='%f', header=key_string)
                        return True
                    else:
                        for key in curr_value.keys():
                            save_weight_dict(curr_value[key], key_string + '/' + key)
                    return True

                save_weight_dict(weights_params)

                print('Saving weights in file', weight_filename)


            jax.debug.callback(save_weights_callback, runner_state[0].params, runner_state[-1])

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
        train_state = jax.tree_util.tree_map(lambda x: x[0], train_states)
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

import matplotlib.pyplot as plt
import numpy as np

def test_velocity_matmul():
    vm          = VelocityMatMul()                    
    batch_size  = 1
    carry0      = vm.initialize_carry(batch_size, vm.N)
    vel_seq     = jnp.asarray(
        [[ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         [ 1.0,  0.0],
         ],
        dtype=jnp.float32
    ).reshape(14, batch_size, 2)                       # (T, B, 2)
    dones_seq = jnp.zeros((vel_seq.shape[0], batch_size),dtype=jnp.bool_)
    rng       = jax.random.PRNGKey(0)
    variables = vm.init(rng, carry0, (vel_seq, dones_seq))

    carryT, outs = vm.apply(variables, carry0, (vel_seq, dones_seq))   # outs → (3, 1, N²)

    np.set_printoptions(precision=4, suppress=True)       # nicer console output
    for t in range(14):
        grid = np.asarray(outs[t, 0]).reshape(vm.N, vm.N) # (12, 12)
        print(f"\n=== carry after input #{t+1}  velocity={vel_seq[t,0]} ===")
        print(grid)

        # Plot heatmap
        plt.figure()
        plt.imshow(grid, origin='lower')
        plt.title(f"Carry after input #{t+1}  velocity={np.asarray(vel_seq[t,0])}")
        plt.colorbar()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    """
    model   = VelocityMatMul(N=12, alpha=0.1)

    #  (T=1, batch=1, 2)   – one time‑step, one example, velocity = (1, 0)
    v_seq   = jnp.array([[[1.0, -0.5]]], dtype=jnp.float32)
    carry0  = ()                       # unused but must be passed in

    # 1️⃣ initialise parameters & constants
    variables = model.init(jax.random.PRNGKey(0), carry0, v_seq)

    # 2️⃣ forward pass while asking for the “intermediates” collection
    (_, _), inter = model.apply(
            variables, carry0, v_seq,
            mutable=["intermediates"])
    
 
    # 3️⃣ pull out the feed‑forward term
    W_val = inter["intermediates"]["W"][0]   # (1, N²)

    print("W shape:", W_val.shape)       # (1, 144)  for N=12
    print(W_val.reshape(model.N, model.N))  # nicer 12×12 view
    """

    #test_velocity_matmul()
    
    
    args = parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={key.upper(): value for key, value in vars(args).items()},
        name=args.run_name,
    )

    run_ppo(wandb.config)
    

    
    