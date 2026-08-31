import argparse
import glob
import gzip
import os
import pickle
import sys

# --gpu_id must take effect before JAX initializes its backend: JAX
# preallocates memory on EVERY visible GPU at backend init (not just the
# device passed to jax.jit), and forageworld.craftax.constants unpickles the
# texture cache onto the default device (GPU 0) at import time. Masking
# CUDA_VISIBLE_DEVICES here, before the jax/forageworld imports below, is the
# only way to keep the process off the other GPUs entirely. If the caller
# already set CUDA_VISIBLE_DEVICES it is respected and --gpu_id indexes
# within the visible devices, as before. The sentinel env var keeps the
# device index stable when this module is re-imported in the same process
# (the GPU mask cannot change once the JAX backend exists).
_gpu_id_parser = argparse.ArgumentParser(add_help=False)
_gpu_id_parser.add_argument("--gpu_id", type=int, default=0)
_gpu_id = _gpu_id_parser.parse_known_args()[0].gpu_id
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_id)
    os.environ["_PPO_RNN_MASKED_CUDA_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"]
if os.environ.get("_PPO_RNN_MASKED_CUDA_DEVICES") == os.environ["CUDA_VISIBLE_DEVICES"]:
    JAX_DEVICE_INDEX = 0  # the masked GPU is the only visible device
else:
    JAX_DEVICE_INDEX = _gpu_id

# Work around an XLA (jaxlib 0.4.33) codegen bug that aborts the process while
# compiling the training step. XLA's Triton GEMM autotuner rewrites some of the
# Dense weight-gradient dots in the minibatch update loop with split-K, i.e.
# into [split_k, M, N] partial sums plus a reduce over dim 0. Those dots have a
# transposed output layout, so the partial-sums tensor is non-monotonic
# (f32[4,517,517]{1,2,0}); when the split-K reduce then gets multi-output-fused
# with the sum-of-squares reduce that optax.clip_by_global_norm emits, the
# fusion becomes kInput and the GPU reduction emitter CHECK-fails with
# "reduction-layout-normalizer must run before code generation" (an abort(),
# not a catchable exception). Which dots get split-K depends on the minibatch
# shape, so whether a run hits this depends on --num_envs / --num_env_steps /
# --num_minibatches / --layer_size: --num_envs 256 aborts while 1024 happens to
# survive (it builds the same transposed split-K reduces, they just land in
# layout-agnostic kLoop fusions). Disabling split-K autotuning removes the
# rewrite entirely; on the 1024-env config it measured within noise of the
# default. XLA_FLAGS must be set before JAX initializes its backend, and an
# explicit setting from the environment wins.
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_enable_split_k_autotuning" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        _xla_flags + " --xla_gpu_enable_split_k_autotuning=false"
    ).strip()

import random
from math import ceil, sqrt
from functools import partial
import jax
import jax.numpy as jnp
import flax.linen as nn
import jaxpruner
import numpy as np
import optax
import time
import pandas as pd

from flax.training import orbax_utils
from matplotlib import pyplot as plt, animation
from orbax.checkpoint import (
    PyTreeCheckpointer,
    CheckpointManagerOptions,
    CheckpointManager,
)

import wandb
from flax.linen.initializers import constant, orthogonal, normal, uniform
from typing import Sequence, NamedTuple, Dict, Callable
from flax.training.train_state import TrainState
import distrax
import functools
from ml_collections import ConfigDict

from forageworld.craftax import craftax_state
from forageworld.environment_base.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    EpisodeInfoWrapper,
    FastVideoWrapper,
    ReduceActionSpaceWrapper, AppendActionToObsWrapper, AppendActionToObsWrapper,
    CurriculumWrapper
)
from forageworld.logz.batch_logging import create_log_dict, batch_log, reset_batch_logs, resumable_wandb_log
from forageworld.models.actor_critic import ActorCritic, ActorCriticConv, ActorCriticSharedRep
from forageworld.connectome_utils import (
    connectome_constraint_loss,
    connectome_constraint_loss_nonsorted,
    connectome_loss_nonzero,
    randomize_target_matrix,
    uniform_random_target_matrix,
)


def build_parser():
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
    parser.add_argument("--connect_coef", type=float, default=0.1, help="Connectome constraint coefficient")
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
    parser.add_argument('--no_hidden_state_csv', action=argparse.BooleanOptionalAction, default=False,
                        help='Disable writing the per-step hidden-state/scalar CSV logs during visualization runs. Logging is on by default.')
    parser.add_argument('--full_action_space', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reward_function", type=str, default='foraging', help="Reward function")
    parser.add_argument("--validation_seed", type=int, default=777, help="Validation seed")
    parser.add_argument("--validation_step_offset", type=int, default=0, help="Validation step offset")
    parser.add_argument("--logging_threads_per_viz", type=int, default=1, help="Logging threads per viz")
    parser.add_argument("--logging_threads_per_viz_val", type=int, default=1, help="Logging threads per viz validation")
    parser.add_argument("--curriculum", type=bool, default=False, help="Use curriculum learning")
    parser.add_argument("--map_size", type=int, default=96, help="The side length for the map")
    parser.add_argument("--directional_vision", action=argparse.BooleanOptionalAction, default=False, help="Turn on directional vision cones")
    parser.add_argument('--connectome_filepath', type=str, default='./', help="path to the preprocessed connectome cell/type file")
    parser.add_argument('--no_memory', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--random_start', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--connectome_init', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--connectome_freeze', action=argparse.BooleanOptionalAction, default=False,
                        help="Freeze the constrained weights at their initial values throughout training. Does not change the initialization on its own — combine with --connectome_init (or --connectome_zero_init) to control what values are frozen.")
    parser.add_argument('--connectome_freeze_zeros', action=argparse.BooleanOptionalAction, default=False,
                        help="Freeze only the entries whose connectome target is zero at their initial values; non-zero-target entries train normally. Does not change the initialization on its own — combine with --connectome_init (or --connectome_zero_init) to control what values are frozen.")
    parser.add_argument('--connectome_randomize_targets', action=argparse.BooleanOptionalAction, default=False,
                        help="Replace the loaded connectome targets with a randomized matrix that preserves the global zero fraction and the distribution of non-zero values; the seed for the shuffle is taken from --seed. Block-sorted to match the default loss, or left unsorted under --fixed_connectome_targets so the control matches that loss's target layout.")
    parser.add_argument('--connectome_uniform_targets', action=argparse.BooleanOptionalAction, default=False,
                        help="Replace the loaded connectome targets with a matrix whose non-zero entries are drawn i.i.d. from Uniform(0, 1) at uniformly random positions, preserving only the count of non-zero entries. Composes with --connectome_init / --connectome_freeze / --connectome_freeze_zeros / --connectome_zero_init; if --connectome_randomize_targets is also set, the uniform replacement is applied after and effectively wins. Seed for the draw is taken from --seed. Block-sorted to match the default loss, or left unsorted under --fixed_connectome_targets so the control matches that loss's target layout.")
    parser.add_argument('--connectome_zero_init', action=argparse.BooleanOptionalAction, default=False,
                        help="Sanity-check init: set the constrained kernel to all zeros instead of the default initializer. Takes precedence over --connectome_init when both are set. Composes with --connectome_freeze / --connectome_freeze_zeros (which then freeze the zero-initialized kernel).")
    parser.add_argument('--no_init_diags', action=argparse.BooleanOptionalAction, default=False,
                        help="Leave the diagonal (each unit's self weight) of the RNN hidden-to-hidden kernel out "
                             "of the manual connectome initialization: the off-diagonal entries are overwritten by "
                             "--connectome_init / --connectome_zero_init as usual, while the diagonal keeps the "
                             "value drawn by --rnn_hidden_initializer (orthogonal / normal / uniform). Only affects "
                             "initialization - it does not change the constraint loss (see --exclude_self_weights "
                             "for that) or which weights train. Requires --connectome_init or "
                             "--connectome_zero_init, since it has nothing to modify otherwise.")
    parser.add_argument('--exclude_self_weights', action=argparse.BooleanOptionalAction, default=False,
                        help="Exclude each unit's self weight (the diagonal of the RNN hidden-to-hidden kernel) from "
                             "the connectome constraint loss: the diagonal is zeroed before the block-sorted "
                             "comparison, so self weights receive no constraint gradient and do not compete with "
                             "other weights for target slots. Requires the connectome targets (incompatible with "
                             "--no_connectome).")
    parser.add_argument('--fixed_connectome_targets', action=argparse.BooleanOptionalAction, default=False,
                        help="Use the non-sorted connectome constraint loss: each hidden-to-hidden weight [i, j] is "
                             "compared directly against target [i, j], a fixed per-weight target for the whole run, "
                             "instead of the default magnitude-sorted within-block pairing. The choice is resolved "
                             "at JIT trace time, so the toggle adds no runtime cost. Composes with "
                             "--exclude_self_weights and the target-replacement options, which skip their "
                             "within-block sort when this is set so the controls stay comparable to the "
                             "unsorted targets; incompatible with --no_connectome.")
    parser.add_argument('--simple_network', action=argparse.BooleanOptionalAction, default=False, help='Use the simplified network architecture with no nonlinearity downstream of the RNN.')
    parser.add_argument('--simpler_network', action=argparse.BooleanOptionalAction, default=False,
                        help='Use an even simpler network architecture which also removes the FC layer upstream of the RNN.')
    parser.add_argument('--truncate_backprop', type=int, default=0,
                        help="Truncate backprop through time in the RNN to at most this many timesteps, by stopping "
                             "the hidden-state gradient at every K-step boundary within the rollout window (chunked "
                             "truncated BPTT: a step's gradient reaches back to the most recent boundary, i.e. "
                             "between 1 and K steps). Forward pass and rollouts are unaffected. 0 (default) means "
                             "full BPTT, with no added cost when disabled.")
    parser.add_argument('--grad_viz_steps', type=int, default=1000,
                        help="Horizon (timesteps) of the periodic RNN gradient-propagation probe. On the same "
                             "interval as the other visualization logging (every --updates_per_viz updates), the "
                             "policy is rolled forward this many steps and the final step's value estimate is "
                             "backpropagated through the RNN; plots of mean |d value / d obs| versus timesteps into "
                             "the past, plus example per-env traces, are written to "
                             "output_path/<run_id>/rnn_grad_mag_<step>.png (raw data in the matching .npz). The "
                             "probe always uses full BPTT regardless of --truncate_backprop, so it shows the "
                             "network's intrinsic gradient propagation. 0 disables the probe.")
    parser.add_argument('--grad_viz_envs', type=int, default=8,
                        help="Number of environments traced by the RNN gradient probe. Probe memory scales linearly "
                             "with this (it stores grad_viz_steps x grad_viz_envs observations plus their gradient).")
    parser.add_argument('--no_connectome', action=argparse.BooleanOptionalAction, default=False,
                        help="Skip loading the connectome targets from --connectome_filepath and skip the connectome constraint term in the loss. Incompatible with --connectome_init / --connectome_zero_init / --connectome_freeze / --connectome_freeze_zeros / --connectome_randomize_targets / --connectome_uniform_targets, since those all require the loaded targets.")
    parser.add_argument("--rnn_hidden_initializer", type=str, default='orthogonal', help='Select an initializer for the hidden-hidden weights of the RNN. Valid options are "orthogonal" (default), "normal", and "uniform"')
    parser.add_argument('--checkpoint_interval', type=int, default=1,
                        help="Save a full training checkpoint (network weights, optimizer state, environment states, "
                             "RNG, and progress counters) every this many outer visualization iterations — the same "
                             "cadence as the other periodic logging, i.e. every checkpoint_interval * updates_per_viz "
                             "updates. Checkpoints are written gzip-compressed to "
                             "output_path/<run_id>/checkpoint_<step>.pkl.gz. 0 disables checkpointing.")
    parser.add_argument('--wandb_resume_run', action=argparse.BooleanOptionalAction, default=False,
                        help="With --resume_from: continue logging to the same WandB run that wrote the checkpoint "
                             "(its run id is stored in the checkpoint) instead of starting a new WandB run. Metrics "
                             "are then logged with an explicit step — the absolute PPO update index, which survives "
                             "the resume — so performance curves continue seamlessly; any updates re-executed "
                             "between the checkpoint and the original run's last logged point are skipped rather "
                             "than double-logged (the resumed computation reproduces them identically). The flag is "
                             "stored in the checkpoint config, so later resumes inherit it unless overridden with "
                             "--no-wandb_resume_run. Requires online WandB (wandb cannot resume offline runs).")
    parser.add_argument('--checkpoint_keep', type=int, default=1,
                        help="Number of most recent checkpoints to retain; after each successful save, older "
                             "checkpoint files in the run's output directory are deleted. Interruption recovery only "
                             "ever needs the latest checkpoint, so the default of 1 keeps a run's checkpoint "
                             "footprint constant (~the size of one compressed checkpoint) regardless of run length. "
                             "0 keeps all checkpoints (e.g. to later branch runs from intermediate states).")
    parser.add_argument('--final_checkpoint', action=argparse.BooleanOptionalAction, default=True,
                        help="Write a training checkpoint once the last training iteration completes (before the "
                             "final logging/validation pass), so a finished run leaves behind the state needed to "
                             "fine-tune or branch from it with --resume_from. Independent of --checkpoint_interval: "
                             "a run with periodic checkpointing disabled still gets this one. Subject to "
                             "--checkpoint_keep, so with the default keep of 1 it is the only checkpoint left on "
                             "disk when the run ends.")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to a checkpoint_<step>.pkl.gz written by a previous run. The full training state "
                             "(all repeats) is restored and training resumes exactly where the checkpoint left off, "
                             "proceeding identically to an uninterrupted run. Configuration comes from the "
                             "checkpoint, except that any flags explicitly passed on this command line override the "
                             "stored values (e.g. to change a hyperparameter mid-run); overrides that change the "
                             "shape of the stored state (network size, env count, map size, num_repeats, ...) fail "
                             "at restore time.")
    parser.add_argument('--auto_resume', action=argparse.BooleanOptionalAction, default=False,
                        help="At startup, scan output_path for training checkpoints left by a previous (interrupted) "
                             "run of this configuration and resume from the one with the highest update step, exactly "
                             "as if it had been passed via --resume_from; start fresh when none exist. The original "
                             "WandB run is continued (as with --wandb_resume_run, so requires online WandB) unless "
                             "--no-wandb_resume_run is passed explicitly. Setting PPO_AUTO_RESUME=1 in the "
                             "environment also enables this flag - utils/auto_resume.py injects that at sbatch time "
                             "so unmodified submission scripts become resumable. When training runs to completion "
                             "and PPO_AUTO_RESUME_STATE_DIR is set, a done_<SLURM_ARRAY_TASK_ID> marker file is "
                             "written there so the auto-resume manager stops submitting continuation jobs.")
    return parser


def parse_args():
    parser = build_parser()
    args = parser.parse_args()
    if os.environ.get("PPO_AUTO_RESUME") == "1":
        # Injected by utils/auto_resume.py via sbatch --export so submission
        # scripts become resumable without editing their command lines.
        args.auto_resume = True
    if args.truncate_backprop < 0:
        parser.error("--truncate_backprop must be >= 0 (0 disables truncation)")
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint_interval must be >= 0 (0 disables checkpointing)")
    if args.checkpoint_keep < 0:
        parser.error("--checkpoint_keep must be >= 0 (0 keeps all checkpoints)")
    if args.wandb_resume_run and args.resume_from is None and not args.auto_resume:
        parser.error("--wandb_resume_run requires --resume_from or --auto_resume (it continues the run that wrote "
                     "the checkpoint)")
    if args.grad_viz_steps > 0 and args.grad_viz_envs < 1:
        parser.error("--grad_viz_envs must be >= 1 when the gradient probe is enabled")
    if args.no_connectome:
        incompatible = [
            name for name in (
                'connectome_init', 'connectome_zero_init',
                'connectome_freeze', 'connectome_freeze_zeros',
                'connectome_randomize_targets', 'connectome_uniform_targets',
                'exclude_self_weights', 'fixed_connectome_targets',
                'no_init_diags',
            ) if getattr(args, name)
        ]
        if incompatible:
            parser.error(
                "--no_connectome is incompatible with: "
                + ", ".join(f"--{n}" for n in incompatible)
            )
    if args.no_init_diags and not (args.connectome_init or args.connectome_zero_init):
        parser.error("--no_init_diags requires --connectome_init or --connectome_zero_init; it only changes "
                     "which entries those manual initializations overwrite, so on its own it does nothing "
                     "(the diagonal already comes from --rnn_hidden_initializer)")
    return args


def parse_explicit_args():
    """Return a dict of only the arguments explicitly present on the command
    line (defaults omitted). Used when resuming from a checkpoint: the
    checkpoint's stored config is the baseline, and only flags the user
    actually typed override it — argument defaults must not clobber the
    original run's settings.
    """
    parser = build_parser()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return vars(parser.parse_known_args()[0])


def find_latest_checkpoint(output_path):
    """Return the path of the newest training checkpoint under output_path, or
    None when there is none (--auto_resume then starts a fresh run).

    Checkpoints live at output_path/<wandb_run_id>/checkpoint_<step>.pkl[.gz].
    A run interrupted and resumed without --wandb_resume_run spreads its
    checkpoints over several <wandb_run_id> subdirectories, so every
    subdirectory is scanned and the highest update step wins (ties broken by
    mtime). Half-written *.tmp files are excluded by the extension check.
    """
    candidates = []
    for path in glob.glob(os.path.join(output_path, "*", "checkpoint_*.pkl*")):
        filename = os.path.basename(path)
        if not (filename.endswith(".pkl") or filename.endswith(".pkl.gz")):
            continue
        try:
            step = int(filename[len("checkpoint_"):].split(".")[0])
        except ValueError:
            continue
        candidates.append((step, os.path.getmtime(path), path))
    return max(candidates)[2] if candidates else None


def get_rnn_hidden_initializer(name):
    """Map a --rnn_hidden_initializer string to a flax initializer for the RNN's
    hidden-to-hidden (recurrent) kernel.

    This feeds the normal flax/jax initialization route (SimpleCell's
    recurrent_kernel_init). The special manual initializations
    (--connectome_init / --connectome_zero_init) are applied after
    network.init and overwrite the recurrent kernel, so they take precedence
    over whatever this picks - except under --no_init_diags, which leaves the
    kernel's diagonal at the value this initializer drew. 'orthogonal'
    reproduces flax's default, so the default option leaves initialization
    unchanged.
    """
    initializers = {
        'orthogonal': orthogonal(),
        'normal': normal(),
        'uniform': uniform(),
    }
    if name not in initializers:
        raise ValueError(
            "Unknown --rnn_hidden_initializer '{}'. Valid options: {}".format(
                name, ', '.join(sorted(initializers))
            )
        )
    return initializers[name]


class ScannedRNN(nn.Module):
    # Hidden size of the recurrent cell. When None it falls back to the input
    # dimensionality (valid only when the RNN is fed a preceding FC layer whose
    # width equals the hidden size). For --simpler_network the RNN takes the raw
    # observation directly, so the hidden size must be set explicitly rather than
    # inheriting the (much larger) input dimensionality.
    hidden_size: int = None
    # When > 0, truncate backprop through time: the gradient on the hidden-state
    # carry is stopped at every `truncate_period`-step boundary (t = 0, K, 2K, ...),
    # so a loss at step t backpropagates through at most `truncate_period`
    # timesteps (back to the most recent boundary). The forward pass is unchanged.
    # 0 disables truncation and traces to exactly the original full-BPTT graph.
    truncate_period: int = 0
    # Initializer for the hidden-to-hidden (recurrent) kernel. Defaults to
    # orthogonal(), matching flax's SimpleCell default, so the default leaves
    # initialization unchanged. Set via --rnn_hidden_initializer.
    recurrent_kernel_init: Callable = orthogonal()

    def __call__(self, carry, x):
        if self.truncate_period > 0:
            ins, resets = x
            stop_grad = (jnp.arange(resets.shape[0]) % self.truncate_period) == 0
            x = (ins, resets, stop_grad)
        return self._scanned_step(carry, x)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def _scanned_step(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        if self.truncate_period > 0:
            ins, resets, stop_grad = x
            # Identity in the forward pass; on boundary steps it zeroes the
            # gradient flowing from this step's carry back into the previous
            # chunk, cutting the BPTT chain there.
            rnn_state = jnp.where(
                stop_grad, jax.lax.stop_gradient(rnn_state), rnn_state
            )
        else:
            ins, resets = x
        hidden_size = self.hidden_size if self.hidden_size is not None else ins.shape[1]
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], hidden_size),
            rnn_state,
        )
        new_rnn_state, y = nn.SimpleCell(
            features=hidden_size,
            recurrent_kernel_init=self.recurrent_kernel_init,
        )(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.SimpleCell(features=hidden_size)
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
        hidden, embedding = ScannedRNN(
            hidden_size=self.config["LAYER_SIZE"],
            truncate_period=self.config.get("TRUNCATE_BACKPROP", 0),
            recurrent_kernel_init=get_rnn_hidden_initializer(
                self.config.get("RNN_HIDDEN_INITIALIZER", "orthogonal")
            ),
        )(hidden, rnn_in)

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

        return hidden, pi, jnp.squeeze(critic, axis=-1), aux

class SimpleActorCriticRNN(nn.Module):
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
        hidden, embedding = ScannedRNN(
            hidden_size=self.config["LAYER_SIZE"],
            truncate_period=self.config.get("TRUNCATE_BACKPROP", 0),
            recurrent_kernel_init=get_rnn_hidden_initializer(
                self.config.get("RNN_HIDDEN_INITIALIZER", "orthogonal")
            ),
        )(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(embedding)

        aux = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(embedding)

        return hidden, pi, jnp.squeeze(critic, axis=-1), aux

class SimplerActorCriticRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        rnn_in = (obs, dones)
        hidden, embedding = ScannedRNN(
            hidden_size=self.config["LAYER_SIZE"],
            truncate_period=self.config.get("TRUNCATE_BACKPROP", 0),
            recurrent_kernel_init=get_rnn_hidden_initializer(
                self.config.get("RNN_HIDDEN_INITIALIZER", "orthogonal")
            ),
        )(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(embedding)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(embedding)

        aux = nn.Dense(2, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(embedding)

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


# Connectome constraint load and preprocess to make targets
def load_connectome_constraints(pkl_filepath, n_neurons):
    matrix = pd.read_pickle(pkl_filepath)
    #zero_matrix = pd.read_pickle(zeros_path)

    targets = np.zeros((n_neurons, n_neurons),np.float32)
    for neuron_n in range(n_neurons):
        # Sample from the matrix
        rand_ind = random.randint(0, matrix.shape[0]-1)
        curr_syns = np.asarray(matrix.iloc[rand_ind].syn_count_list)
        # TODO normalize per-cell or globally?
        curr_syns = curr_syns / curr_syns.max()
        curr_syns.sort()
        curr_syns = np.flip(curr_syns)
        # Handle case where we have too many non-0 values
        if curr_syns.size > n_neurons:
            curr_syns = curr_syns[0:n_neurons]
        if curr_syns.size < n_neurons:
            # TODO make sure this is the right way to 0-pad
            pad_zeros = np.zeros((n_neurons - curr_syns.size), np.float32)
            padded_weights = np.concatenate([curr_syns, pad_zeros], axis=-1)
        else:
            padded_weights = curr_syns
        targets[neuron_n] = padded_weights

    return targets

def load_connectome_constraints_cellstats(pkl_filepath):
    matrix = np.load(pkl_filepath)
    # matrix is (pre_type, post_type, pre_unit, post_unit). The 2D target wants
    # row = pre_type * units_per_type + pre_unit, col = post_type * units_per_type
    # + post_unit, so the two unit axes have to be interleaved with their type
    # axes before flattening. Reshaping straight from the 4D layout instead maps
    # tile (a, b, p) to row 4a + (4b+p)//n_types, col block (4b+p) % n_types --
    # i.e. the row ends up depending on the POST type and the column on the PRE
    # unit, which permutes every tile away from the unit it constrains. The two
    # layouts coincide only when units_per_type == 1.
    targets = matrix.transpose(0, 2, 1, 3).reshape(
        matrix.shape[0] * matrix.shape[2], matrix.shape[1] * matrix.shape[3]
    )
    return targets


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

    def _restore_runner_state(fresh_runner_state, saved_leaves):
        """Replace every array leaf of the freshly initialized runner state with
        the corresponding leaf from a checkpoint. The tree structure and all
        static/aux data (optimizer, apply_fn, ...) come from the fresh state
        built under the CURRENT config, so command-line overrides of anything
        that doesn't change array shapes (lr, coefficients, logging cadence,
        total_timesteps, ...) take effect on resume. A structural conflict
        (different network size, env count, map size, repeat count, ...) shows
        up as a leaf count or shape/dtype mismatch and raises.
        """
        fresh_leaves, treedef = jax.tree_util.tree_flatten_with_path(fresh_runner_state)
        if len(fresh_leaves) != len(saved_leaves):
            raise ValueError(
                "Cannot resume: checkpoint has {} state arrays but the current "
                "configuration produces {}. The checkpoint is structurally "
                "incompatible with the requested configuration.".format(
                    len(saved_leaves), len(fresh_leaves)
                )
            )
        restored = []
        for (path, fresh_leaf), saved_leaf in zip(fresh_leaves, saved_leaves):
            fresh_leaf = jnp.asarray(fresh_leaf)
            saved_leaf = jnp.asarray(saved_leaf)
            if fresh_leaf.shape != saved_leaf.shape or fresh_leaf.dtype != saved_leaf.dtype:
                raise ValueError(
                    "Cannot resume: checkpoint state {} has shape {} / dtype {} "
                    "but the current configuration expects shape {} / dtype {}.".format(
                        jax.tree_util.keystr(path),
                        saved_leaf.shape, saved_leaf.dtype,
                        fresh_leaf.shape, fresh_leaf.dtype,
                    )
                )
            restored.append(saved_leaf)
        return jax.tree_util.tree_unflatten(treedef, restored)

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
    if config['RANDOM_START']:
        static_params.random_start = True
    static_params.map_size = (config['MAP_SIZE'],config['MAP_SIZE'])
    static_params.directional_vision = config['DIRECTIONAL_VISION']

    static_params.max_passive_mobs = config['MAX_COWS']

    if config["ENV_NAME"] == "Craftax-Classic-Symbolic-v1":
        from forageworld.craftax_classic.envs.craftax_symbolic_env import (
            CraftaxClassicSymbolicEnv,
        )

        env = CraftaxClassicSymbolicEnv()
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Classic-Pixels-v1":
        from forageworld.craftax_classic.envs.craftax_pixels_env import (
            CraftaxClassicPixelsEnv,
        )

        env = CraftaxClassicPixelsEnv()
        is_symbolic = False
    elif config["ENV_NAME"] == "Craftax-Symbolic-v1":
        from forageworld.craftax.envs.craftax_symbolic_env import CraftaxSymbolicEnv

        env = CraftaxSymbolicEnv(static_params)
        is_symbolic = True
    elif config["ENV_NAME"] == "Craftax-Pixels-v1":
        from forageworld.craftax.envs.craftax_pixels_env import CraftaxPixelsEnv

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

    # env_viz shares the batching stack with env but uses EpisodeInfoWrapper in
    # place of LogWrapper to add per-step telemetry fields needed by
    # _logging_step, and wraps FastVideoWrapper OUTSIDE of the batching so
    # rendering happens once per step (for a single env thread) instead of
    # once per parallel env.
    env_viz = EpisodeInfoWrapper(env)
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

    env_viz = FastVideoWrapper(
        env_viz,
        output_path=config['OUTPUT_PATH'],
        frames_per_file=config['FRAMES_PER_FILE'],
        do_videos=not config['NO_VIDEOS'],
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES_FOR_LR_ANNEALING"]
        )
        return config["LR"] * frac

    # Builds the training entry points. The wrapper function exists only to
    # keep one indentation level for everything that used to live inside
    # train(rng); it returns the composed train function with .init / .step /
    # .finish attributes exposing the individual phases so run_ppo can drive
    # the outer loop from Python (for checkpointing) while other callers can
    # still jit the whole thing as before.
    def _build_train():

        # INIT NETWORK
        if config['FULL_ACTION_SPACE']:
            action_space_size = env.action_space(env_params).n
        else:
            action_space_size = 17


        if config['NO_MEMORY']:
            if is_symbolic:
                network = ActorCriticSharedRep(action_space_size, config=config)
            else:
                network = ActorCriticConv(action_space_size, config=config)
        else:
            if config['SIMPLE_NETWORK']:
                network = SimpleActorCriticRNN(action_space_size, config=config)
            elif config['SIMPLER_NETWORK']:
                network = SimplerActorCriticRNN(action_space_size, config=config)
            else:
                network = ActorCriticRNN(action_space_size, config=config)

        # Second instance of the same network class for the periodic gradient
        # probe: identical param tree but full BPTT, so the probe measures the
        # network's intrinsic gradient decay even when training truncates it.
        do_grad_viz = (not config['NO_MEMORY']) and config['GRAD_VIZ_STEPS'] > 0
        if do_grad_viz:
            grad_viz_network = type(network)(
                action_space_size,
                config={'LAYER_SIZE': config['LAYER_SIZE'], 'TRUNCATE_BACKPROP': 0},
            )

        # Load connectome constraint targets
        #weight_targets = load_connectome_constraints(config['CONNECTOME_FILEPATH'], config['LAYER_SIZE'])
        if config['NO_CONNECTOME']:
            # Placeholders — never read by the loss when NO_CONNECTOME is set
            # (the Python branch in _loss_fn elides the constraint term at trace time).
            weight_targets = jnp.zeros((1,))
            connectome_block_size = 1
            connectome_targets_signed = False
        else:
            weight_targets = load_connectome_constraints_cellstats(config['CONNECTOME_FILEPATH'])
            # Downstream block size = number of units per post-synaptic cell type in the cellstats matrix.
            # mmap so this reads the .npy header only — the array itself was already
            # materialized by the loader above and is multi-hundred-MB at level_3 x 16.
            connectome_block_size = int(
                np.load(config['CONNECTOME_FILEPATH'], mmap_mode='r').shape[3]
            )
            # --fixed_connectome_targets uses the per-weight loss, which does no
            # within-block sorting. Block-sorting the control there would give it
            # a monotonic structure the real targets lack, so the scramble would
            # no longer be distributionally comparable to what it controls for.
            sort_target_blocks = not config['FIXED_CONNECTOME_TARGETS']
            if config['CONNECTOME_RANDOMIZE_TARGETS']:
                weight_targets = randomize_target_matrix(
                    weight_targets, connectome_block_size, seed=config['SEED'],
                    sort_blocks=sort_target_blocks,
                )
            if config['CONNECTOME_UNIFORM_TARGETS']:
                weight_targets = uniform_random_target_matrix(
                    weight_targets, connectome_block_size, seed=config['SEED'],
                    sort_blocks=sort_target_blocks,
                )
            # A constraint file generated with transmitter signs carries the
            # excitatory/inhibitory polarity of each target in its sign, so the
            # loss must constrain polarity rather than magnitude alone and
            # --connectome_init must not overwrite it with random signs.
            # Detected from the targets themselves (after the controls, both of
            # which keep the sign structure) so that a signed file needs no
            # extra flag and an unsigned one behaves exactly as before. This is
            # a Python bool, so every branch it guards resolves at trace time.
            connectome_targets_signed = bool(np.any(np.asarray(weight_targets) < 0))
            print(f'Connectome targets: {"signed (transmitter polarity constrained)" if connectome_targets_signed else "unsigned (magnitude only)"}')
            weight_targets = jnp.asarray(weight_targets)

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

        frozen_path = ('params', 'ScannedRNN_0', 'SimpleCell_1', 'h', 'kernel')
        if config['CONNECTOME_FREEZE']:
            # The params don't exist yet at this point (they are created per
            # repeat in init_runner_state), so hand optax a callable: it
            # resolves the labels against the actual param tree at tx.init /
            # tx.update time.
            def _freeze_labels(params):
                labels = jax.tree_util.tree_map_with_path(
                    lambda path, _: 'frozen' if tuple(p.key for p in path) == frozen_path else 'trainable',
                    params,
                )
                # frozen_path is hardcoded, so a model without a ScannedRNN (e.g.
                # --no_memory) or a flax version that renames SimpleCell_1 would
                # match nothing and silently train the whole network while
                # reporting the connectome was frozen. Fail loudly instead.
                if 'frozen' not in jax.tree_util.tree_leaves(labels):
                    raise ValueError(
                        f"--connectome_freeze found no parameter at {frozen_path}; "
                        f"available paths: "
                        f"{[tuple(p.key for p in path) for path, _ in jax.tree_util.tree_leaves_with_path(params)]}"
                    )
                return labels

            tx = optax.multi_transform(
                {'trainable': tx, 'frozen': optax.set_to_zero()},
                _freeze_labels,
            )
        elif config['CONNECTOME_FREEZE_ZEROS']:
            # 1.0 where the connectome target is non-zero (trainable), 0.0 where it
            # is zero (frozen). Multiplying the kernel's update by this mask leaves
            # zero-target entries untouched while allowing the rest to train.
            trainable_elem_mask = (weight_targets != 0).astype(weight_targets.dtype)

            def _mask_init(params):
                return optax.EmptyState()

            def _mask_update(updates, state, params=None):
                def _apply(path, leaf):
                    if tuple(p.key for p in path) == frozen_path:
                        return leaf * trainable_elem_mask
                    return leaf
                return jax.tree_util.tree_map_with_path(_apply, updates), state

            tx = optax.chain(
                tx,
                optax.GradientTransformation(_mask_init, _mask_update),
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

        def init_runner_state(rng):
            """Network/optimizer initialization plus env reset: everything the
            training loop's carry starts from. The rng consumption order
            matches the original monolithic train() exactly."""
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

            if not config['NO_CONNECTOME']:
                kernel = network_params['params']['ScannedRNN_0']['SimpleCell_1']['h']['kernel']
                init_kernel = None
                if config['CONNECTOME_ZERO_INIT']:
                    init_kernel = jnp.zeros_like(kernel)
                elif config['CONNECTOME_INIT']:
                    if connectome_targets_signed:
                        # The targets already carry a transmitter sign, which is
                        # what the loss now constrains — randomising the sign here
                        # would start every inhibitory connection on the wrong side
                        # of zero. rng is used without being split below, so
                        # skipping that draw shifts nothing downstream.
                        init_kernel = weight_targets
                    else:
                        # Unsigned targets are magnitudes only, so the sign is
                        # genuinely unspecified; pick one at random per weight.
                        random_sign_mask = jax.random.randint(rng, weight_targets.shape, 0, 2) * 2 - 1.
                        init_kernel = weight_targets * random_sign_mask
                if init_kernel is not None:
                    if config.get('NO_INIT_DIAGS', False):
                        # Keep each unit's self weight as drawn by
                        # --rnn_hidden_initializer (network.init above) and
                        # overwrite only the off-diagonal entries. The rng draw
                        # in the unsigned branch still happens over the full
                        # matrix, so toggling this leaves the rng stream, and
                        # therefore everything downstream, untouched.
                        diag = jnp.eye(kernel.shape[0], kernel.shape[1], dtype=bool)
                        init_kernel = jnp.where(diag, kernel, init_kernel)
                    network_params['params']['ScannedRNN_0']['SimpleCell_1']['h']['kernel'] = init_kernel

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

            rng, _rng = jax.random.split(rng)
            return (
                train_state,
                log_state,
                obsv,
                jnp.zeros((config["NUM_ENVS"]), dtype=bool),
                init_hstate,
                _rng,
                0,
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

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets, weight_targets):
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

                        # Compute connectome constraint loss.
                        # config['NO_CONNECTOME'] / config['FIXED_CONNECTOME_TARGETS']
                        # are Python bools, so these branches resolve at JIT trace
                        # time — the unused variants are never emitted into the
                        # compiled graph (no inner-loop cost).
                        if config['NO_CONNECTOME']:
                            constraint_loss = jnp.zeros(())
                        elif config.get('FIXED_CONNECTOME_TARGETS', False):
                            hh_weights = params['params']['ScannedRNN_0']['SimpleCell_1']['h']['kernel']
                            constraint_loss = connectome_constraint_loss_nonsorted(
                                hh_weights, weight_targets,
                                exclude_self_weights=config.get('EXCLUDE_SELF_WEIGHTS', False),
                                signed_targets=connectome_targets_signed,
                            )
                        else:
                            hh_weights = params['params']['ScannedRNN_0']['SimpleCell_1']['h']['kernel']
                            constraint_loss = connectome_constraint_loss(
                                hh_weights, weight_targets, connectome_block_size,
                                exclude_self_weights=config.get('EXCLUDE_SELF_WEIGHTS', False),
                                signed_targets=connectome_targets_signed,
                            )

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                            + config["AUX_COEF"] * aux_loss
                        )
                        if not config['NO_CONNECTOME']:
                            total_loss = total_loss + config["CONNECT_COEF"] * constraint_loss

                        return total_loss, (value_loss, loss_actor, entropy, aux_loss, constraint_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets, weight_targets,
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

            metric = jax.tree_util.tree_map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )

            to_log = metric

            value_loss, loss_actor, entropy, aux_loss, constraint_loss = loss_info[1]
            loss_log = {
                "loss_actor": loss_actor.mean(),
                "entropy": entropy.mean(),
                "aux_loss": aux_loss.mean(),
                "value_loss": value_loss.mean(),
                "constraint_loss": constraint_loss.mean(),
            }

            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, loss_log, update_step):
                    to_log = create_log_dict(metric, config)
                    to_log.update(loss_log)
                    # Log the LR the optimizer used for this update so it can
                    # be plotted against performance. Computed with the same
                    # linear_schedule the optimizer runs, evaluated at this
                    # update's optimizer step count (every minibatch step of
                    # one update falls in the same schedule bucket).
                    if config["ANNEAL_LR"]:
                        to_log["lr"] = float(linear_schedule(
                            int(update_step) * config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]
                        ))
                    else:
                        to_log["lr"] = config["LR"]
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, to_log, loss_log, update_step)

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
            info['value'] = value
            info['hidden_state'] = hstate
            info['pred_delta'] = aux
            info['delta'] = deltas_to_start
            info['entropy'] = pi.entropy().squeeze(0)
            info['log_prob'] = log_prob

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
                            'entropy', 'log_prob', 'episode_id']

            # Callback function for logging hidden states
            def write_rnn_hstate(hstate, scalars, increment=0):

                header_field_names = ['health','food','drink','energy','done','is_sleeping','is_resting','player_position_x',
                                      'player_position_y','recover','hunger','thirst','fatigue','light_level','dist_to_melee_l1',
                                      'melee_on_screen','dist_to_passive_l1','passive_on_screen','dist_to_ranged_l1',
                                      'ranged_on_screen','num_melee_nearby','num_passives_nearby','num_ranged_nearby','delta_x',
                                      'delta_y', 'pred_delta_x', 'pred_delta_y', 'num_monsters_killed', 'has_sword',
                                      'has_pick', 'held_iron', 'value', 'entropy', 'log_prob', 'episode_id']

                run_out_path = os.path.join(config['OUTPUT_PATH'], wandb.run.id)
                os.makedirs(run_out_path, exist_ok=True)
                # Assemble header for the scalar file(s)
                scalar_file_header = 'action'
                for key in header_field_names:
                    scalar_file_header += ',' + key

                # np.savetxt writes straight to an open append-mode handle, so the
                # old temp-file round-trip (write temp -> read back -> append) is
                # unnecessary. Each logging step appends one new chunk; for the
                # gzipped hstates that chunk is a new gzip member, and concatenated
                # members decompress transparently (gzip/zcat/pandas/np.loadtxt).
                for i in range(logging_threads):
                    # The hstate files are large/expensive; skip them when
                    # --no_hidden_state_csv is passed. The scalar files below
                    # are small/cheap and are always written.
                    if not config['NO_HIDDEN_STATE_CSV']:
                        out_filename_hstates = os.path.join(run_out_path, 'hstates_{}_{}.csv.gz'.format(increment, i))
                        # '%.6e' preserves ~full float32 precision while roughly
                        # halving the per-value text vs np.savetxt's default
                        # '%.18e', cutting both formatting cost and file size.
                        # compresslevel=1 keeps gzip CPU low; the float text still
                        # compresses well.
                        with gzip.open(out_filename_hstates, 'at', compresslevel=1) as out_file_hstates:
                            np.savetxt(out_file_hstates, hstate[:, i, :], delimiter=',', fmt='%.6e')
                        print('Writing log file', out_filename_hstates)
                    # Then do the same thing for the scalars (plaintext, uncompressed)
                    out_filename_scalars = os.path.join(run_out_path, 'scalars_{}_{}.csv'.format(increment, i))
                    with open(out_filename_scalars, 'a') as out_file_scalars:
                        np.savetxt(out_file_scalars, scalars[:, i, :], delimiter=',', fmt='%f',
                                   header=scalar_file_header)
                    print('Writing log file', out_filename_scalars)


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

            # Only the first `logging_threads` env columns are ever written, so
            # slice on-device before the callback. The hidden-state array is
            # (STEPS_PER_VIZ, NUM_ENVS, LAYER_SIZE) ~GB-scale; shipping all
            # NUM_ENVS columns to the host just to write one is the dominant
            # logging cost. When hstate logging is disabled we skip that transfer
            # entirely (None is an empty pytree, so nothing is moved to host).
            scalars_to_log = log_array[:, :logging_threads, :]
            if config['NO_HIDDEN_STATE_CSV']:
                hidden_states_to_log = None
            else:
                hidden_states_to_log = hidden_states[:, :logging_threads, :]
            # write_rnn_hstate always logs the (cheap) scalar CSVs and only
            # writes the (large) hstate CSVs when --no_hidden_state_csv is unset.
            jax.debug.callback(write_rnn_hstate, hidden_states_to_log, scalars_to_log, update_step)

            return runner_state, None

        # --- RNN gradient-propagation probe (--grad_viz_steps) ----------------
        # Rolls the current policy forward GRAD_VIZ_STEPS steps from the live
        # training state (purely functional: training resumes from the same
        # env/obs state, only the rng is split), then replays the first
        # GRAD_VIZ_ENVS envs through the network with full BPTT and takes the
        # gradient of the final step's value estimate w.r.t. every past
        # observation. One backward pass yields the entire magnitude-vs-lag
        # curve; only the small (steps, envs) magnitude matrix leaves the
        # device, so memory is bounded by the sliced obs sequence.

        def _write_grad_viz(mag, within_episode, update_step):
            # Flip time-major arrays so index k means "k timesteps into the
            # past" relative to the probe's final step.
            mag = np.asarray(mag)[::-1]
            within_episode = np.asarray(within_episode)[::-1].astype(bool)
            run_out_path = os.path.join(config['OUTPUT_PATH'], wandb.run.id)
            os.makedirs(run_out_path, exist_ok=True)

            lags = np.arange(mag.shape[0])
            mean_all = mag.mean(axis=1)
            n_valid = within_episode.sum(axis=1)
            mean_within = np.where(
                n_valid > 0,
                (mag * within_episode).sum(axis=1) / np.maximum(n_valid, 1),
                np.nan,
            )

            # plt.Figure (not plt.figure) keeps this off pyplot's global
            # figure registry, which is not thread-safe under debug.callback.
            fig = plt.Figure(figsize=(14, 5))
            axes = fig.subplots(1, 2, sharey=True)
            axes[0].semilogy(lags, mean_all, label='mean over probe envs')
            axes[0].semilogy(lags, mean_within, label='mean, within-episode only')
            axes[0].set_xlabel('timesteps into the past')
            axes[0].set_ylabel('|d value[T] / d obs[T - lag]| (L2)')
            axes[0].set_title(
                'RNN gradient magnitude vs BPTT depth (update {})'.format(int(update_step))
            )
            axes[0].legend()
            axes[0].grid(alpha=0.3)
            for i in range(min(4, mag.shape[1])):
                axes[1].semilogy(lags, mag[:, i], alpha=0.8, label='env {}'.format(i))
            axes[1].set_xlabel('timesteps into the past')
            axes[1].set_title('example single-env traces')
            axes[1].legend()
            axes[1].grid(alpha=0.3)
            fig.tight_layout()
            out_filename = os.path.join(
                run_out_path, 'rnn_grad_mag_{}.png'.format(int(update_step))
            )
            fig.savefig(out_filename, dpi=120)
            np.savez(
                os.path.join(run_out_path, 'rnn_grad_mag_{}.npz'.format(int(update_step))),
                grad_mag=mag,
                within_episode=within_episode,
            )
            if config['USE_WANDB']:
                resumable_wandb_log(
                    {
                        'rnn_grad_viz': wandb.Image(
                            out_filename, caption='update {}'.format(int(update_step))
                        )
                    },
                    update_step,
                    config,
                )
            print('Writing RNN gradient viz', out_filename)

        def _grad_viz(runner_state):
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state
            rng, probe_rng = jax.random.split(rng)
            n_probe = min(config['GRAD_VIZ_ENVS'], config['NUM_ENVS'])

            def _probe_step(carry, unused):
                env_state, obs, done, h, p_rng = carry
                p_rng, _rng = jax.random.split(p_rng)
                ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                h, pi, _, _ = network.apply(train_state.params, h, ac_in)
                action = pi.sample(seed=_rng).squeeze(0)
                p_rng, _rng = jax.random.split(p_rng)
                new_obs, env_state, _, new_done, _ = env.step(
                    _rng, env_state, action, env_params
                )
                # Store inputs for only the probed envs to keep memory bounded.
                return (env_state, new_obs, new_done, h, p_rng), (
                    obs[:n_probe],
                    done[:n_probe],
                )

            _, (obs_seq, done_seq) = jax.lax.scan(
                _probe_step,
                (env_state, last_obs, last_done, hstate, probe_rng),
                None,
                config['GRAD_VIZ_STEPS'],
            )

            h0 = hstate[:n_probe]

            def _probe_loss(obs_in):
                _, _, value, _ = grad_viz_network.apply(
                    train_state.params, h0, (obs_in, done_seq)
                )
                return value[-1].sum()

            grads = jax.grad(_probe_loss)(obs_seq)
            mag = jnp.sqrt(jnp.sum(jnp.square(grads), axis=-1))  # (steps, envs)

            # Mask of (s, env) pairs with no episode reset strictly after s.
            # Resets zero the gradient exactly, so the within-episode mean
            # shows pure BPTT decay undiluted by episode boundaries.
            d = done_seq.astype(jnp.float32)
            resets_after = jnp.cumsum(d[::-1], axis=0)[::-1] - d
            within_episode = resets_after == 0
            jax.debug.callback(_write_grad_viz, mag, within_episode, update_step)

            return (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            )

        # Log model weights. Called on the periodic logging cadence from
        # _update_plot, and once more from finish so the fully trained weights
        # land on disk next to the final logs.
        def save_weights_callback(weights, iter):
            weights_flat = jax.tree.flatten(weights)
            run_out_path = os.path.join(config['OUTPUT_PATH'], wandb.run.id)
            os.makedirs(run_out_path, exist_ok=True)
            # Written gzip-compressed to save disk space; the formatted weight
            # text compresses well and compresslevel=1 keeps the CPU overhead low.
            weight_filename = os.path.join(run_out_path, 'weights_fromto_{}.csv.gz'.format(iter))
            weights_params = weights['params']

            with gzip.open(weight_filename, 'wt', compresslevel=1) as weight_file:
                def save_weight_dict(curr_value, key_string=''):
                    if type(curr_value) != dict:
                        #why was this getting transposed? We want from-to ordering, not to-from
                        #np.savetxt(weight_file, np.transpose(curr_value), delimiter=',', fmt='%f', header=key_string)
                        np.savetxt(weight_file, curr_value, delimiter=',', fmt='%f', header=key_string)
                        return True
                    else:
                        for key in curr_value.keys():
                            save_weight_dict(curr_value[key], key_string + '/' + key)
                    return True

                save_weight_dict(weights_params)

            print('Saving weights in file', weight_filename)


        # Func to interleave update steps and plotting
        def _update_plot(runner_state, unused):

            # Log model weights
            jax.debug.callback(save_weights_callback, runner_state[0].params, runner_state[-1], ordered=True)

            # RNN gradient-propagation probe, on the same cadence as the other
            # visualization logging. Static Python flag: no cost when disabled.
            if do_grad_viz:
                runner_state = _grad_viz(runner_state)

            # First, log things (so we have logs for the untrained network)
            runner_state, empty = jax.lax.scan(
                partial(_logging_step, logging_threads = config["LOGGING_THREADS_PER_VIZ"]), runner_state, None, config['LOGGING_STEPS_PER_VIZ']
            )

            # Then update
            runner_state, metric = jax.lax.scan(
                _update_step, runner_state, None, config["UPDATES_PER_VIZ"]
            )

            return runner_state, metric

        def step(runner_state):
            """One outer iteration of training (periodic logging/visualization
            plus UPDATES_PER_VIZ PPO updates) — the body of the outer training
            loop. run_ppo drives this from Python so checkpoints can be taken
            between iterations."""
            return _update_plot(runner_state, None)

        def finish(runner_state):
            # Final weight dump so the fully trained network is on disk for
            # downstream analysis. _update_plot writes the weights before each
            # iteration's updates, so without this the last iteration's updates
            # would only ever exist inside the final checkpoint.
            jax.debug.callback(save_weights_callback, runner_state[0].params, runner_state[-1], ordered=True)

            # Final logging step so the last training iterations are captured in
            # the logs. _update_plot logs before each update, so without this the
            # updates from the final iteration would never be logged.
            runner_state, empty = jax.lax.scan(
                partial(_logging_step, logging_threads = config["LOGGING_THREADS_PER_VIZ"]), runner_state, None, config['LOGGING_STEPS_PER_VIZ']
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
            return runner_state

        def train(rng):
            """Whole-run entry point, semantically the original train():
            init, NUM_UPDATES outer iterations, then final logging and
            validation. run_ppo instead drives init/step/finish itself so it
            can checkpoint (and resume) between outer iterations; this
            composed form remains for callers that jit the whole run."""
            runner_state = init_runner_state(rng)
            runner_state, metric = jax.lax.scan(
                _update_plot, runner_state, None, int(config["NUM_UPDATES"])
            )
            runner_state = finish(runner_state)
            return {"runner_state": runner_state, "metric": metric}

        train.init = init_runner_state
        train.step = step
        train.finish = finish
        train.restore_runner_state = _restore_runner_state
        return train

    return _build_train()


def run_ppo(config, checkpoint=None):

    reset_batch_logs()

    if not config["JIT"]:
        jax.config.update("jax_disable_jit", True)
        print('JIT disabled')

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train = make_train(config)
    device = jax.devices()[JAX_DEVICE_INDEX]
    init_jit = jax.jit(jax.vmap(train.init), device=device)
    # The outer training loop runs here in Python rather than inside one big
    # lax.scan: every iteration then executes the same compiled step program,
    # so a run resumed from a checkpoint performs bit-for-bit the computation
    # the uninterrupted run would have (and reuses its compilation cache
    # entries). donate_argnums recycles the carry buffers like scan did.
    step_jit = jax.jit(jax.vmap(train.step), device=device, donate_argnums=0)
    finish_jit = jax.jit(jax.vmap(train.finish), device=device)

    num_outer_iters = int(config["NUM_UPDATES"])
    checkpoint_interval = config.get("CHECKPOINT_INTERVAL", 0)
    checkpoint_keep = config.get("CHECKPOINT_KEEP", 0)

    def _write_checkpoint(runner_state, outer_iter):
        # Checkpoints capture the loop carry at the start of outer iteration
        # `outer_iter` (all repeats), i.e. exactly the state --resume_from
        # restarts from. update_step is shared across repeats; read repeat 0.
        update_step = int(np.asarray(jax.device_get(runner_state[-1]))[0])
        run_out_path = os.path.join(config["OUTPUT_PATH"], wandb.run.id)
        os.makedirs(run_out_path, exist_ok=True)
        payload = {
            # Config snapshot is the baseline for --resume_from; explicit
            # command-line flags override it at resume time.
            "config": dict(config),
            "update_step": update_step,
            "outer_iter": outer_iter,
            "wandb_run_id": wandb.run.id,
            "leaves": jax.device_get(jax.tree_util.tree_leaves(runner_state)),
        }
        out_filename = os.path.join(run_out_path, "checkpoint_{}.pkl.gz".format(update_step))
        # Write-then-rename so an interrupt mid-write can't leave a truncated
        # file under the final checkpoint name.
        tmp_filename = out_filename + ".tmp"
        with gzip.open(tmp_filename, "wb", compresslevel=1) as tmp_file:
            pickle.dump(payload, tmp_file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_filename, out_filename)
        print("Saving training checkpoint", out_filename)
        if checkpoint_keep > 0:
            stale = sorted(
                (
                    fname for fname in os.listdir(run_out_path)
                    if fname.startswith("checkpoint_")
                    and (fname.endswith(".pkl") or fname.endswith(".pkl.gz"))
                ),
                key=lambda fname: int(fname[len("checkpoint_"):].split(".")[0]),
                reverse=True,
            )[checkpoint_keep:]
            for old_filename in stale:
                os.remove(os.path.join(run_out_path, old_filename))

    t0 = time.time()

    runner_state = init_jit(rngs)
    # Round-trip the freshly initialized state through host memory so its
    # avals (dtypes / weak-type flags) are identical to a checkpoint-restored
    # state's; fresh and resumed runs then trace, compile, and execute the
    # very same step program.
    runner_state = jax.tree_util.tree_map(lambda x: jnp.asarray(np.asarray(x)), runner_state)

    start_iter = 0
    if checkpoint is not None:
        runner_state = train.restore_runner_state(runner_state, checkpoint["leaves"])
        start_iter = int(checkpoint["outer_iter"])
        print(
            "Resuming from update step {} (outer iteration {}): {} of {} outer iterations remain".format(
                int(checkpoint["update_step"]), start_iter,
                max(num_outer_iters - start_iter, 0), num_outer_iters,
            )
        )

    metrics = []
    for outer_iter in range(start_iter, num_outer_iters):
        if checkpoint_interval > 0 and outer_iter % checkpoint_interval == 0:
            _write_checkpoint(runner_state, outer_iter)
        runner_state, metric = step_jit(runner_state)
        metrics.append(metric)

    # Final checkpoint: the loop carry at the end of the last training
    # iteration, i.e. the state --resume_from needs to continue (or fine-tune)
    # from a finished run. Written before finish() so that resuming from it
    # replays the final logging/validation rollouts on exactly the state the
    # original run fed them, and independently of --checkpoint_interval so a
    # run that skips periodic checkpoints still leaves its trained weights
    # behind.
    if config.get("FINAL_CHECKPOINT", True):
        _write_checkpoint(runner_state, num_outer_iters)

    runner_state = finish_jit(runner_state)
    out = {
        "runner_state": runner_state,
        # Stacked to the (repeats, iterations, updates) layout the old
        # whole-run scan returned.
        "metric": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=1), *metrics) if metrics else None,
    }

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


if __name__ == "__main__":

    args = parse_args()

    # Persistent XLA compilation cache: relaunching an identical configuration
    # skips the (multi-minute) compile of the training programs. Resumed runs
    # benefit too — the step/finish programs are identical to the original
    # run's, so resuming mostly reuses its cache entries. An externally
    # configured cache (JAX_COMPILATION_CACHE_DIR / max size) is left
    # untouched.
    if jax.config.jax_compilation_cache_dir is None:
        jax.config.update(
            "jax_compilation_cache_dir",
            os.path.expanduser("~/.cache/jax_comp_cache"),
        )
    if jax.config.jax_compilation_cache_max_size < 0:
        jax.config.update("jax_compilation_cache_max_size", 5 * 2**30)

    # --auto_resume: pick up where a previous leg of this run left off, if
    # there is anything to pick up. The scan happens before checkpoint loading
    # so the discovered path flows through the ordinary --resume_from path.
    auto_resumed = False
    if args.auto_resume and args.resume_from is None:
        latest_checkpoint = find_latest_checkpoint(args.output_path)
        if latest_checkpoint is not None:
            args.resume_from = latest_checkpoint
            auto_resumed = True
        else:
            print("Auto-resume: no checkpoint found under", args.output_path, "- starting fresh")

    checkpoint = None
    if args.resume_from is not None:
        # Checkpoints are gzip-compressed pickles; sniff the magic bytes so
        # uncompressed checkpoints from older runs keep working.
        with open(args.resume_from, "rb") as checkpoint_file:
            is_gzip = checkpoint_file.read(2) == b"\x1f\x8b"
        opener = gzip.open if is_gzip else open
        with opener(args.resume_from, "rb") as checkpoint_file:
            checkpoint = pickle.load(checkpoint_file)
        # The checkpoint's stored configuration is the baseline; flags
        # explicitly typed on this command line override it. Options added to
        # the code after the checkpoint was written fall back to their current
        # defaults. Derived values make_train computes (NUM_UPDATES, ...) are
        # recomputed from the merged config, and overrides that conflict with
        # the stored state's shapes fail at restore time in make_train.
        config = dict(checkpoint["config"])
        explicit_args = parse_explicit_args()
        for key, value in explicit_args.items():
            config[key.upper()] = value
        for key, value in vars(args).items():
            config.setdefault(key.upper(), value)
        if auto_resumed and "wandb_resume_run" not in explicit_args:
            # Auto-resume continues the original WandB run by default. The
            # checkpoint's stored WANDB_RESUME_RUN (False on the first leg)
            # must not override that; only an explicit --no-wandb_resume_run
            # on this command line may.
            config["WANDB_RESUME_RUN"] = True
        print("Resuming run from checkpoint", args.resume_from)
    else:
        config = {key.upper(): value for key, value in vars(args).items()}

    # --wandb_resume_run: continue the WandB run that wrote the checkpoint
    # instead of starting a new one, so long checkpointed trainings produce a
    # single continuous set of curves.
    wandb_init_kwargs = {}
    if checkpoint is not None and config.get("WANDB_RESUME_RUN"):
        resume_run_id = checkpoint.get("wandb_run_id")
        if not resume_run_id:
            raise ValueError(
                "--wandb_resume_run: the checkpoint does not record a wandb run id"
            )
        # "must" fails loudly if the run can't be found/attached, rather than
        # silently forking a fresh run with a broken step axis.
        wandb_init_kwargs = dict(id=resume_run_id, resume="must")
        print("Continuing WandB run", resume_run_id)

    wandb.init(
        project=config["WANDB_PROJECT"],
        entity=config["WANDB_ENTITY"],
        config=config,
        name=config["RUN_NAME"],
        **wandb_init_kwargs,
    )

    if wandb_init_kwargs:
        # Push the merged (checkpoint + CLI overrides) config onto the resumed
        # run, and record where its history ends: resumable_wandb_log skips
        # re-logging update steps below WANDB_RESUME_STEP0 (the resumed run
        # reproduces that overlap bit-identically, and wandb rejects
        # non-increasing steps anyway).
        wandb.config.update(dict(config), allow_val_change=True)
        wandb.config.update(
            {"WANDB_RESUME_STEP0": int(wandb.run.step)}, allow_val_change=True
        )

    if checkpoint is not None:
        # wandb.Config rejects item assignment that changes an existing value,
        # and a resumed config already carries the derived keys (NUM_UPDATES,
        # MINIBATCH_SIZE, ...) the checkpoint stored — which make_train
        # recomputes and writes back, and which legitimately change whenever
        # the resume overrides what they derive from (fine-tuning a finished
        # run from its final checkpoint with an extended --total_timesteps is
        # exactly that). Hand run_ppo a plain-dict copy so those writes don't
        # trip wandb's immutability; the wandb UI keeps the original run's
        # derived values.
        run_config = dict(wandb.config)
    else:
        run_config = wandb.config

    run_ppo(run_config, checkpoint)

    # Training ran to its final update (as opposed to being killed by the
    # wallclock limit): leave a done marker for the auto-resume manager
    # (utils/auto_resume.py, which provides the state dir via sbatch --export)
    # so it stops submitting continuation jobs for this (array) task.
    auto_resume_state_dir = os.environ.get("PPO_AUTO_RESUME_STATE_DIR")
    if auto_resume_state_dir:
        os.makedirs(auto_resume_state_dir, exist_ok=True)
        marker_name = "done_{}".format(os.environ.get("SLURM_ARRAY_TASK_ID", "noarray"))
        with open(os.path.join(auto_resume_state_dir, marker_name), "w") as marker_file:
            marker_file.write("finished {} wandb_run={}\n".format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                wandb.run.id if wandb.run is not None else "unknown",
            ))