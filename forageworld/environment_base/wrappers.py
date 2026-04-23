import os
import subprocess

import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces

from forageworld.craftax.renderer import render_craftax_pixels


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env: environment.Environment, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


# Wrapper to restrict to only the first 17 actions (the first floor stuff, basically)
class ReduceActionSpaceWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)
        self.action_space().n = 17

# Wrapper to add the action to the observation vector. Simple(?)
class AppendActionToObsWrapper(GymnaxWrapper):

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
        ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        obs, env_state, reward, done, info = self._env.step(key, state, action, params)

        obs = jnp.concatenate([obs, action.reshape(action.shape + (1,))], axis=0)

        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:

        obs, env_state = self._env.reset(key, params)
        obs = jnp.concatenate([obs, jnp.zeros((1,))], axis=0)

        return obs, env_state



class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)

        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
                                     + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                     + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


# Wrapper that mirrors LogWrapper but also populates per-episode info fields
# (health, food, distance-to-mob, etc.) used downstream for telemetry logging.
# Expected to sit INSIDE the batching wrappers (same position as LogWrapper),
# because the info-field indexing assumes a single, unbatched env state.
class EpisodeInfoWrapper(LogWrapper):
    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = super().step(key, state, action, params)

        env_state = state.env_state

        # Add fields to be logged
        info['action'] = action
        info['health'] = env_state.player_health
        info['food'] = env_state.player_food
        info['drink'] = env_state.player_drink
        info['energy'] = env_state.player_energy
        info['done'] = done
        info['is_sleeping'] = env_state.is_sleeping
        info['is_resting'] = env_state.is_resting
        info['player_position_x'] = env_state.player_position[0]
        info['player_position_y'] = env_state.player_position[1]
        info['recover'] = env_state.player_recover
        info['hunger'] = env_state.player_hunger
        info['thirst'] = env_state.player_thirst  # print("A " + str(type(log_state)))
        info['fatigue'] = env_state.player_fatigue
        info['light_level'] = env_state.light_level
        # TODO why is this an array? It's supposed to be an int...
        info['episode_id'] = env_state.env_id.squeeze()

        melee_pos = env_state.melee_mobs.position[env_state.player_level]
        melee_mask = env_state.melee_mobs.mask[env_state.player_level]

        dists_to_melee = jnp.linalg.norm(env_state.player_position - melee_pos, ord=1, axis=-1)
        dists_to_melee = jnp.where(melee_mask, dists_to_melee, jnp.inf)
        closest_melee_idx = jnp.argmin(dists_to_melee)
        closest_melee_dist_xy = env_state.player_position - melee_pos[closest_melee_idx]
        melee_on_screen = jnp.logical_and(jnp.abs(closest_melee_dist_xy[0]) <= 5,
                                          jnp.abs(closest_melee_dist_xy[1]) <= 4)
        melee_on_screen = jnp.logical_and(melee_on_screen, melee_mask[closest_melee_idx])
        dist_to_melee = dists_to_melee[closest_melee_idx]

        passive_pos = env_state.passive_mobs.position[env_state.player_level]
        passive_mask = env_state.passive_mobs.mask[env_state.player_level]

        dists_to_passive = jnp.linalg.norm(env_state.player_position - passive_pos, ord=1, axis=-1)
        dists_to_passive = jnp.where(passive_mask, dists_to_passive, jnp.inf)
        closest_passive_idx = jnp.argmin(dists_to_passive)
        closest_passive_dist_xy = env_state.player_position - passive_pos[closest_passive_idx]
        passive_on_screen = jnp.logical_and(jnp.abs(closest_passive_dist_xy[0]) <= 5,
                                            jnp.abs(closest_passive_dist_xy[1]) <= 4)
        passive_on_screen = jnp.logical_and(passive_on_screen, passive_mask[closest_passive_idx])
        dist_to_passive = dists_to_passive[closest_passive_idx]

        ranged_pos = env_state.ranged_mobs.position[env_state.player_level]
        ranged_mask = env_state.ranged_mobs.mask[env_state.player_level]

        dists_to_ranged = jnp.linalg.norm(env_state.player_position - ranged_pos, ord=1, axis = -1)
        dists_to_ranged = jnp.where(ranged_mask, dists_to_ranged, jnp.inf)
        closest_ranged_idx = jnp.argmin(dists_to_ranged)
        closest_ranged_dist_xy = env_state.player_position - ranged_pos[closest_ranged_idx]
        ranged_on_screen = jnp.logical_and(jnp.abs(closest_ranged_dist_xy[0]) <= 5, jnp.abs(closest_ranged_dist_xy[1]) <= 4)
        ranged_on_screen = jnp.logical_and(ranged_on_screen, ranged_mask[closest_ranged_idx])
        dist_to_ranged = dists_to_ranged[closest_ranged_idx]

        # Slightly bigger radius than the screen, basically what mobs is the agent able to quickly interact with
        nearby_distance = 9
        num_melee_nearby = (dists_to_melee <= nearby_distance).sum()
        num_passives_nearby = (dists_to_passive <= nearby_distance).sum()
        num_ranged_nearby = (dists_to_ranged <= nearby_distance).sum()

        num_monsters_killed = env_state.monsters_killed[env_state.player_level]

        info['dist_to_melee_l1'] = dist_to_melee
        info['dist_to_passive_l1'] = dist_to_passive
        info['dist_to_ranged_l1'] = dist_to_ranged
        info['melee_on_screen'] = melee_on_screen
        info['passive_on_screen'] = passive_on_screen
        info['ranged_on_screen'] = ranged_on_screen
        info['num_melee_nearby'] = num_melee_nearby
        info['num_passives_nearby'] = num_passives_nearby
        info['num_ranged_nearby'] = num_ranged_nearby
        info['num_monsters_killed'] = num_monsters_killed
        info['has_sword'] = env_state.inventory.sword
        info['has_pick'] = env_state.inventory.pickaxe
        info['held_iron'] = env_state.inventory.iron

        return obs, state, reward, done, info


# Lightweight video recorder that pipes raw RGB frames straight into an ffmpeg
# subprocess. Waits for the first episode boundary before starting a file so
# that each recorded video begins at a fresh episode. Replaces the prior
# matplotlib ArtistAnimation path, which was dominated by imshow/agg overhead.
class FastVideoRenderer(object):
    def __init__(self, frame_shape, save_path, frames_per_file=500, fps=10,
                 codec='libx264', file_prefix='example_episode'):
        self.height = int(frame_shape[0])
        self.width = int(frame_shape[1])
        self.save_path = save_path
        self.frames_per_file = frames_per_file
        self.fps = fps
        self.codec = codec
        self.file_prefix = file_prefix

        os.makedirs(save_path, exist_ok=True)

        self._proc = None
        self._recording = False
        self._frames_in_file = 0
        self._videos_written = 0

    def _open_writer(self):
        out_path = os.path.join(
            self.save_path, f'{self.file_prefix}_{self._videos_written}.mp4'
        )
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-',
            '-an',
            '-vcodec', self.codec,
            '-pix_fmt', 'yuv420p',
            out_path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print('Recording video to', out_path)

    def _close_writer(self):
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            self._proc.wait(timeout=30)
        except Exception:
            self._proc.kill()
        self._proc = None
        self._videos_written += 1

    def add_frame(self, frame, done):
        # Wait for the first episode boundary before recording, so every video
        # file starts at a fresh episode. Under the auto-reset wrappers, the
        # state observed on a done=True step is already the new episode's
        # reset state, so it's a valid frame zero.
        if not self._recording:
            if bool(done):
                self._open_writer()
                self._recording = True
                self._frames_in_file = 0
            else:
                return

        self._write_frame(frame)
        self._frames_in_file += 1

        if self._frames_in_file >= self.frames_per_file:
            self._close_writer()
            self._recording = False

    def _write_frame(self, frame):
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        frame = np.ascontiguousarray(frame)
        try:
            self._proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            # ffmpeg died; drop remaining frames in this file
            self._proc = None
            self._recording = False

    def close(self):
        if self._recording:
            self._close_writer()
            self._recording = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# Fast video-logging wrapper. Renders only a single environment thread per
# step (env_idx, default 0) rather than every parallel thread, emits one
# host callback per step, and writes videos directly via ffmpeg rather than
# through matplotlib. Apply OUTSIDE of OptimisticResetVecEnvWrapper /
# BatchEnvWrapper so state is already batched and env_idx can be sliced out.
class FastVideoWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment, output_path='./',
                 frames_per_file=500, do_videos=True,
                 env_idx=0, block_pixel_size=16, fps=10):
        super().__init__(env)
        self.output_path = output_path
        self.frames_per_file = frames_per_file
        self.do_videos = do_videos
        self.env_idx = env_idx
        self.block_pixel_size = block_pixel_size
        self.fps = fps
        self._renderer = None  # lazily created on first host callback

    def _emit_frame(self, frame, done):
        frame = np.asarray(frame)
        if self._renderer is None:
            self._renderer = FastVideoRenderer(
                frame.shape, self.output_path,
                frames_per_file=self.frames_per_file, fps=self.fps,
            )
        self._renderer.add_frame(frame, done)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ):
        obs, state, reward, done, info = self._env.step(key, state, action, params)

        if self.do_videos:
            # The wrapped env may be LogWrapped; drill through to the raw env
            # state before rendering. Slice out a single parallel env so we
            # render one frame per step instead of num_envs frames per step.
            raw = state.env_state if hasattr(state, 'env_state') else state
            single_state = jax.tree.map(lambda x: x[self.env_idx], raw)
            single_done = done[self.env_idx]
            frame = render_craftax_pixels(single_state, self.block_pixel_size)
            frame_u8 = jnp.clip(frame, 0, 255).astype(jnp.uint8)
            jax.debug.callback(self._emit_frame, frame_u8, single_done, ordered=True)

        return obs, state, reward, done, info


class CurriculumWrapper(GymnaxWrapper):

    def __init__(self, env: environment.Environment,
                 num_envs: int,
                 num_steps: int,
                 use_curriculum: bool,
                 predators: bool,
                 ):
        super().__init__(env)

        self.num_envs = num_envs
        self.total_steps = num_steps
        self.use_curriculum = use_curriculum
        self.predators = predators

        self.num_levels = 5

    def disable_predators(self, log_state):
        batched_max_melee_mobs = jnp.full((self.num_envs,), 0, dtype=jnp.int32)
        batched_max_ranged_mobs = jnp.full((self.num_envs,), 0, dtype=jnp.int32)

        env_state = log_state.env_state
        env_state = env_state.replace(max_melee_mobs=batched_max_melee_mobs,
                                      max_ranged_mobs=batched_max_ranged_mobs)
        return log_state.replace(env_state=env_state)

    def reset(
            self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, LogEnvState]:
        obs, log_state = self._env.reset(rng, params)

        log_state = jax.lax.cond(
            not self.predators,
            lambda ls: self.disable_predators(log_state),
            lambda ls: log_state,
            log_state
        )

        return obs, log_state

    @partial(jax.jit, static_argnums=(0, 5))
    def step(
            self,
            key: chex.PRNGKey,
            log_state: LogEnvState,
            action: Union[int, float],
            update_step: int,
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:

        obs, log_state, reward, done, info = self._env.step(key, log_state, action, params)

        def update_curriculum(log_state, update_step):
            state = log_state.env_state

            # update level
            level = jnp.floor(update_step * self.num_levels / self.total_steps + 1).astype(jnp.int32)
            batched_level = jnp.full((self.num_envs,), level)
            state = state.replace(level=batched_level)

            # update spawn chances
            level_melee_spawn_chance = level / self.num_levels * state.max_melee_spawn_chance
            level_ranged_spawn_chance = level / self.num_levels * state.max_ranged_spawn_chance
            state = state.replace(max_melee_spawn_chance=level_melee_spawn_chance,
                                  max_ranged_spawn_chance=level_ranged_spawn_chance)

            # update max mobs
            max_melee_mobs = level / self.num_levels * 10
            max_ranged_mobs = level / self.num_levels * 10
            batched_max_melee_mobs = jnp.full((self.num_envs,), max_melee_mobs, dtype=jnp.int32)
            batched_max_ranged_mobs = jnp.full((self.num_envs,), max_ranged_mobs, dtype=jnp.int32)
            state = state.replace(max_melee_mobs=batched_max_melee_mobs,
                                  max_ranged_mobs=batched_max_ranged_mobs)

            return log_state.replace(env_state=state)

        log_state = jax.lax.cond(
            self.use_curriculum,
            lambda ls: update_curriculum(log_state, update_step),
            lambda ls: log_state,
            log_state
        )

        log_state = jax.lax.cond(
            not self.predators,
            lambda ls: self.disable_predators(log_state),
            lambda ls: log_state,
            log_state
        )


        return obs, log_state, reward, done, info