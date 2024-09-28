from math import ceil, sqrt

import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from matplotlib import pyplot as plt, animation

from craftax.craftax.renderer import render_craftax_pixels


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
            state = jax.tree_map(
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
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
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


# Wrapper for plotting videos (every expensive op due to CPU latency, only run with this wrapper rarely!
class VideoPlotWrapper(LogWrapper):
    def __init__(self, env: environment.Environment, output_path='./', frames_per_file=500, do_videos=True):
        super().__init__(env)
        self.vis_renderer = None
        self.curr_env_id = -9999
        self.n_frames_seen = 0
        self.output_path = output_path
        self.frames_per_file = frames_per_file
        self.do_videos = do_videos

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = super().reset(key, params)
        # Either flush the video or do setup (if this is the first reset)
        if self.do_videos:
            if self.vis_renderer:
                self.vis_renderer.flush_video()
            else:
                example_frame = render_craftax_pixels(state.env_state, 16)
                self.vis_renderer = VisualizationRenderer(example_frame.shape, self.output_path, 0, True,
                                                          frames_per_file=self.frames_per_file)

        return obs, state

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

        # This needs to leave the jax ecosystem so we use callback
        def callback_func(new_obs, t, done):
            #if self.do_videos:
            self.vis_renderer.add_frame(new_obs, t, done)


        # Alternative to satisfy jax's cond function
        # TODO is there a builtin no-op function I could use instead?
        def null_func(env_state):
            return 0

        def do_callback(env_state):
            new_obs = render_craftax_pixels(env_state, 16)
            jax.debug.callback(callback_func, new_obs, state.env_state.env_id, done)
            return 1


        #is_new_env_id = jnp.logical_or(self.curr_env_id == -9999, self.n_frames_seen >= 10000000)


        #self.n_frames_seen = (self.n_frames_seen + 1) % 10000001
        # TODO disable callback for non-rendered envs, it incurs a huge performance penalty
        # TODO this equality test isn't working for some reason, it always returns true
        #is_callback = jnp.logical_and(self.do_videos, self.curr_env_id == this_env_id)
        #is_callback = self.curr_env_id == this_env_id

        def debug_print(array_of_stuff):
            for thing in array_of_stuff:
                print(thing)
            return np.int32(0)

        this_env_id = env_state.env_id.squeeze()
        # This is false
        #is_new_env_id = this_env_id == self.curr_env_id
        # EVALUATES FALSE BRANCH
        #self.curr_env_id = jax.lax.select(is_new_env_id, this_env_id, self.curr_env_id)
        # EVALUATES TRUE BRANCH, WTF
        callback_result = jax.lax.cond(self.do_videos, do_callback, null_func, env_state)
        #result = jax.pure_callback(debug_print, np.int32(0), is_new_env_id, self.curr_env_id, this_env_id])



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
        info['thirst'] = env_state.player_thirst
        info['fatigue'] = env_state.player_fatigue
        info['light_level'] = env_state.light_level
        # TODO why is this an array? It's supposed to be an int...
        info['episode_id'] = env_state.env_id.squeeze()

        melee_pos = env_state.melee_mobs.position[env_state.player_level]
        melee_mask = env_state.melee_mobs.mask[env_state.player_level]

        dists_to_melee = jnp.linalg.norm(env_state.player_position - melee_pos, ord=1, axis = -1)
        dists_to_melee = jnp.where(melee_mask, dists_to_melee, jnp.inf)
        closest_melee_idx = jnp.argmin(dists_to_melee)
        closest_melee_dist_xy = env_state.player_position - melee_pos[closest_melee_idx]
        melee_on_screen = jnp.logical_and(jnp.abs(closest_melee_dist_xy[0]) <= 5, jnp.abs(closest_melee_dist_xy[1]) <= 4)
        melee_on_screen = jnp.logical_and(melee_on_screen, melee_mask[closest_melee_idx])
        dist_to_melee = dists_to_melee[closest_melee_idx]

        passive_pos = env_state.passive_mobs.position[env_state.player_level]
        passive_mask = env_state.passive_mobs.mask[env_state.player_level]

        dists_to_passive = jnp.linalg.norm(env_state.player_position - passive_pos, ord=1, axis = -1)
        dists_to_passive = jnp.where(passive_mask, dists_to_passive, jnp.inf)
        closest_passive_idx = jnp.argmin(dists_to_passive)
        closest_passive_dist_xy = env_state.player_position - passive_pos[closest_passive_idx]
        passive_on_screen = jnp.logical_and(jnp.abs(closest_passive_dist_xy[0]) <= 5, jnp.abs(closest_passive_dist_xy[1]) <= 4)
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

        info['dist_to_melee_l1'] = dist_to_melee
        info['dist_to_passive_l1'] = dist_to_passive
        info['dist_to_ranged_l1'] = dist_to_ranged
        info['melee_on_screen'] = melee_on_screen
        info['passive_on_screen'] = passive_on_screen
        info['ranged_on_screen'] = ranged_on_screen
        info['num_melee_nearby'] = num_melee_nearby
        info['num_passives_nearby'] = num_passives_nearby
        info['num_ranged_nearby'] = num_ranged_nearby

        return obs, state, reward, done, info


# Class to progressively render visualization frames during test rollouts.
# TODO remove residual cruft
class VisualizationRenderer(object):
    # Set up plotting
    def __init__(self, frame_shape, save_path, enumerator, is_rgb=False, draw_only_first=False, frames_per_file=500):
        self.frame_shape = frame_shape
        self.save_path = save_path
        self.enumerator = enumerator
        self.is_rgb = is_rgb
        self.draw_only_first = draw_only_first
        self.frames_per_file = frames_per_file

        # frame_shape should be shaped like <x, y, n_channels>

        # Simple grid layout: n-by-n grid, possibly underfull
        self.side_length = 1

        # Determine figure aspect ratio
        self.obs_x = frame_shape[0]
        self.obs_y = frame_shape[1]

        self.fig = plt.figure(figsize=(ceil(self.obs_y / 100), ceil(self.obs_x / 100)))
        self.axs = self.fig.subplots(self.side_length, self.side_length, squeeze=(not draw_only_first))

        # ims is a list of lists, each row is a list of artists to draw in the
        # current frame; here we are just animating one artist, the image, in each frame
        self.ims = []

        self.n_frames_logged = 0
        self.last_timestep = 0
        self.n_videos_logged = 0
        self.key = None
        self.add_frame_callcount = 0

    # Render a new frame
    # Frames should have the shape described in frame_shape, with the first dimension being (usually) 1
    def add_frame(self, frame, timestep, done):

        self.add_frame_callcount += 1

        # If we finished, grab a different episode to log
        if self.add_frame_callcount % 1000000 == 0:
            self.key = None

        if not self.key:
            self.key = timestep
        # Attempt to log only one parallel env
        if timestep != self.key:
            return

        if self.n_frames_logged % 50 == 0:
            print('Logged', self.n_frames_logged, 'frames')
        #self.last_timestep = timestep
        curr_artist = []
        frame = frame / 255.
        # Draw the frame!
        im = self.axs.imshow(frame, animated=True, vmin=0, vmax=1)
        self.axs.set_xticks([])
        self.axs.set_yticks([])
        curr_artist.append(im)
        self.ims.append(curr_artist)
        self.n_frames_logged += 1

        if self.n_frames_logged >= self.frames_per_file:
            self.flush_video()

    # Write out the rendered frames as an mp4 using ffmpeg
    def flush_video(self):

        print('Flushing', len(self.ims), 'frames')
        # Animate/render the set of frames
        ani = animation.ArtistAnimation(self.fig, self.ims, interval=200, blit=True,
                                        repeat_delay=1000, repeat=False)

        # Pipe to ffmpeg for encoding and writing to disk
        writer = animation.FFMpegWriter(
            fps=10, bitrate=-1, codec='hevc_nvenc')
        ani.save(self.save_path + "/example_episode_" + str(self.n_videos_logged) + ".mp4", writer=writer)

        plt.close()

        self.last_timestep = 0
        self.ims = []
        self.n_frames_logged = 0
        self.fig = plt.figure(figsize=(ceil(self.obs_y / 100), ceil(self.obs_x / 100)))
        self.axs = self.fig.subplots(self.side_length, self.side_length, squeeze=(not self.draw_only_first))
        self.n_videos_logged += 1
