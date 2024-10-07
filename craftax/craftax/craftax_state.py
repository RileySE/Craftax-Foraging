from dataclasses import field
from typing import Tuple, Any

import jax
from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Inventory:
    wood: int
    stone: int
    coal: int
    iron: int
    diamond: int
    sapling: int
    pickaxe: int
    sword: int
    bow: int
    arrows: int
    armour: jnp.ndarray
    torches: int
    ruby: int
    sapphire: int
    potions: jnp.ndarray
    books: int


@struct.dataclass
class Mobs:
    position: jnp.ndarray
    health: jnp.ndarray
    mask: jnp.ndarray
    attack_cooldown: jnp.ndarray
    type_id: jnp.ndarray


@struct.dataclass
class EnvState:
    env_id: int
    map: jnp.ndarray
    item_map: jnp.ndarray
    mob_map: jnp.ndarray
    light_map: jnp.ndarray
    down_ladders: jnp.ndarray
    up_ladders: jnp.ndarray
    chests_opened: jnp.ndarray
    monsters_killed: jnp.ndarray

    player_position: jnp.ndarray
    player_level: int
    player_direction: int
    # Initial coordinates of the player at start of the episode
    player_starting_position: jnp.ndarray

    # Intrinsics
    player_health: float
    player_food: int
    player_drink: int
    player_energy: int
    player_mana: int
    is_sleeping: bool
    is_resting: bool

    # Second order intrinsics
    player_recover: float
    player_hunger: float
    player_thirst: float
    player_fatigue: float
    player_recover_mana: float

    # Attributes
    player_xp: int
    player_dexterity: int
    player_strength: int
    player_intelligence: int

    inventory: Inventory

    melee_mobs: Mobs
    passive_mobs: Mobs
    ranged_mobs: Mobs

    mob_projectiles: Mobs
    mob_projectile_directions: jnp.ndarray
    player_projectiles: Mobs
    player_projectile_directions: jnp.ndarray

    growing_plants_positions: jnp.ndarray
    growing_plants_age: jnp.ndarray
    growing_plants_mask: jnp.ndarray

    potion_mapping: jnp.ndarray
    learned_spells: jnp.ndarray

    sword_enchantment: int
    bow_enchantment: int
    armour_enchantments: jnp.ndarray

    boss_progress: int
    boss_timesteps_to_spawn_this_round: int

    light_level: float

    achievements: jnp.ndarray

    state_rng: Any

    timestep: int

    level: int

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)

    # moved for curriculum
    level: int = 1
    max_melee_mobs: int = 2
    max_ranged_mobs: int = 3
    max_melee_spawn_chance: float = 0.02  # 0.02
    max_ranged_spawn_chance: float = 0.05  # 0.05
    @property
    def floor_mob_spawn_chance(self) -> jnp.ndarray:
        return jnp.array(
            [
                jnp.array([0.1, self.max_melee_spawn_chance, self.max_ranged_spawn_chance, 0.0]),  # Floor 0 (overworld)
                jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 1 (gnomish mines)
                jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 2 (dungeon)
                jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 3 (sewers)
                jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 4 (vaults)
                jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 5 (troll mines)
                jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 6 (fire)
                jnp.array([0.0, 0.06, 0.05, 0.0]),  # Floor 7 (ice)
                jnp.array([0.1, 0.06, 0.05, 0.0]),  # Floor 8 (boss)
            ],
            dtype=jnp.float32,
        )



@struct.dataclass
class EnvParams:

    max_timesteps: int = 100000
    day_length: int = 300

    melee_mob_health: int = 5
    passive_mob_health: int = 3
    ranged_mob_health: int = 3

    mob_despawn_distance: int = 14  # increased for curriculum
    max_attribute: int = 5

    god_mode: bool = False

    fractal_noise_angles: tuple[int, int, int, int] = (None, None, None, None)


# HACK: Removed the static "struct.dataclass" declaration. Hope that didn't break any optimizations!
class StaticEnvParams:
    map_size: Tuple[int, int] = (96, 96)
    num_levels: int = 9
    reward_func: str = 'foraging'

    # Mobs
    # HACK: Doubled to 18 for patch depletion stuff
    max_passive_mobs: int = 18
    max_growing_plants: int = 30
    max_mob_projectiles: int = 3
    max_player_projectiles: int = 3
