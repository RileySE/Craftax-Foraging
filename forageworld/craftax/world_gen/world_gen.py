from random import randint

import jax
import jax.scipy as jsp

from forageworld.craftax.constants import *
from forageworld.craftax.game_logic import calculate_light_level, get_distance_map
from forageworld.craftax.craftax_state import EnvState, Inventory, Mobs
from forageworld.craftax.util.noise import generate_fractal_noise_2d
from forageworld.craftax.world_gen.world_gen_configs import (
    ALL_DUNGEON_CONFIGS,
    ALL_SMOOTHGEN_CONFIGS,
    FEATURELESS_SMOOTHGEN_CONFIGS,
)


def get_new_empty_inventory():
    return Inventory(
        wood=jnp.asarray(0, dtype=jnp.int32),
        stone=jnp.asarray(0, dtype=jnp.int32),
        coal=jnp.asarray(0, dtype=jnp.int32),
        iron=jnp.asarray(0, dtype=jnp.int32),
        diamond=jnp.asarray(0, dtype=jnp.int32),
        sapling=jnp.asarray(0, dtype=jnp.int32),
        pickaxe=jnp.asarray(0, dtype=jnp.int32),
        sword=jnp.asarray(0, dtype=jnp.int32),
        bow=jnp.asarray(0, dtype=jnp.int32),
        arrows=jnp.asarray(0, dtype=jnp.int32),
        torches=jnp.asarray(0, dtype=jnp.int32),
        ruby=jnp.asarray(0, dtype=jnp.int32),
        sapphire=jnp.asarray(0, dtype=jnp.int32),
        books=jnp.asarray(0, dtype=jnp.int32),
        potions=jnp.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=jnp.int32,
        ),
        armour=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
    )


def get_new_full_inventory():
    return Inventory(
        wood=jnp.asarray(99, dtype=jnp.int32),
        stone=jnp.asarray(99, dtype=jnp.int32),
        coal=jnp.asarray(99, dtype=jnp.int32),
        iron=jnp.asarray(99, dtype=jnp.int32),
        diamond=jnp.asarray(99, dtype=jnp.int32),
        sapling=jnp.asarray(99, dtype=jnp.int32),
        pickaxe=jnp.asarray(4, dtype=jnp.int32),
        sword=jnp.asarray(4, dtype=jnp.int32),
        bow=jnp.asarray(1, dtype=jnp.int32),
        arrows=jnp.asarray(99, dtype=jnp.int32),
        torches=jnp.asarray(99, dtype=jnp.int32),
        ruby=jnp.asarray(99, dtype=jnp.int32),
        sapphire=jnp.asarray(99, dtype=jnp.int32),
        books=jnp.asarray(99, dtype=jnp.int32),
        potions=jnp.array(
            [
                99,
                99,
                99,
                99,
                99,
                99,
            ],
            dtype=jnp.int32,
        ),
        armour=jnp.array([2, 2, 2, 2], dtype=jnp.int32),
    )


def generate_dungeon(rng, static_params, config):
    chunk_size = 16
    world_chunk_width = static_params.map_size[0] // chunk_size
    world_chunk_height = static_params.map_size[1] // chunk_size
    room_occupancy_chunks = jnp.ones(world_chunk_width * world_chunk_height)

    num_rooms = 8
    min_room_size = 5
    max_room_size = 10

    rng, _rng, __rng = jax.random.split(rng, 3)
    room_sizes = jax.random.randint(
        __rng, shape=(num_rooms, 2), minval=min_room_size, maxval=max_room_size
    )

    map = jnp.ones(static_params.map_size, dtype=jnp.int32) * BlockType.WALL.value
    padded_map = jnp.pad(map, max_room_size, constant_values=0)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)
    padded_item_map = jnp.pad(item_map, max_room_size, constant_values=0)

    def _add_room(carry, room_index):
        block_map, item_map, room_occupancy_chunks, rng = carry

        rng, _rng = jax.random.split(rng)
        room_chunk = jax.random.choice(
            _rng,
            jnp.arange(world_chunk_width * world_chunk_height),
            p=room_occupancy_chunks,
        )
        room_occupancy_chunks = room_occupancy_chunks.at[room_chunk].set(0)

        room_position = jnp.array(
            [
                (room_chunk % world_chunk_height) * chunk_size,
                (room_chunk // world_chunk_height) * chunk_size,
            ]
        ) + jnp.array([max_room_size, max_room_size])
        rng, _rng = jax.random.split(rng)
        room_position += jax.random.randint(
            _rng, (2,), minval=0, maxval=chunk_size - min_room_size
        )

        slice = jax.lax.dynamic_slice(
            block_map, room_position, (max_room_size, max_room_size)
        )
        xs = jnp.expand_dims(jnp.arange(max_room_size), axis=-1).repeat(
            max_room_size, axis=-1
        )
        ys = jnp.expand_dims(jnp.arange(max_room_size), axis=0).repeat(
            max_room_size, axis=0
        )

        room_mask = jnp.logical_and(
            xs < room_sizes[room_index, 0], ys < room_sizes[room_index, 1]
        )

        slice = room_mask * BlockType.PATH.value + (1 - room_mask) * slice

        block_map = jax.lax.dynamic_update_slice(
            block_map,
            slice,
            room_position,
        )

        # Torches in corner
        item_map = item_map.at[room_position[0], room_position[1]].set(
            ItemType.TORCH.value
        )
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1, room_position[1]
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0], room_position[1] + room_sizes[room_index, 1] - 1
        ].set(ItemType.TORCH.value)
        item_map = item_map.at[
            room_position[0] + room_sizes[room_index, 0] - 1,
            room_position[1] + room_sizes[room_index, 1] - 1,
        ].set(ItemType.TORCH.value)

        # Chest
        rng, _rng = jax.random.split(rng)
        chest_position = jax.random.randint(
            _rng,
            shape=(2,),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        block_map = block_map.at[
            room_position[0] + chest_position[0], room_position[1] + chest_position[1]
        ].set(BlockType.CHEST.value)

        # Fountain
        rng, _rng, __rng = jax.random.split(rng, 3)
        fountain_position = jax.random.randint(
            _rng,
            shape=(2,),
            minval=jnp.ones(2),
            maxval=room_sizes[room_index] - jnp.ones(2),
        )
        room_has_fountain = jax.random.uniform(__rng) > 0.5
        fountain_block = (
            room_has_fountain * config.fountain_block
            + (1 - room_has_fountain)
            * block_map[
                room_position[0] + fountain_position[0],
                room_position[1] + fountain_position[1],
            ]
        )
        block_map = block_map.at[
            room_position[0] + fountain_position[0],
            room_position[1] + fountain_position[1],
        ].set(fountain_block)

        return (block_map, item_map, room_occupancy_chunks, rng), room_position

    rng, _rng = jax.random.split(rng)
    (padded_map, padded_item_map, _, _), room_positions = jax.lax.scan(
        _add_room,
        (padded_map, padded_item_map, room_occupancy_chunks, _rng),
        jnp.arange(num_rooms),
    )

    def _add_path(carry, path_index):
        cmap, included_rooms_mask, rng = carry

        path_source = room_positions[path_index]

        rng, _rng = jax.random.split(rng)
        sink_index = jax.random.choice(
            _rng, jnp.arange(num_rooms), p=included_rooms_mask
        )
        path_sink = room_positions[sink_index]

        # Horizontal component
        entire_row = cmap[path_source[0]]
        path_indexes = jnp.arange(static_params.map_size[0] + 2 * max_room_size)
        path_indexes = path_indexes - path_source[1]
        horizontal_distance = path_sink[1] - path_source[1]
        path_indexes = path_indexes * jnp.sign(horizontal_distance)

        horizontal_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask, jnp.sign(horizontal_distance)
        )
        horizontal_mask = jnp.logical_and(
            horizontal_mask, entire_row == BlockType.WALL.value
        )

        new_row = (
            horizontal_mask * BlockType.PATH.value + (1 - horizontal_mask) * entire_row
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_row, axis=0),
            path_source,
        )

        # Vertical component
        entire_col = cmap[:, path_sink[1]]
        path_indexes = jnp.arange(static_params.map_size[1] + 2 * max_room_size)
        path_indexes = path_indexes - path_source[0]
        vertical_distance = path_sink[0] - path_source[0]
        path_indexes = path_indexes * jnp.sign(vertical_distance)

        vertical_mask = jnp.logical_and(
            path_indexes >= 0, path_indexes <= jnp.abs(vertical_distance)
        )
        vertical_mask = jnp.logical_and(vertical_mask, jnp.sign(vertical_distance))

        vertical_mask = jnp.logical_and(
            vertical_mask, entire_col == BlockType.WALL.value
        )

        new_col = (
            vertical_mask * BlockType.PATH.value + (1 - vertical_mask) * entire_col
        )

        cmap = jax.lax.dynamic_update_slice(
            cmap,
            jnp.expand_dims(new_col, axis=-1),
            path_sink,
        )

        rng, _rng = jax.random.split(rng)
        included_rooms_mask = included_rooms_mask.at[path_index].set(True)
        return (cmap, included_rooms_mask, _rng), None

    rng, _rng = jax.random.split(rng)
    included_rooms_mask = jnp.zeros(num_rooms, dtype=bool).at[-1].set(True)
    (padded_map, _, _), _, = jax.lax.scan(
        _add_path, (padded_map, included_rooms_mask, _rng), jnp.arange(0, num_rooms)
    )

    # Place special block in a random room
    special_block_position = room_positions[0] + jnp.array([2, 2])
    padded_map = padded_map.at[
        special_block_position[0], special_block_position[1]
    ].set(config.special_block)

    map = padded_map[max_room_size:-max_room_size, max_room_size:-max_room_size]
    item_map = padded_item_map[
        max_room_size:-max_room_size, max_room_size:-max_room_size
    ]

    # Visual stuff
    c_path_map = map != BlockType.WALL.value
    z = jnp.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    adj_path_map = jsp.signal.convolve(c_path_map, z, mode="same")
    adj_path_map = adj_path_map > 0.5

    rng, _rng = jax.random.split(rng)
    rare_map = jax.random.choice(
        _rng,
        jnp.array([False, True]),
        static_params.map_size,
        p=jnp.array([0.9, 0.1]),
    )

    wall_map = (
        rare_map * BlockType.WALL_MOSS.value + (1 - rare_map) * BlockType.WALL.value
    )

    rare_map = jnp.logical_and(rare_map, map == BlockType.PATH.value)
    rare_map = jnp.logical_and(rare_map, item_map == ItemType.NONE.value)
    path_map = rare_map * config.rare_path_replacement_block + (1 - rare_map) * map

    is_wall_map = jnp.logical_and(map == BlockType.WALL.value, adj_path_map)
    is_darkness_map = jnp.logical_not(adj_path_map)
    is_path_map = jnp.logical_not(jnp.logical_or(is_wall_map, is_darkness_map))

    map = (
        is_path_map * path_map
        + is_wall_map * wall_map
        + is_darkness_map * BlockType.DARKNESS.value
    )

    light_map = jnp.ones(static_params.map_size, dtype=jnp.float32)

    # Ladders
    valid_ladder_down = (map.flatten() == BlockType.PATH.value).astype(jnp.float32)
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_down / valid_ladder_down.sum(),
    )
    ladder_down_position = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    item_map = item_map.at[ladder_down_position[0], ladder_down_position[1]].set(
        ItemType.LADDER_DOWN.value
    )

    valid_ladder_up = map.flatten() == BlockType.PATH.value
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_up,
    )
    ladder_up_position = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )
    item_map = item_map.at[ladder_up_position[0], ladder_up_position[1]].set(
        ItemType.LADDER_UP.value
    )

    return map, item_map, light_map, ladder_down_position, ladder_up_position


def generate_smoothworld(rng, static_params, player_position, config, params=None):
    if params is not None:
        fractal_noise_angles = params.fractal_noise_angles
    else:
        fractal_noise_angles = (None, None, None, None, None)

    player_proximity_map = get_distance_map(
        player_position, static_params.map_size
    ).astype(jnp.float32)
    player_proximity_map_water = (
        player_proximity_map / config.player_proximity_map_water_strength
    )
    player_proximity_map_water = jnp.clip(
        player_proximity_map_water, 0.0, config.player_proximity_map_water_max
    )

    player_proximity_map_mountain = (
        player_proximity_map / config.player_proximity_map_mountain_strength
    )
    player_proximity_map_mountain = jnp.clip(
        player_proximity_map_mountain,
        0.0,
        config.player_proximity_map_mountain_max,
    )

    larger_res = (static_params.map_size[0] // 4, static_params.map_size[1] // 4)
    small_res = (static_params.map_size[0] // 16, static_params.map_size[1] // 16)
    x_res = (static_params.map_size[0] // 8, static_params.map_size[1] // 2)

    rng, _rng = jax.random.split(rng)
    water = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        small_res,
        octaves=1,
        override_angles=fractal_noise_angles[0],
    )
    water = water + player_proximity_map_water - 1.0

    # Water
    rng, _rng = jax.random.split(rng)
    map = jnp.where(
        water > config.water_threshold, config.sea_block, config.default_block
    )

    sand_map = jnp.logical_and(
        water > config.sand_threshold,
        map != config.sea_block,
    )

    map = jnp.where(sand_map, config.coast_block, map)

    # Mountain vs grass
    mountain_threshold = 0.7

    rng, _rng = jax.random.split(rng)
    mountain = (
        generate_fractal_noise_2d(
            _rng,
            static_params.map_size,
            small_res,
            octaves=1,
            override_angles=fractal_noise_angles[1],
        )
        + 0.05
    )
    mountain = mountain + player_proximity_map_mountain - 1.0
    map = jnp.where(mountain > mountain_threshold, config.mountain_block, map)

    # Paths
    rng, _rng = jax.random.split(rng)
    path_x = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        x_res,
        octaves=1,
        override_angles=fractal_noise_angles[2],
    )
    path = jnp.logical_and(mountain > mountain_threshold, path_x > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    path_y = path_x.T
    path = jnp.logical_and(mountain > mountain_threshold, path_y > 0.8)
    map = jnp.where(path > 0.5, config.path_block, map)

    # Caves
    rng, _rng = jax.random.split(rng)
    caves = jnp.logical_and(mountain > 0.85, water > 0.4)
    map = jnp.where(caves > 0.5, config.inner_mountain_block, map)

    # Trees
    rng, _rng = jax.random.split(rng)
    tree_noise = generate_fractal_noise_2d(
        _rng,
        static_params.map_size,
        larger_res,
        octaves=1,
        override_angles=fractal_noise_angles[3],
    )
    tree = (tree_noise > config.tree_threshold_perlin) * jax.random.uniform(
        rng, shape=static_params.map_size
    ) > config.tree_threshold_uniform
    tree = jnp.logical_and(tree, map == config.tree_requirement_block)
    map = jnp.where(tree, config.tree, map)

    # Ores
    def _add_ore(carry, index):
        rng, map = carry
        rng, _rng = jax.random.split(rng)
        ore_map = jnp.logical_and(
            map == config.ore_requirement_blocks[index],
            jax.random.uniform(_rng, static_params.map_size)
            < config.ore_chances[index],
        )
        map = jnp.where(ore_map, config.ores[index], map)

        return (rng, map), None

    rng, _rng = jax.random.split(rng)
    (_, map), _ = jax.lax.scan(_add_ore, (_rng, map), jnp.arange(5))

    # Lava
    lava_map = jnp.logical_and(
        mountain > 0.85,
        tree_noise > 0.7,
    )
    map = jnp.where(lava_map, config.lava, map)

    # Light map
    light_map = (
        jnp.ones(static_params.map_size, dtype=jnp.float32) * config.default_light
    )

    # Make sure player spawns on grass
    map = map.at[player_position[0], player_position[1]].set(config.player_spawn)

    item_map = jnp.zeros(static_params.map_size, dtype=jnp.int32)

    valid_ladder_down = map.flatten() == config.valid_ladder
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_down,
    )
    ladder_down = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    item_map = item_map.at[ladder_down[0], ladder_down[1]].set(
        ItemType.LADDER_DOWN.value * config.ladder_down
        + item_map[ladder_down[0], ladder_down[1]] * (1 - config.ladder_down)
    )

    valid_ladder_up = map.flatten() == config.valid_ladder
    rng, _rng = jax.random.split(rng)
    ladder_index = jax.random.choice(
        _rng,
        jnp.arange(static_params.map_size[0] * static_params.map_size[1]),
        p=valid_ladder_up,
    )
    ladder_up = jnp.array(
        [
            ladder_index // static_params.map_size[0],
            ladder_index % static_params.map_size[0],
        ]
    )

    LIGHT_MAP_AROUND_LADDER = TORCH_LIGHT_MAP * (
        1 - config.default_light
    ) + config.default_light * jnp.ones((9, 9))

    light_map = jax.lax.dynamic_update_slice(
        light_map, LIGHT_MAP_AROUND_LADDER, ladder_up - jnp.array([4, 4])
    )

    z = jnp.array([[0.2, 0.7, 0.2], [0.7, 1, 0.7], [0.2, 0.7, 0.2]]) * (
        config.lava == BlockType.LAVA.value
    )
    light_map += jsp.signal.convolve(lava_map, z, mode="same")
    light_map = jnp.clip(light_map, 0.0, 1.0)

    item_map = item_map.at[ladder_up[0], ladder_up[1]].set(
        ItemType.LADDER_UP.value * config.ladder_up
        + item_map[ladder_up[0], ladder_up[1]] * (1 - config.ladder_up)
    )

    return map, item_map, light_map, ladder_down, ladder_up


# Map seeds are drawn from [0, MAP_SEED_BOUND): non-negative int32, so a seed
# round-trips exactly through the scalar logs and the command line.
MAP_SEED_BOUND = 2**31 - 1


def strip_world_level_zero(rng, state, static_params):
    """Replace level 0 with the minimal evaluation world: nothing but grass,
    one water tile, one passive mob (cow) and one predator. The three are
    placed uniformly at random over distinct cells, none of them the player's,
    so no mob starts in the water or on top of the other. Levels 1-8 are left
    as generated, being legacy and unused.

    Nothing respawns here: passive and monster spawning both require a PATH
    tile (see spawn_mobs) and this world has none, so the cow is gone once
    eaten, and the predator is the only one there will ever be. It also never
    despawns (update_mobs), unlike a normally spawned predator, which vanishes
    once the player walks mob_despawn_distance away.

    Meant for studying a trained agent's dynamics in isolation, not training.
    """
    map_size = static_params.map_size
    n_cells = map_size[0] * map_size[1]
    cell_weights = (
        jnp.ones(n_cells)
        .at[state.player_position[0] * map_size[1] + state.player_position[1]]
        .set(0.0)
    )
    cells = jax.random.choice(
        rng, n_cells, shape=(3,), replace=False, p=cell_weights / cell_weights.sum()
    )
    water_position, passive_position, melee_position = jnp.stack(
        [cells // map_size[1], cells % map_size[1]], axis=-1
    ).astype(jnp.int32)

    level_map = jnp.full(map_size, BlockType.GRASS.value, dtype=state.map.dtype)
    level_map = level_map.at[water_position[0], water_position[1]].set(
        BlockType.WATER.value
    )

    # Mob types and health match what spawn_mobs would give level 0's mobs.
    passive_type = FLOOR_MOB_MAPPING[0, MobType.PASSIVE.value]
    melee_type = FLOOR_MOB_MAPPING[0, MobType.MELEE.value]

    return state.replace(
        map=state.map.at[0].set(level_map),
        item_map=state.item_map.at[0].set(
            jnp.full(map_size, ItemType.NONE.value, dtype=state.item_map.dtype)
        ),
        light_map=state.light_map.at[0].set(
            jnp.ones(map_size, dtype=state.light_map.dtype)
        ),
        mob_map=state.mob_map.at[0].set(
            jnp.zeros(map_size, dtype=bool)
            .at[passive_position[0], passive_position[1]]
            .set(True)
            .at[melee_position[0], melee_position[1]]
            .set(True)
        ),
        passive_mobs=state.passive_mobs.replace(
            position=state.passive_mobs.position.at[0, 0].set(passive_position),
            health=state.passive_mobs.health.at[0, 0].set(
                MOB_TYPE_HEALTH_MAPPING[passive_type, MobType.PASSIVE.value]
            ),
            mask=state.passive_mobs.mask.at[0, 0].set(True),
            type_id=state.passive_mobs.type_id.at[0, 0].set(passive_type),
        ),
        melee_mobs=state.melee_mobs.replace(
            position=state.melee_mobs.position.at[0, 0].set(melee_position),
            health=state.melee_mobs.health.at[0, 0].set(
                MOB_TYPE_HEALTH_MAPPING[melee_type, MobType.MELEE.value]
            ),
            mask=state.melee_mobs.mask.at[0, 0].set(True),
            type_id=state.melee_mobs.type_id.at[0, 0].set(melee_type),
        ),
    )


def generate_world(rng, params, static_params):
    # The world is generated from an integer map seed drawn from rng rather
    # than from rng directly, and the seed is recorded in state.map_seed. The
    # same seed and static params always rebuild the same world (terrain on
    # every level, player start position, potion mapping), so
    # static_params.replay_map_seed can replace the draw to reproduce a logged
    # episode's map. env_id comes from rng instead of the seed so that replayed
    # episodes still get distinct episode ids.
    seed_rng, env_id_rng = jax.random.split(rng)
    if static_params.replay_map_seed is None:
        map_seed = jax.random.randint(
            seed_rng, (), 0, MAP_SEED_BOUND, dtype=jnp.int32
        )
    else:
        map_seed = jnp.asarray(static_params.replay_map_seed, dtype=jnp.int32)
    rng = jax.random.PRNGKey(map_seed)

    player_position = jnp.array(
        [static_params.map_size[0] // 2, static_params.map_size[1] // 2]
    )
    rng, _rng = jax.random.split(rng)
    # Random start position option, any position not on the edge of the arena is valid
    #HACK: Limit to middle 50%
    if static_params.random_start:
        player_position = jax.random.randint(_rng, (2,), 40, static_params.map_size[0] - 40)

    # Toggle for featureless arena
    world_configs = ALL_SMOOTHGEN_CONFIGS
    if static_params.featureless_world:
        world_configs = FEATURELESS_SMOOTHGEN_CONFIGS

    # Generate smoothgens (overworld, caves, elemental levels, boss level)
    rngs = jax.random.split(rng, 7)
    rng, _rng = rngs[0], rngs[1:]


    overworld = generate_smoothworld(
        _rng[0],
        static_params,
        player_position,
        jax.tree.map(lambda x: x[0], world_configs),
        params=params,
    )

    smoothgens = jax.vmap(generate_smoothworld, in_axes=(0, None, None, 0))(
        _rng[1:],
        static_params,
        player_position,
        jax.tree.map(lambda x: x[1:], world_configs),
    )

    smoothgens = jax.tree.map(
        lambda over, others: jnp.concatenate([jnp.array([over]), others], axis=0),
        overworld,
        smoothgens,
    )

    # Generate dungeons
    rngs = jax.random.split(rng, 4)
    rng, _rng = rngs[0], rngs[1:]
    dungeons = jax.vmap(generate_dungeon, in_axes=(0, None, 0))(
        _rng, static_params, ALL_DUNGEON_CONFIGS
    )

    # Splice smoothgens and dungeons in order of levels
    map, item_map, light_map, ladders_down, ladders_up = jax.tree.map(
        lambda x, y: jnp.stack(
            (x[0], y[0], x[1], y[1], y[2], x[2], x[3], x[4], x[5]), axis=0
        ),
        smoothgens,
        dungeons,
    )

    # Mobs
    def generate_empty_mobs(max_mobs):
        return Mobs(
            position=jnp.zeros(
                (static_params.num_levels, max_mobs, 2), dtype=jnp.int32
            ),
            health=jnp.ones((static_params.num_levels, max_mobs), dtype=jnp.float32),
            mask=jnp.zeros((static_params.num_levels, max_mobs), dtype=bool),
            attack_cooldown=jnp.zeros(
                (static_params.num_levels, max_mobs), dtype=jnp.int32
            ),
            type_id=jnp.zeros((static_params.num_levels, max_mobs), dtype=jnp.int32),
        )

    melee_mobs = generate_empty_mobs(2)  # make space for changing curriculum
    ranged_mobs = generate_empty_mobs(3)
    passive_mobs = generate_empty_mobs(static_params.max_passive_mobs)

    # Projectiles
    def _create_projectiles(max_num):
        projectiles = generate_empty_mobs(max_num)

        projectile_directions = jnp.ones(
            (static_params.num_levels, max_num, 2), dtype=jnp.int32
        )

        return projectiles, projectile_directions

    mob_projectiles, mob_projectile_directions = _create_projectiles(
        static_params.max_mob_projectiles
    )
    player_projectiles, player_projectile_directions = _create_projectiles(
        static_params.max_player_projectiles
    )

    # Plants
    growing_plants_positions = jnp.zeros(
        (static_params.max_growing_plants, 2), dtype=jnp.int32
    )
    growing_plants_age = jnp.zeros(static_params.max_growing_plants, dtype=jnp.int32)
    growing_plants_mask = jnp.zeros(static_params.max_growing_plants, dtype=bool)

    # Potion mapping for episode
    rng, _rng = jax.random.split(rng)
    potion_mapping = jax.random.permutation(_rng, jnp.arange(6))

    # Inventory
    inventory = jax.tree.map(
        lambda x, y: jax.lax.select(params.god_mode, x, y),
        get_new_full_inventory(),
        get_new_empty_inventory(),
    )

    rng, _rng = jax.random.split(rng)

    state = EnvState(
        env_id=jax.random.randint(env_id_rng, shape=(1,), minval=0, maxval=999999),
        map_seed=map_seed,
        map=map,
        item_map=item_map,
        mob_map=jnp.zeros(
            (static_params.num_levels, *static_params.map_size), dtype=bool
        ),
        light_map=light_map,
        down_ladders=ladders_down,
        up_ladders=ladders_up,
        chests_opened=jnp.zeros(static_params.num_levels, dtype=bool),
        monsters_killed=jnp.zeros(static_params.num_levels, dtype=jnp.int32)
        .at[0]
        .set(10),  # First ladder starts open
        player_position=player_position,
        player_starting_position=player_position,
        player_direction=jnp.asarray(Action.UP.value, dtype=jnp.int32),
        player_level=jnp.asarray(0, dtype=jnp.int32),
        player_health=jnp.asarray(9.0, dtype=jnp.float32),
        player_food=jnp.asarray(9, dtype=jnp.int32),
        player_drink=jnp.asarray(9, dtype=jnp.int32),
        player_energy=jnp.asarray(9, dtype=jnp.int32),
        player_mana=jnp.asarray(9, dtype=jnp.int32),
        player_recover=jnp.asarray(0.0, dtype=jnp.float32),
        player_hunger=jnp.asarray(0.0, dtype=jnp.float32),
        player_thirst=jnp.asarray(0.0, dtype=jnp.float32),
        player_fatigue=jnp.asarray(0.0, dtype=jnp.float32),
        player_recover_mana=jnp.asarray(0.0, dtype=jnp.float32),
        is_sleeping=False,
        is_resting=False,
        player_xp=jnp.asarray(0, dtype=jnp.int32),
        player_dexterity=jnp.asarray(1, dtype=jnp.int32),
        player_strength=jnp.asarray(1, dtype=jnp.int32),
        player_intelligence=jnp.asarray(1, dtype=jnp.int32),
        inventory=inventory,
        sword_enchantment=jnp.asarray(0, dtype=jnp.int32),
        bow_enchantment=jnp.asarray(0, dtype=jnp.int32),
        armour_enchantments=jnp.array([0, 0, 0, 0], dtype=jnp.int32),
        melee_mobs=melee_mobs,
        ranged_mobs=ranged_mobs,
        passive_mobs=passive_mobs,
        mob_projectiles=mob_projectiles,
        mob_projectile_directions=mob_projectile_directions,
        player_projectiles=player_projectiles,
        player_projectile_directions=player_projectile_directions,
        growing_plants_positions=growing_plants_positions,
        growing_plants_age=growing_plants_age,
        growing_plants_mask=growing_plants_mask,
        potion_mapping=potion_mapping,
        learned_spells=jnp.array([False, False], dtype=bool),
        boss_progress=jnp.asarray(0, dtype=jnp.int32),
        boss_timesteps_to_spawn_this_round=jnp.asarray(
            BOSS_FIGHT_SPAWN_TURNS, dtype=jnp.int32
        ),
        achievements=jnp.zeros((len(Achievement),), dtype=bool),
        light_level=jnp.asarray(calculate_light_level(0, params), dtype=jnp.float32),
        state_rng=_rng,
        timestep=jnp.asarray(0, dtype=jnp.int32),
    )

    if static_params.stripped_world:
        # Level 0 is generated and then thrown away here. Wasteful, but this
        # world is only ever used for evaluation, and going through the normal
        # generation keeps every other part of the state consistent.
        state = strip_world_level_zero(rng, state, static_params)

    return state
