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

            x = nn.Dense(self.action_dim)(x)
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
            x = nn.Dense(self.action_dim)(x)
        return x


def run_pqn(config):

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

    args = parse_args()

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={key.upper(): value for key, value in vars(args).items()},
        name=args.run_name,
    )

    run_pqn(wandb.config)