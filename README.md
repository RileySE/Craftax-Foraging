
# ForageWorld
Forageworld is an simulated foraging arena RL task built on top of <a href="https://github.com/MichaelTMatthews/Craftax/">Craftax</a>. It limits the scope of the environment to the first "level" from Craftax and modifies this extensively to enable and encourage open-ended naturalistic foraging behavior. 

<p align="middle">
 
</p>

## Needed Python Packages (may work with other versions but we have not tested)
Note: `pip>=23.0` is required
```
black                    24.4.2
chex                     0.1.86
distrax                  0.1.5
flax                     0.8.5
gymnax                   0.0.8
imageio                  2.34.2
jax                      0.4.30
jax-cuda12-pjrt          0.4.30
jax-cuda12-plugin        0.4.30
jaxlib                   0.4.30
matplotlib               3.9.1
ml-collections           0.1.1
numpy                    2.0.1
optax                    0.2.3
orbax-checkpoint         0.5.23
pre-commit               3.8.0
pygame                   2.6.0
wandb                    0.17.5
```

## Jaxpruner
You will need to install Jaxpruner manually by going to `https://github.com/google-research/jaxpruner`, downloading the repo, and then doing:
`cd jaxpruner`
`pip install -e .`

## Setup
Setup is broadly similar to Craftax. Install the above packages using your package manager of choice, then, while in the top level `Craftax-Foraging` directory, run
`pip install -e .`

## GPU-Enabled JAX
By default, JAX will install on the CPU.  If you want to run JAX on a GPU, you'll need to install the correct wheel for your system from <a href="https://github.com/google/jax?tab=readme-ov-file#installation">JAX</a>.
For NVIDIA GPU the command is:
```
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that a GPU with at least 24 GB of memory is required to run the experiments in the paper.

# 📈 Experiments
The following command line runs our baseline configuration (which takes about 36 hours on an H100-equivalent GPU):
```
python forageworld/ppo_rnn.py --no_videos --output_path <path> --wandb_project foraging_baseline --logging_steps_per_viz 32 --aux_coef 0.025 --updates_per_viz 2048 --sparsity 0.9 --max_cows 108
```
To run other configurations, additional command line options may be added, such as:
```
--directional_vision
--map_size 48
--no_memory
--sparse_alg no_prune
```
and so on.
See the command line options in `forageworld/ppo_rnn.py` for more options, and the paper appendix for a description of each option.
Other than the environment features noted to be varied in a given experiment, all other command line options should remain the same.