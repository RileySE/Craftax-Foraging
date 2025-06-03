
# ForageWorld
Forageworld is an simulated foraging arena RL task built on top of <a href="https://github.com/MichaelTMatthews/Craftax/">Craftax</a>. It limits the scope of the environment to the first "level" from Craftax and modifies this extensively to enable and encourage open-ended naturalistic foraging behavior. 

<p align="middle">
 
</p>

## Jaxpruner
You will need to install Jaxpruner manually by going to `https://github.com/google-research/jaxpruner`, downloading the repo, and then doing:

```
cd jaxpruner
pip install -e .
```

## ForageWorld Setup
(Note: `pip>=23.0` is required)

Setup is broadly similar to Craftax. Install Jaxpruner as above, then while in the top level `Craftax-Foraging` directory, run:
```
pip install -e .
```
## JAX
Finally, a specific version of GPU JAX is required (version 4.3). For an Nvidia GPU, run the following:
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