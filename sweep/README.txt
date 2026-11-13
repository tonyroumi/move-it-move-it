Each child directory of this folder is one training run. Run the sweep from the move-it-move-it
repo root, so the manifests' clip paths resolve:

  ./isaaclab.sh -p scripts/sweep.py [sweep_root] [--force] [--dry_run] [--headless ...]

The sweep root defaults to sweep/. Pass --dry_run first to check every run and print its
train.py command without training.

A run directory holds the parameter files that run trains with:

  <run-name>/experiment.yaml          seed, experiment directory and name, logger, description
  <run-name>/agents/<algo>.yaml       the agent's hyperparameters (agents/ppo.yaml or agents/amp.yaml)
  <run-name>/manifests/<name>.yaml    the motion manifest

The directory name picks among those files and tells runs apart. It is key=value segments joined
by a double underscore, for example:

  algo=PPO__manifest=box_manifest__num_envs=4096__seed=1__experiment.experiment_name=boxing

  algo                        required. PPO or AMP. Trains with agents/<algo lowercased>.yaml.
                              That file's agent.class_type, if set, must match.
  manifest                    required. Trains with manifests/<manifest>.yaml.
  experiment.experiment_name  optional. Overrides experiment.yaml and is train.py's --name.
  seed                        optional. Overrides experiment.yaml.
  num_envs                    optional. train.py --num_envs; otherwise the env default.

Percent-encode a value containing a double underscore, "=" or "/" (%5F%5F, %3D, %2F). A value
ending in "_" needs that "_" encoded as %5F.

The name's overrides apply to that run only, through a temporary copy of its experiment.yaml.
The files in the run directory and in configs/ are never rewritten. Change a run by editing its
files, or by changing a segment of its name.

Logs, dumped params and checkpoints are written inside the run directory, under
<experiment.directory>/<experiment_name>/<algo>/<timestamp>/. A run that already has its final
checkpoint (checkpoints/<runner.timesteps>.pt) is skipped unless --force is passed. A run with a
bad name or a missing file is reported, and the sweep goes on to the next run.
