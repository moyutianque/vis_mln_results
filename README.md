# Visualize mln predictions

> [Optional] For correctness label, should first process the results in simulator

## Data preparation

+ ```all_traj_pred.json``` indicates scores of all trajactory

+ ```best_routes.json``` indicates best selected path

+ [optional] ```sim_results.json``` simulation results (success or failed) for each episode

## Dump results

```bash
python draw_predictions.py --input_dir [path to dir] --dump_dir [path to dir] --split [val_seen|val_unseen]
```

example scripts are located in  ```scripts/run.sh```