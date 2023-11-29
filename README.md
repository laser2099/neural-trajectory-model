# neural trajectory model
#### A neural trajectory model using PyTorch 

## How To Run?

### Training

Run neural trajectory model training. Make sure you have a proper mesh file. The coordinate range of mesh should not be too large. You can use Blender to reduce the range of the mesh file. The command to train on Blender scenes is:

```
python run.py --mesh-path {your path to store the mesh} --data {your training data path} --eval-data {your eval data path} --save-to {the path you expect to store the model} --A-star-num-agents {expected number of agents}
```
If you only have a mesh file, the script would generate A star trajectories, optimize A star trajectories or convert existing csv data to proper format based on the circumstances.

#### Generate A star trajectories
The single-agent trajectories are generated in mesh environment based on classical A star algorithm. You can adjust the correponding parameters to get the proper A star-based trajectory generation.
```
python run.py --mesh-path {your path to store the mesh} --A-star-grid-size {the grid map size of the mesh map} --A-star-num-traj {total number of A star trajectories} --A-star-num-step {the density of A star trajectories} --A-star-num-agents {expected number of agents}
```
#### Optimize A star trajectories to corresponding multi-agent trajectories
The single-agent trajectories are then optimized by an optimizer to get the corresponding multi-agent trajectories. You can adjust the correponding parameters to get the proper optimized trajectory generation.
```
python run.py --mesh-path {your path to store the mesh} --optimizer-lr {the learning rate when optimizing multi-agent trajectories} --A-star-num-agents {expected number of agents} --safe-dis-sep {safe distance for all agents} --sdf-thresh {safe sdf thresh for potential environmental collision}
```
#### Other training options
There are more options to guide you to train a neural trajectory model. If you set ```train_compute_sdf_loss``` to False, sdf loss during training process will be calculated based on nvidia kaolin library. If you set it to True, a neural sdf network is required to be selected. You can also change the hidden dim of neural trajectory model by adjusting ```hidden-dim```.

### Evaluation

Once training has finished or you've achieved satisfactory results, the checkpoint will be in the ```{save-to}``` folder. The command to evaluate the result is:
```
python test_neural_traj.py --mesh-path {your path to store the mesh} --test-data {your test data path} --model-path {the checkpoit path} --num-agents { number of agents}
```

