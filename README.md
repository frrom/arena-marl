# ARENA MARL
Arena marl is a package from the ***Arena ROSNav*** organisation. Here we develop all modular functionalities for a **M**ulti **A**gent **R**einforcement **L**earning setting. This includes homogeneous and heterogeneous setups! This fork extends arena-marl to incorporate a communication stack, as well as the framework for a dynamic warehouse scenario with goal selection among robots.

## Installation and setup

First you have to create a standard catkin workspace

```bash
mkdir MARL_ws && mkdir MARL_ws/src
```

Now clone this repository into the src directory
```bash
cd MARL_ws/src
git clone git@github.com:frrom/arena-marl.git
```

Install and activate the poetry environment
```bash
cd arena-marl
poetry install && poetry shell
```

Add all additional necessary packages via the .rosinstall
```bash
rosws update
```

Build the whole catkin workspace
```bash
cd ../..
catkin_make -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_CXX_STANDARD=14
```

Source the correct setup path.
> When using bash use .bash/ .bashrc and when using zsh use .zsh/ .zshrc
```bash
source $HOME/MARL_ws/devel/setup.zsh
```

take the files from `src/arena-marl/planners` and `src/arena-marl/forks` and merge into `src/planners and src/forks` respectively, to enable new feature extractors and models.

You are good to go!

## Example Usage

### Deployment

1. In the first terminal launch all necessary nodes.
    > num_robots is the number off all agents combined 
    ```bash
    roslaunch arena_bringup start_evaluation.launch map_folder_name:=gridworld num_robots:=4     
    ```
    you can change the map according to your desired scenario. Available maps can be found in `arena-marl/forks/arena-simulation-setup/maps` .

2. Define deployment setup in [deployment_config.yaml](training/configs/deployment_config.yaml).
    > You can choose your desired architecture and number of robots as you like, as well as define the number of evaluation episodes. 
    > Descriptions of architectures are commented in that file. It is important to choose the correct architecture name for the corresponding deployment file, defined in "resume". 
    > The loaded file in that directory must have the name "best_model.zip". When looking into the directory, the trained architectures are named according to their training setup, as described in the thesis. Rename them for deployment. Also make sure, the total number of robots is equal to the num_robots parameter in your launch execution.

3. Setup your deployment.
    > open `src/arena-marl/testing/scripts/marl_deployment.py`. Set parameters, for deployed architecture and setup. If you use the Sliding Window architecture for example, you must set the "sliding window" parameter to True. Set the desired deployment stage with the "/curr_stage" parameter.

3. In the second terminal start the deployment script.
    ```bash
    python marl_deployment.py --config deployment_config.yaml     
    ```
### Map Creation
if you want to create your own costum map follow these steps:
1. start the following script in the terminal.
    ```bash
    roslaunch warehouse map_launch.launch map_folder_name:=<map_name> shelf_cols:=2 shelf_rows:=3 scale:=100   
    ```
2. If the map name contains the words "crossroad" it will generate a crossroad with 2 vertial and 3 horizontal highways. If the name contains the word "empty", it will generate an empty map. Otherwise a Warehose will be generated. The `scale` variable defines the size of the created map.

## Trouble shooting
If you have some issues with packages not being found try exporting the following paths:
```bash
export ROS_PACKAGE_PATH=$HOME/MARL_ws/src:/opt/ros/noetic/share
export PYTHONPATH=$HOME/MARL_ws/src/arena-marl:$HOME/MARL_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}
```

If you have issues with the obervation space when starting the simulation, try changing the value of "high" in line 278 of `src/arena-marl/rl_utils/rl_utils/base_agent_wrapper.py` from 5 to 6 or vice versa.

If the error persists, look into `src/forks/arena-simulation-setup/robot/jackal/jackal.model.yaml` and change the number of lidar beams according to the used architecture, the lidar composition may change between:

```bash
angle: {min: -3.12413936, max: 3.14159265359, increment: 0.01745}
```
and 
```bash
angle: {min: -3.12413936, max: 3.14159265359, increment: 0.0349}
```
