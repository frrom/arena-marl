**Usage** 

```
roslaunch warehouse map_launch.launch
```

Change the map attributes like shelf_cols, shelf_rows, col_height, bigger_highways, random_map in the launch file
or pass them from the command line.


**Visualizer**
The visualizer can also be run seperately with
```
rosrun warehouse rviz_visualization.py
```

**Some Explantions**
The roslaunch command runs the map_(creation) node first and stores the created map in gridworld.
After the map is created we call the rest.launch from the map_(creation) node.

gridworld.py (also the map_node) publishes the numpy array of the map to the /gridworld_base topic as an OccupancyGrid.

(new) The Vis_node (rviz_visualization.py) loads the map and yaml from the gridworld folder and subscribes to sim_?/open_tasks to get the occupied shelfs and goals.

(old)The Vis_node (rviz_visualization.py) gets the map from the /gridworld topic and creates the markers for the map.


The temporary task manager (task_gen1.py, only for testing stuff on my end) then creates random tasks and publishes the updated map to /gridworld

All gridworld topics are OccupancyGrids



