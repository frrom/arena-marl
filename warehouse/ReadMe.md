**Launchfile** 

```
roslaunch warehouse map_launch.launch
```

1. Change the map attributes like shelf_cols, shelf_rows, col_height, bigger_highways, random_map in the launch file
2. or pass them from the command line.

**Map Creator**

The visualizer can also be run seperately with

```
python gridworld.py shelf_cols:=3 shelf_rows:=3 col_height:=3 scale:=100 bigger_highways:=True rand_map:=False additional_goals:=None
```
Arguments:
- shels_cols: Number of shelf columns (int)
- shelf_rows: Number of shelf rows (int)
- col_height: Columns inside each shelf (int)
- scale: Scale for upsizing the map (just keep it 100) (int)
- bigger_highways: Increase the width of the highways by 1 (bool)
- rand_map: randomize map parameters (bool)
- additional_goals: Add more goals on the top (use N) or on the left, right (L,R)


**Visualizer**

The visualizer can also be run seperately with
```
rosrun warehouse rviz_visualization.py
```

**Some Explantions**
1. The roslaunch command runs the map_(creation) node first and stores the created map in gridworld.
After the map is created we call the rest.launch from the map_(creation) node.

2. gridworld.py (also the map_node) publishes the numpy array of the map to the /gridworld_base topic as an OccupancyGrid.

3. (new) The Vis_node (rviz_visualization.py) loads the map and yaml from the gridworld folder and subscribes to sim_?/open_tasks to get the occupied shelfs and goals.

4. (old)The Vis_node (rviz_visualization.py) gets the map from the /gridworld topic and creates the markers for the map.


5. The temporary task manager (task_gen1.py, only for testing stuff on my end) then creates random tasks and publishes the updated map to /gridworld

6. All gridworld topics are OccupancyGrids



