**Usage** 
(No copy&pasting)

'roslaunch warehouse map_launch.launch'


Change the map attributes like shelf_cols, shelf_rows, col_height, bigger_highways, random_map in the launch file
or pass them from the command line.



TODO: pass the arguments that are need in rest.launch also to map.launch and call rest.launch with them

**Some Explantions*'
The roslaunch command runs the map_(creation) node first and stores the created map in gridworld.
After the map is created we call the rest.launch from the map_(creation) node.

gridworld.py (also the map_node) publishes the numpy array of the map to the /gridworld_base topic as an OccupancyGrid.

The temporary task manager (task_gen1.py, only for testing stuff on my end) then creates random tasks and publishes the updated map to /gridworld

The Vis_node (rviz_visualization.py) gets the map from the /gridworld topic and creates the markers for the map.

All gridworld topics are OccupancyGrids



