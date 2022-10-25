uint8 action
uint8 PICKUP   = 0 # used when picking up crate
uint8 TRANSIT  = 1 # used when transiting with crate
uint8 DROPOFF  = 2 # used when dropping off crate
uint8 ASSIGN   = 3 # used by task_manager when assigning goal to robot
uint8 REQUEST  = 4 # used by robot to request a new task from task_manager
uint8 BLOCK    = 5 # used by robot to reserve crate

string text # debugging