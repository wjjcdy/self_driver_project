^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package waypoint_updater 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
2019-6-18
* Modified LOOKAHEAD_WPS from 100 to 40 to optimizate delay problem

2019-6-7
* Modified LOOKAHEAD_WPS from 200 to 100 to optimizate delay problem
* Modified stop stopline in front 10 point from 2 point to stop in front of stopline  

2019-6-5
-------------------
* Add traffic_light callback function to get traffic point index
* Modified the type of '/traffic_waypoint' as Int32
* Modified the publish_waypoints based on traffic_light, need stop the stopline when light is red

2019-6-3
-------------------
* Modified waypoint_tree KDTREE has been initialized, the get_closest_waypoint_idx function can work
* Modified a bug, change a variable name from pos to pos_current
* Add publish_waypoints function used publish_waypoints 200 points in front of the car pose

2019-5-30
-------------------
* Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
* Add global pose and base_waypoins
* Add main loop for get the closest points in base_waypoins
* Finished base_points and pos_current callback function









