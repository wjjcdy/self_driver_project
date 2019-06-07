#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import numpy as np
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 10       # Max acceleration is 10 m/s^2
MAX_JERK = 10       # Max jerk is 10 m/s^2

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint',Int32,self.traffic_cb)
        rospy.Subscriber('/obstacle_waypoint',Lane,self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.pose_current = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1
        self.loop()
    
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            if self.pose_current and self.waypoint_tree:
                self.publish_waypoints()
                rate.sleep()
    
    def get_closest_waypoint_idx(self):
        x = self.pose_current.pose.position.x
        y = self.pose_current.pose.position.y
        closest_idx = self.waypoint_tree.query([x,y],1)[1]

        # Check if closest is ahead or behind vehicles
        closest_coord = self.waypoints_2d[closest_idx]
        pre_coord = self.waypoints_2d[closest_idx-1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(pre_coord)
        pos_vect = np.array([x,y])

        val = np.dot(cl_vect-prev_vect,pos_vect-cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1)% len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self):
        # Generate the best lane based on basepoints and traffic light
        lane = self.generate_lane()
        self.final_waypoints_pub.publish(lane)

    def generate_lane(self):
        lane = Lane()
        closest_idx = self.get_closest_waypoint_idx()
        farest_idx = closest_idx + LOOKAHEAD_WPS
        
        # When nearest light is not red or is farer than LOOKAHEAD_WPS, just ignore
        if (self.stopline_wp_idx == -1) or (self.stopline_wp_idx >= farest_idx):
            base_waypoints = self.base_waypoints.waypoints[closest_idx:farest_idx]
            lane.waypoints = base_waypoints
        else:
            # Need get closest_idx to stopline pose as next waypoints, and need speed down
            base_waypoints = self.base_waypoints.waypoints[closest_idx:self.stopline_wp_idx]
            lane.waypoints = self.decelerate(base_waypoints,closest_idx)
        return lane

    def decelerate(self, waypoints, closest_idx):
        temp_waypoints = []
        # Get stop idx which the car must stop in front of the stopline, rospy.logerrthe lenth of car is about 2 points
        stop_idx = max(0, self.stopline_wp_idx - closest_idx - 10) 
        for i, wp in enumerate(waypoints):
            dist_i = self.distance(waypoints, i, stop_idx)
            # Get target speed which can stop in dist_is distance, can used dist*a = 0.5*a*t^2 *a
            vel = math.sqrt(2*MAX_DECEL*dist_i)

            if vel < 1.0:     #Need stop 
                vel = 0

            p = Waypoint()
            p.pose = wp.pose
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp_waypoints.append(p)
        return temp_waypoints


    def pose_cb(self, msg):
        # TODO: Implement
        self.pose_current = msg         # Get the pose of the car right now

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints # Get and store the base waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.stopline_wp_idx = msg.data      # Get the nearest traffic light stopline index of basewaypoints

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
