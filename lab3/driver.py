#!/usr/bin/env python3
# Bill Smart, smartw@oregonstate.edu
#
# driver.py
# Drive the robot towards a goal, going around an object
# Every Python node in ROS2 should include these lines.  rclpy is the basic Python
# ROS2 stuff, and Node is the class we're going to use to set up the node.
import rclpy
from rclpy.node import Node
# Velocity commands are given with Twist messages, from geometry_msgs
from geometry_msgs.msg import Twist, PoseStamped
# math stuff
from math import atan2, tanh, sqrt, pi, fabs, cos, sin
import numpy as np
# Header for the twist message
from std_msgs.msg import Header
# The twist command and the goal
from geometry_msgs.msg import TwistStamped, PointStamped
# For publishing markers to rviz
from visualization_msgs.msg import Marker
# The laser scan message type
from sensor_msgs.msg import LaserScan
# These are all for setting up the action server/client
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
# This is the format of the message sent by the client - it is another node under lab 2
from nav_targets.action import NavTarget
# These are for transforming points/targets in the world into a point in the robot's coordinate space
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_geometry_msgs import do_transform_point
# This sets up multi-threading so the laser scan can happen at the same time we're processing the target goal
from rclpy.executors import MultiThreadedExecutor
# Thread safety
from threading import Lock
class Lab3Driver(Node):
	def __init__(self, threshold=0.2):
		""" We have parameters this time
		@param threshold - how close do you have to be before saying you're at the goal? Set to width of robot
		"""
		# Initialize the parent class, giving it a name.  The idiom is to use the
		# super() class.
		super().__init__('driver')
		# Goal will be set later. The action server will set the goal; you don't set it directly
		self.goal = None
		# Lock because MultiThreadedExecutor can run scan + action at same time
		self.goal_lock = Lock()
		# A controllable parameter for how close you have to be to the goal to say "I'm there"
		self.threshold = threshold
		# Make a Marker to put in RViz to show the current goal/target the robot is aiming for
		self.target_marker = None
		# Publisher before subscriber
		self.cmd_pub = self.create_publisher(TwistStamped, 'cmd_vel', 1)
		# Publish the current target as a marker (so RViz can show it)
		self.target_pub = self.create_publisher(Marker, 'current_target', 1)
		# Subscriber after publisher; this is the laser scan
		self.sub = self.create_subscription(LaserScan, 'base_scan', self.scan_callback, 10)
		# Create a buffer to put the transform data in
		self.tf_buffer = Buffer()
        
		# This sets up a listener for all of the transform types created
		self.transform_listener = TransformListener(self.tf_buffer, self)
		# Action client for passing "target" messages/state around
		# An action has a goal, feedback, and a result. This class (the driver) will have the action server side, and be
		#   responsible for sending feed back and result
		# The SendPoints class will have the action client - it will send the goals and cancel the goal and send another when 
		#    the server says it has completed the goal
		# There is an initial call and response (are you ready for a target?) followed by the target itself
		#   goal_accept_callback handles accepting the goal
		#   cancel_callback is called if the goal is actually canceled by the action client
		#   execute_callback actually starts moving toward the goal
		self.action_server = ActionServer(node=self,
									action_type=NavTarget,
									action_name="nav_target",
									callback_group=ReentrantCallbackGroup(),
									goal_callback=self.goal_accept_callback,
									cancel_callback=self.cancel_callback,
									execute_callback=self.action_callback)
		# This is the goal in the robot's coordinate system, calculated in set_target
		self.target = PointStamped()
		self.target.point.x = 0.0
		self.target.point.y = 0.0
		# GUIDE: Declare any variables here
		# YOUR CODE HERE
		self.target_dist  = 0.0
		self.target_angle = 0.0
		# PDF recommendation: filter output values with 0.7 * new + 0.3 * old to smooth commands
		self.filtered_linear  = 0.0
		self.filtered_angular = 0.0
		# Track how long distance has been stuck to detect an unreachable blocked goal
		self.last_dist   = 0.0
		self.stuck_count = 0
		self.stuck_limit = 10   # skip goal after this many consecutive identical distance readings
		# Timer to make sure we publish the target marker (once we get a goal)
		self.marker_timer = self.create_timer(1.0, self._marker_callback)
		self.count_since_last_scan = 0
		self.print_twist_messages = False
		self.print_distance_messages = False
	def zero_twist(self):
		"""This is a helper class method to create and zero-out a twist"""
		# Don't really need to do this - the default values are zero - but can't hurt
		t = TwistStamped()
		t.header.frame_id = 'base_link'
		t.header.stamp = self.get_clock().now().to_msg()
		t.twist.linear.x = 0.0
		t.twist.linear.y = 0.0
		t.twist.linear.z = 0.0
		t.twist.angular.x = 0.0
		t.twist.angular.y = 0.0
		t.twist.angular.z = 0.0
		return t
	def _marker_callback(self):
		"""Publishes the target so it shows up in RViz"""
		with self.goal_lock:
			has_goal = (self.goal is not None)
		if not has_goal:
			# No goal, get rid of marker if there is one
			if self.target_marker:
				self.target_marker.action = Marker.DELETE
				self.target_pub.publish(self.target_marker)
				self.target_marker = None
				self.get_logger().info(f"Driver: Had an existing target marker; removing")
			return
		
		# If we do not currently have a marker, make one
		if not self.target_marker:
			self.target_marker = Marker()
			# Lock to read goal safely
			with self.goal_lock:
				if self.goal:
					self.target_marker.header.frame_id = self.goal.header.frame_id
			self.target_marker.id = 0
		
			self.get_logger().info(f"Driver: Creating Marker")
		# Build a marker for the target point
		#   - this prints out the green dot in RViz (the current target)
		self.target_marker.header.stamp = self.get_clock().now().to_msg()
		self.target_marker.type = Marker.SPHERE
		self.target_marker.action = Marker.ADD
		with self.goal_lock:
			if self.goal:
				self.target_marker.pose.position = self.goal.point
			else:
				# Goal disappeared; just bail out
				return
		self.target_marker.scale.x = 0.3
		self.target_marker.scale.y = 0.3
		self.target_marker.scale.z = 0.3
		self.target_marker.color.r = 0.0
		self.target_marker.color.g = 1.0
		self.target_marker.color.b = 0.0
		self.target_marker.color.a = 1.0
		# Publish the marker
		self.target_pub.publish(self.target_marker)
		# Turn off the timer so we don't just keep making and deleting the target Marker
		#   Will get turned back on when we get an goal request
		self.marker_timer.cancel()
	def goal_accept_callback(self, goal_request : ServerGoalHandle):
		"""Accept a request for a new goal"""
		self.get_logger().info("Received a goal request")
		# Timer to make sure we publish the new target
		self.marker_timer.reset()
		# Accept all goals. You can use this (in the future) to NOT accept a goal if you want
		return GoalResponse.ACCEPT
	
	def cancel_callback(self, goal_handle : ServerGoalHandle):
		"""Accept or reject a client request to cancel an action."""
		self.get_logger().info('Received a cancel request')
		# Make sure our goal is removed
		with self.goal_lock:
			self.goal = None
		# ...and robot stops
		t = self.zero_twist()
		self.cmd_pub.publish(t)
				
		# Timer to make sure we remove the current target (if there is one)
		self.marker_timer.reset()
		return CancelResponse.ACCEPT
	
	def close_enough(self):
		""" Return true if close enough to goal. This will be used in action_callback to stop moving toward the goal
		@ return true/false """
		# YOUR CODE HERE
		# target_dist is updated in set_target() from the scan callback
		# Close enough normally
		if self.target_dist <= self.threshold:
			self.stuck_count = 0
			self.last_dist = 0.0
			return True
		# If distance jumped up significantly, a new goal just arrived — reset stuck counter
		if self.target_dist > self.last_dist + 0.5:
			self.stuck_count = 0
		# Adapt to objects: if distance has not changed, goal is likely blocked by an obstacle.
		# Count consecutive checks with the same distance and skip after stuck_limit.
		# This lets the robot move on when a goal is unreachable due to an obstacle on top of it.
		if fabs(self.target_dist - self.last_dist) < 0.01:
			self.stuck_count += 1
		else:
			self.stuck_count = 0
		self.last_dist = self.target_dist
		if self.stuck_count >= self.stuck_limit:
			self.get_logger().info(f"Stuck at dist {self.target_dist:.2f} for {self.stuck_count} checks, skipping goal")
			self.stuck_count = 0
			return True
		return False

	def distance_to_target(self):
		""" Communicate with send points - set to distance to target"""
		return np.sqrt(self.target.point.x ** 2 + self.target.point.y ** 2)
	
	# Respond to the action request.
	def action_callback(self, goal_handle : ServerGoalHandle):
		""" This gets called when the new goal is sent by SendPoints
		@param goal_handle - this has the new goal
		@return a NavTarget return when done """
		self.get_logger().info(f'Received an execute goal request... {goal_handle.request.goal.point}')
	
		# Save the new goal as a stamped point
		new_goal = PointStamped()
		new_goal.header = goal_handle.request.goal.header
		new_goal.point = goal_handle.request.goal.point
		with self.goal_lock:
			self.goal = new_goal
		
		# Build a result to send back
		result = NavTarget.Result()
		result.success = False
		# Reset target
		self.set_target()
		# Keep publishing feedback, then sleeping (so the laser scan can happen)
		# GUIDE: If you aren't making progress, stop the while loop and mark the goal as failed
		rate = self.create_rate(0.5)
		while not self.close_enough():
			with self.goal_lock:
				if not self.goal:
					self.get_logger().info(f"Goal was canceled")
					return result
			
			feedback = NavTarget.Feedback()
			feedback.distance.data = self.distance_to_target()
			
			# Publish feedback - this gets sent back to send_points
			goal_handle.publish_feedback(feedback)
			# sleep so we can process the next scan
			rate.sleep()
			
		# Timer to make sure we remove the current target
		self.marker_timer.reset()
		# Don't keep processing goals
		with self.goal_lock:
			self.goal = None 
		# Publish the zero twist
		t = self.zero_twist()
		self.cmd_pub.publish(t)
		self.get_logger().info(f"Completed goal")
		# Set the succeed value on the handle
		goal_handle.succeed()
		# Set the result to True and return
		result.success = True
		return result

	def set_target(self):
		""" Convert the goal into an x,y position (target) in the ROBOT's coordinate space
		@return the new target as a Point """
		with self.goal_lock:
			local_goal = self.goal
		if local_goal:
			# Transforms for all coordinate frames in the robot are stored in a transform tree
			#  odom is the coordinate frame of the "world", base_link is the base link of the robot
			# A transform stores a rotation/translation to go from one coordinate system to the other
			transform = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
			# This applies the transform to the Stamped Point
			#    Note: This does not work, for reasons that are unclear to me
			self.target = do_transform_point(local_goal, transform)
			
			# This does the transform manually, by calculating the theta rotation from the quaternion
			euler_ang = -atan2(2 * transform.transform.rotation.z * transform.transform.rotation.w,
			                   1.0 - 2 * transform.transform.rotation.z * transform.transform.rotation.z)
			
			# Translate to the base link's origin
			x = local_goal.point.x - transform.transform.translation.x
			y = local_goal.point.y - transform.transform.translation.y
			# Do the rotation
			rot_x = x * cos(euler_ang) - y * sin(euler_ang)
			rot_y = x * sin(euler_ang) + y * cos(euler_ang)
			self.target.point.x = rot_x
			self.target.point.y = rot_y
			if self.print_distance_messages:
				self.get_logger().info(f'Target relative to robot: ({self.target.point.x:.2f}, {self.target.point.y:.2f}), orig ({local_goal.point.x, local_goal.point.y})')
			
		else:
			if self.print_distance_messages:
				self.get_logger().info(f'No target to get distance to')
			self.target = None		
		
		# GUIDE: Calculate any additional variables here
		#  Remember that the target's location is in its own coordinate frame at 0,0, angle 0 (x-axis)
		# YOUR CODE HERE
		if self.target:
			self.target_dist  = sqrt(self.target.point.x ** 2 + self.target.point.y ** 2)
			self.target_angle = atan2(self.target.point.y, self.target.point.x)
		else:
			self.target_dist  = 0.0
			self.target_angle = 0.0
		return self.target

	def scan_callback(self, scan):
		""" Lidar scan callback
		@param scan - has information about the scan, and the distances (see stopper.py in lab1)"""
	
		if self.print_twist_messages:
			self.get_logger().info("In scan callback")
		# Got a scan - set back to zero
		self.count_since_last_scan = 0
		with self.goal_lock:
			has_goal = (self.goal is not None)
		# If we have a goal, then act on it, otherwise stay still
		if has_goal:
			# Recalculate the target point (assumes we've moved)
			self.set_target()
			# Call the method to actually calculate the twist
			t = self.get_twist(scan)
		else:
			t = self.zero_twist()
			#t.twist.linear.x = 0.1
			if self.print_twist_messages:
				self.get_logger().info(f"No goal, sitting still")
		# Publish the new twist
		self.cmd_pub.publish(t)

	def get_obstacle(self, scan):
		""" check if an obstacle
		@param scan - the lidar scan
		@return Currently True/False and speed, angular turn"""
		if not self.target:
			return False, 0.0, 0.0
		
		# GUIDE: Use this method to collect obstacle information - is something in front of, to the left, or to 
		# the right of the robot? Start with your stopper code from Lab1
		# YOUR CODE HERE

		# Intelligent detection: detect each situation separately so get_twist
		# can decide the priority order (PDF recommendation).
		# Situations detected:
		#   A) Obstacle in front (between robot and target) — highest priority
		#   B) Wall too close on the left  — nudge right
		#   C) Wall too close on the right — nudge left
		# PDF reminder: positive y / positive angle = left, negative = right
		# PDF reminder: positive angular.z = turn left, negative = turn right

		# Only react to front obstacles that are actually closer than the target
		# (intelligent: don't react to walls that are past the goal)
		front_trigger = min(0.5, self.target_dist * 0.8)  # don't react beyond 80% of target dist
		front_trigger = max(front_trigger, 0.3)            # but always keep a minimum safety zone
		side_trigger  = 0.4    # react if side wall is this close
		cone_angle    = np.radians(40)   # front cone +-40 deg
		side_angle    = np.radians(80)   # side zone 40-80 deg

		angle_min       = scan.angle_min
		angle_increment = scan.angle_increment

		min_front_left  = float('inf')
		min_front_right = float('inf')
		min_side_left   = float('inf')
		min_side_right  = float('inf')

		for i, r in enumerate(scan.ranges):
			if r < scan.range_min or r > scan.range_max or np.isnan(r) or np.isinf(r):
				continue
			angle = angle_min + i * angle_increment
			if fabs(angle) <= cone_angle:
				if angle >= 0.0:
					min_front_left  = min(min_front_left,  r)
				else:
					min_front_right = min(min_front_right, r)
			elif cone_angle < angle <= side_angle:
				min_side_left  = min(min_side_left,  r)
			elif -side_angle <= angle < -cone_angle:
				min_side_right = min(min_side_right, r)

		nearest_front = min(min_front_left, min_front_right)
		front_blocked = nearest_front < front_trigger
		wall_left     = min_side_left  < side_trigger
		wall_right    = min_side_right < side_trigger

		if not front_blocked and not wall_left and not wall_right:
			return False, 0.0, 0.0

		linear_adjust  = 0.0
		angular_adjust = 0.0

		# Priority order (PDF recommendation: decide what order to handle situations):
		# 1) Front obstacle is most dangerous — handle first
		# 2) Side walls are secondary — only if not also front-blocked
		if front_blocked:
			# Slow down proportional to how close (tanh-like mapping, PDF recommendation)
			linear_adjust = -(front_trigger - nearest_front) / front_trigger  # -1 to 0

			# Turn toward whichever side has more space
			# If left is closer, turn right (negative); if right is closer, turn left (positive)
			if min_front_left <= min_front_right:
				angular_adjust = -1.0   # obstacle closer on left — turn right
			else:
				angular_adjust = 1.0    # obstacle closer on right — turn left

			# Special case: if BOTH sides are blocked in front, pick the side
			# with more space in the wider side zone
			if min_front_left < front_trigger and min_front_right < front_trigger:
				if min_side_left >= min_side_right:
					angular_adjust = 1.0    # more side space on left — turn left
				else:
					angular_adjust = -1.0   # more side space on right — turn right

		elif wall_left and not wall_right:
			# Too close on left only — nudge right
			angular_adjust = -0.5

		elif wall_right and not wall_left:
			# Too close on right only — nudge left
			angular_adjust = 0.5

		elif wall_left and wall_right:
			# Both sides close — go straight, no turn
			angular_adjust = 0.0

		return True, linear_adjust, angular_adjust

	def get_twist(self, scan):
		"""This is the method that calculate the twist
		@param scan - a LaserScan message with the current data from the LiDAR.  Use this for obstacle avoidance. 
		    This is the same as your lab1 go and stop code
		@return a twist command"""
		t = self.zero_twist()
		# GUIDE:
		#  Step 1) Calculate the angle the robot has to turn to in order to point at the target
		#  Step 2) Set your speed based on how far away you are from the target, as before
		#  Step 3) Add code that veers left (or right) to avoid an obstacle in front of it
		# Reminder: t.linear.x = 0.1    sets the forward speed to 0.1
		#           t.angular.z = pi/2   sets the angular speed to 90 degrees per sec
		# Reminder 2: target is in self.target 
		#  Note: If the target is behind you, might turn first before moving
		#  Note: 0.4 is a good speed if nothing is in front of the robot
		min_speed = 0.05
		max_speed = 0.2         # This moves about 0.01 m between scans
		max_turn = np.pi * 0.1  # This turns about 2 degrees between scans
		# YOUR CODE HERE
		if not self.target:
			return t

		# Step 3: check for obstacle first — determine situation before choosing action
		obstacle, linear_adjust, angular_adjust = self.get_obstacle(scan)

		# Step 1: angle and distance to target in robot frame
		tx    = self.target.point.x
		ty    = self.target.point.y
		angle = atan2(ty, tx)
		dist  = sqrt(tx * tx + ty * ty)

		# If target is behind us, spin toward it — obstacle avoidance paused
		if tx < 0.0:
			new_linear  = 0.0
			new_angular = max_turn if angle >= 0.0 else -max_turn

		elif obstacle and linear_adjust <= -0.8:
			# Very close to obstacle — stop completely, spin away
			# Obstacle takes full priority over goal seeking
			new_linear  = 0.0
			new_angular = max_turn * 4.0 * angular_adjust

		else:
			# Step 2: goal-seeking using tanh as PDF specifies
			# Map angle to [-1,1] range for tanh input (pi/4 = 45 deg hits max turn)
			new_angular = max_turn * tanh(angle / (pi / 4.0))
			# Speed: tanh of distance, scaled so 2m = near max speed
			new_linear  = max_speed * tanh(dist / 2.0)
			# Slow down when turning hard — graceful approach to goals
			turn_factor = max(0.0, 1.0 - fabs(angle) / (pi / 2.0))
			new_linear  = new_linear * turn_factor
			# Keep minimum speed when aligned and not at goal yet
			if dist > self.threshold * 1.5 and fabs(angle) < 0.35:
				new_linear = max(new_linear, min_speed)

			# Step 3: blend obstacle correction on top of goal seeking
			if obstacle:
				# Reduce speed based on obstacle proximity
				new_linear  = max(0.0, new_linear + max_speed * linear_adjust)
				# Add obstacle turn correction — obstacle gets priority over goal turn
				new_angular = float(np.clip(
					new_angular + max_turn * 3.0 * angular_adjust, -pi, pi))

		# PDF recommendation: filter output with 0.7 * new + 0.3 * old
		# This smooths out wobble from scan-to-scan noise
		self.filtered_linear  = 0.7 * new_linear  + 0.3 * self.filtered_linear
		self.filtered_angular = 0.7 * new_angular + 0.3 * self.filtered_angular

		t.twist.linear.x  = float(self.filtered_linear)
		t.twist.angular.z = float(self.filtered_angular)

		if self.print_twist_messages:
			self.get_logger().info(f"Setting twist forward {t.twist.linear.x} angle {t.twist.angular.z}")
		return t

# The idiom in ROS2 is to use a function to do all of the setup and work.  This
# function is referenced in the setup.py file as the entry point of the node when
# we're running the node with ros2 run.  The function should have one argument, for
# passing command line arguments, and it should default to None.
def main(args=None):
	# Initialize rclpy.  We should do this every time.
	rclpy.init(args=args)
	# Make a node class.  The idiom in ROS2 is to encapsulte everything in a class
	# that derives from Node.
	driver = Lab3Driver()
	# Multi-threaded execution
	executor = MultiThreadedExecutor()
	executor.add_node(driver)
	executor.spin()
	
	# Make sure we shutdown everything cleanly.  This should happen, even if we don't
	# include this line, but you should do it anyway.
	rclpy.shutdown()
	
# If we run the node as a script, then we're going to start here.
if __name__ == '__main__':
	# The idiom in ROS2 is to set up a main() function and to call it from the entry
	# point of the script.
	main()
