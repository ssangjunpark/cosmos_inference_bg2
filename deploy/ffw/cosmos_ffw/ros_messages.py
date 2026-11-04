"""ROS message conversion without initializing a ROS context or opening a connection."""
def joint_trajectory(data):
    from builtin_interfaces.msg import Duration, Time
    from std_msgs.msg import Header
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    msg = JointTrajectory()
    msg.header = Header(stamp=Time(**data["header"]["stamp"]), frame_id=data["header"].get("frame_id", ""))
    msg.joint_names = list(data["joint_names"])
    for point in data["points"]:
        item = JointTrajectoryPoint()
        item.positions = list(map(float, point["positions"]))
        item.time_from_start = Duration(**point["time_from_start"])
        msg.points.append(item)
    return msg
