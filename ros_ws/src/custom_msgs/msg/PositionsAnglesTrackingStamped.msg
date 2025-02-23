# PointAnglesStamped.msg
std_msgs/Header header
float32[] positions_x  # Array of x-coordinates
float32[] positions_y  # Array of y-coordinates
float32[] angles_sin # Array of sin values
float32[] angles_cos # Array of cos values
float32 target_tracking_velocity 