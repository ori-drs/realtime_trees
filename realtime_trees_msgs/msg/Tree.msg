# Definition of a tree
std_msgs/Header header
int64 id

geometry_msgs/Pose pose
realtime_trees_msgs/Circle[] circles
shape_msgs/Mesh canopy_mesh

# Tree traits (-1.0 if not available)
float64 coverage_angle
float64 dbh
float64 canopy_volume
bool reconstructed

