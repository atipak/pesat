Header header        # Header timestamp should be acquisition time

uint32 height         # height, that is, number of rows
uint32 width          # width, that is, number of columns

geometry_msgs/Point center # center of the image

pesat_msgs/FloatArray data  # actual matrix data