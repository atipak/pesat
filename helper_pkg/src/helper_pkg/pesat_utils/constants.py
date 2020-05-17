class Constants():
    class InputOutputParameterType:
        pose = 0
        map = 1
        pointcloud = 2

    class MapPresenceParameter:
        yes = 0
        no = 1

    class PredictionParametersNames:
        input_type = 0
        drone_positions_list = 1
        target_positions_list = 2
        map_presence = 3
        output_type = 4

    class PointCloudNames:
        X = 0
        Y = 1
        Z = 2
        ROLL = 3
        PITCH = 4
        YAW = 5
        PREDECESSOR = 6
        WEIGHT = 7
