import numpy as np


def velocity_join_files():
    value_index = 2
    folder = "/home/patik/.ros/avoidance_test/"
    result_file_name = "data_table.csv"
    files = ["log_file_1.txt", "log_file_2.txt", "log_file_3.txt", "log_file_4.txt", "log_file_5.txt", "log_file_6.txt"]
    column_names = ["Time"]
    for i in range(len(files), 0, -1):
        column_names.append("; 6/{} m/s".format(i))
    join_files(value_index, folder, result_file_name, files, column_names)


def texture_join_files():
    value_index = 2
    folder = "/home/patik/.ros/avoidance_test/"
    result_file_name = "data_table_small_textures.csv"
    files = []
    column_names = ["Time"]
    for typ in ["colour", "mossy", "red_bricks", "squares"]:
        for size in ["small"]:
            column_names.append("; {} - {}".format(typ, size))
            files.append("log_file_{}_{}.txt".format(typ, size))
    join_files(value_index, folder, result_file_name, files, column_names)


def join_files(value_index, folder, result_file_name, files, column_names):
    files_lines = []
    for file in files:
        with open(folder + file, "r") as f:
            files_lines.append(f.readlines())
    indices = np.ones(len(files), dtype=np.int32)
    with open(folder + result_file_name, "w") as result_file:
        line = ""
        line = line.join(column_names)
        for column_name in column_names:
            pass
        result_file.write(line + "\n")
        for t in np.arange(0, 60, 0.25):
            line = "{}".format(t)
            for i in range(len(files)):
                while True:
                    if indices[i] < len(files_lines[i]):
                        index_line = files_lines[i][indices[i]].split(",")
                        if indices[i] + 1 >= len(files_lines[i]):
                            line += "; {}".format(float(index_line[value_index]))
                            break
                        plus_one_index_line = files_lines[i][indices[i] + 1].split(",")
                        if float(index_line[1]) > t or float(index_line[1]) <= t < float(plus_one_index_line[1]):
                            line += "; {}".format(float(index_line[value_index]))
                            break
                    else:
                        index_line = files_lines[i][indices[i] - 1].split(",")
                        line += "; {}".format(float(index_line[value_index]))
                        break
                    indices[i] += 1
            result_file.write(line + "\n")


texture_join_files()
