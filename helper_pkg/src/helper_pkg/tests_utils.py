import numpy as np
import matplotlib.pyplot as plt
import yaml
from pesat_utils.pesat_math import Math
from pesat_utils.base_map import BaseMap
import itertools
import copy
from scipy import stats
import os
import re
import json
from section_algorithm_utils import SectionUtils


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


def draw_target_tracks():
    folder_path = "/home/patik/.ros/tracking_test"
    data = {}
    for file in os.listdir(folder_path):
        if file.startswith("target"):
            file_id = file[16:len(file) - 4]
            target_file = "{}/target_log_file_{}.txt".format(folder_path, file_id)
            drone_file = "{}/drone_log_file_{}.txt".format(folder_path, file_id)
            if not os.path.exists(drone_file):
                print("Drone file {} not found, skipping target file {}.".format(drone_file, target_file))
                continue
            target_data = read_data_from_file(target_file)
            true_sampled_data, true_start_time = get_true_target_track(target_data, target_file)
            drone_data = read_data_from_file(drone_file)
            pred_sampled_data, pred_start_time = get_prediction_target_track(drone_data)
            drone_sampled_data, drone_start_time = get_drone_track(drone_data)
            drone_time_orientation_data, times_start_time = get_drone_time_or_data(drone_data)
            start_time = max(true_start_time, pred_start_time)
            true_index = int(start_time - true_start_time)
            pred_index = int(start_time - pred_start_time)
            end_time = int(min(len(true_sampled_data) - true_index, len(pred_sampled_data) - pred_index))
            end_time = np.clip(end_time, None, 121)
            if end_time < 60:
                print("Skipping processing of files with id {} because simulation was finished too early {}.".format(
                    file_id, end_time))
                continue
            under_vision, out_of_vision, uv_lines, oov_lines = drone_target_comparison(pred_sampled_data,
                                                                                       true_sampled_data, pred_index,
                                                                                       end_time)
            diffs_calling_speed = calling_speed(drone_time_orientation_data, pred_index, end_time)
            diffs_orientation = drone_rotation(drone_sampled_data, pred_index, end_time)
            diffs_drone_speed, diffs_drone_speed_without_staying = drone_speed(drone_sampled_data, pred_index, end_time)
            diffs_target_speed, diffs_target_speed_without_staying = target_speed(true_sampled_data, pred_index,
                                                                                  end_time)
            path_to_file = "{}/result_{}.txt".format(folder_path, file_id)
            with open(path_to_file, "w") as file_stream:
                drone_target_comparison_statistics(under_vision, out_of_vision, uv_lines, oov_lines, file_stream, data,
                                                   file_id)
                drone_diffs_statistics(file_stream, diffs_calling_speed, diffs_orientation, diffs_drone_speed,
                                       diffs_target_speed, data, diffs_drone_speed_without_staying,
                                       diffs_target_speed_without_staying, file_id)
    return data


def calling_speed(drone_sampled_data, pred_index, end_time):
    diffs = []
    for i in range(pred_index, pred_index + end_time - 2):
        diff = drone_sampled_data[i][0] - drone_sampled_data[i + 1][0]
        diffs.append(abs(diff))
    return diffs


def target_speed(target_sampled_data, pred_index, end_time):
    diffs = []
    diffs_without_staying = []
    maximal_speed = 3.1
    minimal_speed = 0.3
    for i in range(pred_index, pred_index + end_time - 2):
        diff_x = target_sampled_data[i][0] - target_sampled_data[i + 1][0]
        diff_y = target_sampled_data[i][1] - target_sampled_data[i + 1][1]
        result = np.hypot(diff_x, diff_y)
        if result <= maximal_speed:
            if result > minimal_speed:
                diffs.append(result)
                diffs_without_staying.append(result)
            else:
                diffs.append(result)
        else:
            print(result)
    return diffs, diffs_without_staying


def drone_speed(drone_sampled_data, pred_index, end_time):
    diffs = []
    diffs_without_staying = []
    maximal_speed = 7
    minimal_speed = 0.2
    for i in range(pred_index, pred_index + end_time - 2):
        diff_x = drone_sampled_data[i][0] - drone_sampled_data[i + 1][0]
        diff_y = drone_sampled_data[i][1] - drone_sampled_data[i + 1][1]
        diff_z = drone_sampled_data[i][2] - drone_sampled_data[i + 1][2]
        result = np.hypot(np.hypot(diff_x, diff_y), diff_z)
        if result <= maximal_speed:
            if result > minimal_speed:
                diffs.append(np.hypot(np.hypot(diff_x, diff_y), diff_z))
                diffs_without_staying.append(np.hypot(np.hypot(diff_x, diff_y), diff_z))
            else:
                diffs.append(np.hypot(np.hypot(diff_x, diff_y), diff_z))

    return diffs, diffs_without_staying


def drone_rotation(drone_data, pred_index, end_time):
    diffs = []
    maximal_speed = 0.7
    for i in range(pred_index, pred_index + end_time - 2):
        # time = np.abs(drone_data[i][0] - drone_data[i + 1][0])
        time = 1
        diff_orientation = np.abs(drone_data[i][3] - drone_data[i + 1][3])
        diffs.append(np.clip(diff_orientation, None, maximal_speed))
        if diff_orientation <= time * maximal_speed and time > 0.1:
            pass  # diffs.append(diff_orientation)
        else:
            print(diff_orientation)
    return diffs


def drone_diffs_statistics(file_stream, diffs_time, diffs_orientation, diffs_drone_speed, diffs_target_speed, data,
                           diffs_drone_speed_without_staying, diffs_target_speed_without_staying, file_id):
    diffs_stats = stats.describe(diffs_time)
    data[file_id].append([diffs_stats[0], diffs_stats[1][0], diffs_stats[1][1], diffs_stats[2],
                          diffs_stats[3], diffs_stats[4], diffs_stats[5]])
    write_stats(file_stream, diffs_stats)
    diffs_stats = stats.describe(diffs_orientation)
    data[file_id].append([diffs_stats[0], diffs_stats[1][0], diffs_stats[1][1], diffs_stats[2],
                          diffs_stats[3], diffs_stats[4], diffs_stats[5]])
    write_stats(file_stream, diffs_stats)
    diffs_stats = stats.describe(diffs_drone_speed)
    data[file_id].append([diffs_stats[0], diffs_stats[1][0], diffs_stats[1][1], diffs_stats[2],
                          diffs_stats[3], diffs_stats[4], diffs_stats[5]])
    write_stats(file_stream, diffs_stats)
    diffs_stats = stats.describe(diffs_target_speed)
    data[file_id].append([diffs_stats[0], diffs_stats[1][0], diffs_stats[1][1], diffs_stats[2],
                          diffs_stats[3], diffs_stats[4], diffs_stats[5]])
    write_stats(file_stream, diffs_stats)
    diffs_stats = stats.describe(diffs_drone_speed_without_staying)
    data[file_id].append([diffs_stats[0], diffs_stats[1][0], diffs_stats[1][1], diffs_stats[2],
                          diffs_stats[3], diffs_stats[4], diffs_stats[5]])
    write_stats(file_stream, diffs_stats)
    diffs_stats = stats.describe(diffs_target_speed_without_staying)
    data[file_id].append([diffs_stats[0], diffs_stats[1][0], diffs_stats[1][1], diffs_stats[2],
                          diffs_stats[3], diffs_stats[4], diffs_stats[5]])
    write_stats(file_stream, diffs_stats)


def drone_target_comparison_statistics(under_vision, out_of_vision, uv_lines, oov_lines, file_stream, data, file_id):
    # print(file_id)
    d = []
    uv_stats = stats.describe(under_vision)
    # print("Under vision", uv_stats)
    d.append(
        [uv_stats[0], uv_stats[1][0], uv_stats[1][1], uv_stats[2], uv_stats[3], uv_stats[4], uv_stats[5]])
    write_stats(file_stream, uv_stats)
    oov_stats = stats.describe(out_of_vision)
    # print("Out of vision", oov_stats)
    d.append([oov_stats[0], oov_stats[1][0], oov_stats[1][1], oov_stats[2], oov_stats[3], oov_stats[4],
              oov_stats[5]])
    write_stats(file_stream, oov_stats)
    uv_lines_stats = stats.describe(uv_lines)
    # print("Under vision lines", uv_lines_stats)
    d.append([uv_lines_stats[0], uv_lines_stats[1][0], uv_lines_stats[1][1], uv_lines_stats[2],
              uv_lines_stats[3], uv_lines_stats[4], uv_lines_stats[5]])
    write_stats(file_stream, uv_lines_stats)
    oov_lines_stats = stats.describe(oov_lines)
    # print("Out of vision lines", oov_lines_stats)
    d.append([oov_lines_stats[0], oov_lines_stats[1][0], oov_lines_stats[1][1], oov_lines_stats[2],
              oov_lines_stats[3], oov_lines_stats[4], oov_lines_stats[5]])
    write_stats(file_stream, oov_lines_stats)
    data[file_id] = d


def drone_target_comparison(pred_sampled_data, true_sampled_data, pred_index, end_time):
    under_vision = []
    out_of_vision = []
    uv_lines = []
    oov_lines = []
    uv_oov_bool_var = not pred_sampled_data[0, 2] == 1
    uv_oov_var = 0
    # plt.figure()
    # world_map = BaseMap.map_from_file(map_file, 2)
    # draw_list_obstacles(world_map)
    for i in range(pred_index, pred_index + end_time - 1):
        """plt.plot([pred_sampled_data[i, 0], pred_sampled_data[i + 1, 0]],
                 [pred_sampled_data[i, 1], pred_sampled_data[i + 1, 1]],
                 "b" + line_type(pred_sampled_data[i, 2]))
        plt.plot([true_sampled_data[i, 0], true_sampled_data[i + 1, 0]],
                 [true_sampled_data[i, 1], true_sampled_data[i + 1, 1]],
                 "k" + line_type(true_sampled_data[i, 2]))"""
        uv_oov_var += 1
        if pred_sampled_data[i, 2] == 1:
            under_vision.append(np.hypot(pred_sampled_data[i, 0] - true_sampled_data[i, 0],
                                         pred_sampled_data[i, 1] - true_sampled_data[i, 1]))
            if not uv_oov_bool_var:
                oov_lines.append(uv_oov_var)
                uv_oov_bool_var = True
                uv_oov_var = 0
        else:
            out_of_vision.append(np.hypot(pred_sampled_data[i, 0] - true_sampled_data[i, 0],
                                          pred_sampled_data[i, 1] - true_sampled_data[i, 1]))
            if uv_oov_bool_var:
                uv_lines.append(uv_oov_var)
                uv_oov_bool_var = False
                uv_oov_var = 0
    if uv_oov_bool_var:
        uv_lines.append(uv_oov_var)
    else:
        oov_lines.append(uv_oov_var)
    return under_vision, out_of_vision, uv_lines, oov_lines


def show_and_save_statistics(data, file_name, folder_path):
    reduced_data = reduce_by_key(data)
    reduced_data_columns = [key for key in reduced_data]
    show_nods_for_every_group(reduced_data)
    f_array = [height_vs_size_and_built, size_vs_built_and_height, built_vs_size_and_height]
    cartesian_row_names = create_cartesian(["mean", "min", "max"])
    name = ["nobs", "min", "max", "mean", "variance", "skewness", "kurtosis"]
    types = ["Under vision", "Out of vision", "Under vision lines", "Out of vision lines", "Calling times",
             "Orientation", "Drone speed", "Target speed", "Non zero drone speed", "Non zero target speed"]
    statisctics_types = ["min:min", "max:max", "mean:mean", "max:min", "min:max"]
    sizes = [10, 100]
    builts = [0.1, 0.2, 0.3]
    heights = [0, 20, 1]
    if False:
        with open(file_name, "w") as file_stream:
            for f in f_array:
                columns = f(heights, sizes, builts, reduced_data_columns)
                for column in columns:
                    if len(column) <= 1:
                        continue
                    result_rows = compare(column, reduced_data, name, types)
                    rows, rows_names, columns_names = get_correct_columns(column, statisctics_types, types)
                    filled_rows = fill_rows(rows, result_rows)
                    plot_graphs(filled_rows, columns_names, rows_names, folder_path)
                    print(column)
                    file_stream.write("{}\n".format(array_to_csv(column)))
                    print(columns_names)
                    file_stream.write(",{}\n".format(array_to_csv(columns_names)))
                    for i in range(len(filled_rows)):
                        file_stream.write("{}, {}\n".format(rows_names[i], array_to_csv(filled_rows[i])))
                    file_stream.write("\n\n".format(array_to_csv(column)))
    maps_types = ["10_0.1_0", "10_0.3_0", "100_0.1_0", "100_0.2_0", "100_0.3_0", "100_0.1_20", "100_0.3_20"]
    result_rows = compare(maps_types, reduced_data, name, types)
    info = get_subplot_informations(types, statisctics_types, maps_types)
    resolve_subplot_info(info, result_rows, folder_path)
    """
    result_rows, row_names, column_names = compare(["100_0.1_0", "100_0.1_20"], reduced_data)
    result_rows, row_names = get_only_begin_with(["mean", "variance", "min", "max"], result_rows, row_names)
    print(column_names)
    for i in range(len(result_rows)):
        print(row_names[i], result_rows[i])
    """
    # plt.savefig("{}/image_{}.png".format(folder_path, file_id))
    # plt.close()


def plot_graphs(rows, column_names, rows_names, folder_path):
    shift = int(len(column_names) / 8)
    for i in range(0, len(column_names), shift):
        filled_rows = []
        for j in range(len(rows)):
            if rows[j][i] != "":
                c_names = []
                for k in range(shift):
                    c_names.append(column_names[i + k])
                rows_data = []
                for k in range(shift):
                    rows_data.append(rows[j][i + k])
                filled_rows.append(
                    [rows_names[j], c_names, rows_data])
        show_plot(filled_rows, folder_path)


def show_plot(data, folder_path):
    names = []
    [_, title] = data[0][1][0].split("-")
    for i in range(len(data)):
        fig, axs = plt.subplots(1, 1, constrained_layout=True, squeeze=False, figsize=(3, 6))
        fig.suptitle(translation(title), fontsize=16)
        names = [data[i][1][t].split("-")[0] for t in range(len(data[i][1]))]
        rects = axs[0, 0].bar(names, data[i][2])
        axs[0, 0].set_title(translation(data[i][0]))
        axs[0, 0].set_ylabel(metrics(title))
        for rect in rects:
            height = np.round(rect.get_height(), 3)
            axs[0, 0].annotate('{}'.format(height),
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
        s = ""
        for n in names:
            s += "{}-".format(n)
        if not os.path.exists(os.path.join(folder_path, title)):
            os.mkdir(os.path.join(folder_path, title))
        if not os.path.exists(os.path.join(folder_path, title, translation(data[i][0]))):
            os.mkdir(os.path.join(folder_path, title, translation(data[i][0])))
        plt.savefig(
            "{}/{}{}.png".format(os.path.join(folder_path, title, translation(data[i][0])), s, translation(title)))
        plt.close()


def metrics(value):
    metric = "Metry"
    value = value.strip()
    if "Under vision lines" == value or "Out of vision lines" == value or "Calling times" == value:
        metric = "Sekundy"
    if "Drone speed" == value or "Target speed" == value:
        metric = "Metry za sekundu"
    if "Orientation" == value:
        metric = "Radiany"
    return metric


def translation(values):
    str_value = False
    if isinstance(values, str):
        values = [values]
        str_value = True
    translated = []
    for value in values:
        translation = ""
        value = value.strip()
        if "Under vision" == value:
            translation = "Pod dohledem"
        if "Out of vision" == value:
            translation = "Mimo dohled"
        if "Under vision lines" == value:
            translation = "Viditelná řada"
        if "Out of vision lines" == value:
            translation = "Neviditelná řada"
        if "Calling times" == value:
            translation = "Časová odezva"
        if "Drone speed" == value:
            translation = "Rychlost drona"
        if "Target speed" == value:
            translation = "Rychlost cíle"
        if "Non zero target speed" == value:
            translation = "Nenulová rychlost cíle"
        if "Non zero drone speed" == value:
            translation = "Nenulová rychlost drona"
        if "Orientation" == value:
            translation = "Orientace"
        if "min:min" == value:
            translation = "Minimum"
        if "max:max" == value:
            translation = "Maximum"
        if "mean:mean" == value:
            translation = "Průměr"
        if "max:min" == value:
            translation = "Minimum"
        if "min:max" == value:
            translation = "Maximum"

        translated.append(translation)
    if str_value:
        translated = translated[0]
    return translated


def fill_rows(rows, data):
    filled_rows = []
    for row in rows:
        filled_row = []
        for r in row:
            if r is not None:
                filled_row.append(data[r])
            else:
                filled_row.append("")
        filled_rows.append(filled_row)
    return filled_rows


def array_to_csv(array, delimiter=","):
    s = ""
    for i in range(len(array) - 1):
        s += "{}{}".format(array[i], delimiter)
    s += str(array[-1])
    return s


def create_cartesian(names):
    result_names = []
    for name in names:
        for name1 in names:
            result_names.append("{}:{}".format(name, name1))
    return result_names


def height_vs_size_and_built(heights, sizes, builts, corect_names):
    result = []
    for size in sizes:
        for built in builts:
            r = []
            for height in heights:
                n = "{}_{}_{}".format(size, built, height)
                if n in corect_names:
                    r.append("{}_{}_{}".format(size, built, height))
            if len(r) > 1:
                result.append(r)
    return result


def built_vs_size_and_height(heights, sizes, builts, corect_names):
    result = []
    for size in sizes:
        for height in heights:
            r = []
            for built in builts:
                n = "{}_{}_{}".format(size, built, height)
                if n in corect_names:
                    r.append("{}_{}_{}".format(size, built, height))
            if len(r) > 1:
                result.append(r)
    return result


def size_vs_built_and_height(heights, sizes, builts, corect_names):
    result = []
    for built in builts:
        for height in heights:
            r = []
            for size in sizes:
                n = "{}_{}_{}".format(size, built, height)
                if n in corect_names:
                    r.append("{}_{}_{}".format(size, built, height))
            if len(r) > 1:
                result.append(r)
    return result


def show_nods_for_every_group(data):
    print("NODS")
    for name in data:
        print("{}:{}".format(name, data[name][0]))


def get_only_begin_with(types, data, rows_names):
    i = 0
    ii = []
    chosen_names = []
    for name in rows_names:
        for typ in types:
            if str(name).startswith(typ):
                ii.append(i)
                chosen_names.append(name)
                i += 1
    return np.array(data)[ii], chosen_names


def get_correct_columns(keys, statisctics_types, types):
    s = {}
    s[types[0]] = [statisctics_types[0], statisctics_types[2]]
    s[types[1]] = [statisctics_types[4], statisctics_types[2]]
    s[types[2]] = [statisctics_types[4], statisctics_types[2]]
    s[types[3]] = [statisctics_types[1], statisctics_types[2]]
    s[types[4]] = [statisctics_types[1], statisctics_types[2]]
    s[types[5]] = [statisctics_types[2]]
    s[types[6]] = [statisctics_types[1], statisctics_types[2]]
    s[types[7]] = [statisctics_types[2]]
    s[types[8]] = [statisctics_types[1], statisctics_types[2]]
    s[types[9]] = [statisctics_types[2]]
    rows = []
    for st in statisctics_types:
        row = []
        for typ in types:
            if st in s[typ]:
                for key in keys:
                    row.append("{}:{}:{}".format(st, typ, key))
            else:
                for _ in keys:
                    row.append(None)
        rows.append(row)
    columns_names = []
    for typ in types:
        for key in keys:
            columns_names.append("{} - {}".format(key, typ))
    rows_names = statisctics_types
    return rows, rows_names, columns_names


def resolve_subplot_info(info, d_set, folder_path):
    index = -1
    for i in info:
        index += 1
        data = info[i]
        s = 0
        key_count = 0
        for j in range(len(i)):
            for key in data[j]:
                key_count += 1
                s += len(data[j][key])
        fig, axs = plt.subplots(1, key_count, constrained_layout=True, squeeze=False, figsize=(2 * s, 6))
        plt.box(False)
        k = -1
        for j in range(len(i)):
            typ = i[j]
            # fig.suptitle(translation(typ), fontsize=16)
            key_data_set = data[j]
            for statistic_type in key_data_set:
                k += 1
                map_types = key_data_set[statistic_type]
                values = []
                for map_type in map_types:
                    set_key = "{}:{}:{}".format(statistic_type, typ, map_type)
                    values.append(d_set[set_key])
                rects = axs[0, k].bar(map_types, values)
                axs[0, k].set_title("{}\n{}".format(translation(typ), translation(statistic_type)), fontsize=18)
                axs[0, k].set_ylabel(metrics(typ), fontsize=16)
                axs[0, k].tick_params(axis="both", which="major", labelsize=16, top=False, right=False)
                axs[0, k].set_frame_on(False)
                axs[0, k].yaxis.grid(True)
                for rect in rects:
                    height = np.round(rect.get_height(), 3)
                    axs[0, k].annotate('{}'.format(height),
                                       xy=(rect.get_x() + rect.get_width() / 2, height),
                                       xytext=(0, 3),  # 3 points vertical offset
                                       textcoords="offset points",
                                       ha='center', va='bottom', fontsize=16)
        # plt.show()
        if not os.path.exists(os.path.join(folder_path, "images")):
            os.mkdir(os.path.join(folder_path, "images"))
        plt.savefig(
            "{}/{}.png".format(os.path.join(folder_path, "images"), str(index)))
        plt.close()


def get_subplot_informations(types, statistics_types, maps_types):
    """types = ["Under vision", "Out of vision", "Under vision lines", "Out of vision lines", "Calling times",
             "Orientation", "Drone speed", "Target speed", "Non zero drone speed", "Non zero target speed"]

             statisctics_types = ["min:min", "max:max", "mean:mean", "max:min", "min:max"]"""
    info = {tuple((types[1], types[2])): [{
        statistics_types[2]: [maps_types[5], maps_types[2], maps_types[3], maps_types[4], maps_types[1]]},
        {statistics_types[2]: [maps_types[5], maps_types[2], maps_types[3], maps_types[4], maps_types[1],
                               maps_types[0]]}
    ],

        tuple((types[3],)): [
            {statistics_types[2]: [maps_types[1], maps_types[0], maps_types[2], maps_types[3], maps_types[4]],
             statistics_types[1]: [maps_types[2], maps_types[3], maps_types[4]]}],
        tuple((types[4],)): [{statistics_types[2]: [maps_types[0], maps_types[2], maps_types[3], maps_types[4]],
                              statistics_types[1]: [maps_types[1], maps_types[4]]}],
        tuple((types[5], types[7])): [
            {statistics_types[2]: [maps_types[0], maps_types[1], maps_types[4], maps_types[2]]},
            {statistics_types[2]: [maps_types[0], maps_types[2], maps_types[4]]}],
        tuple((types[6], types[0])): [
            {statistics_types[2]: [maps_types[6], maps_types[2], maps_types[3], maps_types[4]]},
            {statistics_types[2]: [maps_types[0], maps_types[2], maps_types[3], maps_types[4]]}]
    }
    return info


def reduce_by_key(data):
    set_d = {}
    for d in data:
        splited = d.split("-")
        [size, built] = splited[0:2]
        if splited[2] == "":
            height = "0"
        else:
            height = splited[2]
        built = np.round(float(built.replace("_", ".")), 1)
        key = size + "_" + str(built) + "_" + height
        if key not in set_d:
            set_d[key] = [[] for i in range(len(data[d]))]
        for i in range(len(data[d])):
            set_d[key][i].append(data[d][i])
    for key in set_d:
        stat = stats.describe(set_d[key], axis=1)
        set_d[key] = [stat[0], stat[1][0], stat[1][1], stat[2], stat[3], stat[4], stat[5]]
    return set_d


def compare(keys, set_d, name, types):
    row_names = []
    column_names = []
    result_rows = []
    s = ""
    for t in types:
        for k in keys:
            column_names.append(k + "-" + t)
            s += "   {}   ".format(k + " " + t)
    result_set = {}
    for i in range(1, len(name)):
        item = set_d[keys[0]][i]
        item_type = item[0]
        if len(item_type) > 1:
            for k in range(len(item_type)):
                s = name[i] + ":" + name[k]
                row_names.append(name[i] + ":" + name[k])
                row = []
                for j in range(len(types)):
                    for key in keys:
                        row.append(set_d[key][i][j][k])
                        set_key = "{}:{}:{}:{}".format(name[i], name[k], types[j], key)
                        result_set[set_key] = set_d[key][i][j][k]
                        s += "   {}  ".format(set_d[key][i][j][k])
                result_rows.append(row)
        else:
            s = ""
            row = []
            for j in range(len(types)):
                for key in keys:
                    row.append(set_d[key][i][j])
                    s += "     {}    ".format(set_d[key][i][j])
            result_rows.append(row)
    return result_set

    """
    key = "100_0.1_0"
    print("{}:{}".format(key, stats.describe(set_d[key], axis=1)))
    key = "100_0.1_20"
    print("{}:{}".format(key, stats.describe(set_d[key], axis=1)))
    """


def write_stats(file, stats):
    file.write(
        "{},{},{},{},{},{},{}\n".format(stats[0], stats[1][0], stats[1][1], stats[2], stats[3], stats[4], stats[5]))


def get_drone_track(drone_data):
    data = []
    # drone: x (1), y (2), z(3), camera: horizontal(4), vertical(5), drone yaw(6), target quotioent (7), target: x (8), y (9), z (10), roll (11), pitch (12), yaw (13)
    # predicted new position: x (14), y (15), z (16), camera: horizontal (17), vertical (18), drone yaw (19), time (20)
    for line in drone_data:
        data.append([float(line[19]), float(line[0]), float(line[1]), float(line[2]), float(line[5])])
    sampled_data = np.array(sample_data(data))
    return sampled_data, np.ceil(data[0][0])


def get_drone_time_or_data(drone_data):
    data = []
    # drone: x (1), y (2), z(3), camera: horizontal(4), vertical(5), drone yaw(6), target quotioent (7), target: x (8), y (9), z (10), roll (11), pitch (12), yaw (13)
    # predicted new position: x (14), y (15), z (16), camera: horizontal (17), vertical (18), drone yaw (19), time (20)
    for line in drone_data:
        data.append([float(line[19]), float(line[5])])
    return data, np.ceil(data[0][0])


def line_type(data):
    if data == 1:
        return "-"
    else:
        return "--"


def read_data_from_file(path_to_file):
    data = []
    with open(path_to_file, "r") as f:
        for line in f:
            data.append(line.split(","))
    return data


def get_prediction_target_track(drone_data):
    data = []
    for line in drone_data:
        data.append([float(line[19]), float(line[7]), float(line[8]), int(line[6])])
    sampled_data = np.array(sample_data(data))
    return sampled_data, np.ceil(data[0][0])


def get_true_target_track(target_data, drone_file_path):
    data = []
    for line in target_data:
        try:
            data.append([float(line[1]), float(line[2]), float(line[3]), 1])
        except Exception as e:
            print("Exception {} occured. Wrong line with values {}, quiting processing of file {}.".format(e, line,
                                                                                                           drone_file_path))
    sampled_data = np.array(sample_data(data))
    return sampled_data, np.ceil(data[0][0])


def sample_data(data):
    start_time = np.ceil(data[0][0])
    sampled_data = []
    i = 0
    while True:
        if i + 1 >= len(data):
            break
        if data[i + 1][0] > start_time >= data[i][0]:
            sample = []
            for l in range(1, len(data[i])):
                interp = interpolate(data[i][0], data[i + 1][0], data[i][l], data[i + 1][l])
                sample.append(interp + data[i][l])
            sampled_data.append(sample)
            # sampled_data.append([x + data[i][1], y + data[i][2], data[i][3]])
            start_time += 1
        else:
            i += 1
    return sampled_data


def interpolate(t0, t1, x0, x1):
    return ((np.ceil(t0) - t0) * (x1 - x0)) / (t1 - t0)


def draw_list_obstacles(world_map):
    obstacles = obstacles_corners(world_map)
    for o in obstacles:
        plt.fill(o[:, 0], o[:, 1], color=(0.2, 0.2, 0.2, 0.3))


def obstacles_corners(world_map):
    corners = copy.copy(world_map.corners)
    iterations = int(len(corners) / 4)
    obstacles = []
    for i in range(iterations):
        obstacle = []
        for j in range(4):
            if i * 4 + j >= len(corners):
                break
            corner = corners[i * 4 + j]
            obstacle.append(corner)
        obstacles.append(obstacle)
    return np.array(obstacles)


def get_data_from_target_prediction(text_file_path, less=True):
    data = {}
    with open(text_file_path, "r") as file:
        for line in file.readlines():
            parts = convert_to_float(line.split(","))
            rounded_time = np.round(parts[0])
            if rounded_time in data:
                if ((parts[1] < data[rounded_time][1]) == less):
                    data[rounded_time] = parts
                    data[rounded_time][0] = parts[0]
            else:
                data[rounded_time] = parts
    return data


def convert_to_float(arr):
    ar = []
    for a in arr:
        try:
            ar.append(float(a))
        except:
            ar.append(a)
    return ar


def fill_unfilled(data, found_time=np.inf, max_time=np.inf):
    keys = data.keys()
    sorted = np.sort(list(keys))
    filled_data = {}
    last_in = int(sorted[0])
    end_time = int(sorted[-1]) if max_time == np.inf else int(max_time)
    finish = False
    for i in range(0, end_time + 1):
        if i not in keys:
            filled_data[i] = data[last_in]
        else:
            filled_data[i] = data[i]
            last_in = i
        if i in keys and i > found_time and data[i][1] < 6.0:
            finish = True
        if i in keys and finish and i > found_time and data[i][1] > 6.0:
            break
    for j in range(i, end_time):
        filled_data[j] = data[last_in]
    return filled_data


def find_seen(data, ignore_until=0.0):
    for k in np.sort(list(data.keys())):
        if float(k) < ignore_until:
            continue
        if data[k][1] == 1:
            return k
    return np.inf


def legend(paths):
    names = []
    for path in paths:
        name = os.path.basename(path)
        name = parse_name(name)
        names.append(name)
    return names


def parse_name(name):
    # 'drone_log_file_100-0_2999--1-f--1-n-10_7_32.json (kopie).txt'
    a = re.findall(".*_([0-9]*)-0_([0-9]*)-.*-[n|f]-.*-[n|y]-(.*).json(.*)", name)
    world_name = "Svět vel: {}, zast: 0.{}, id: {}".format(int(a[0][0]), int(a[0][1]), a[0][2])
    return [world_name, a[0][0], a[0][1], a[0][2], a[0][3]]


def create_time_entropy_plot(data, names, size, ratio, folder_path):
    fig, ax = plt.subplots()
    for i in range(len(data)):
        d = data[i]
        times = []
        values = []
        for k in np.sort(list(d.keys())):
            times.append(k)
            values.append(d[k][1])
        ax.plot(times, values, label=names[i])
    ax.set_xlabel("Čas")
    ax.set_ylabel("Entropie")
    ax.set_title("Mapy o zastavěnosti {}%".format(ratio))
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.2,
                     box.width, box.height * 0.8])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              fancybox=True, shadow=True)
    if not os.path.exists(os.path.join(folder_path, "images", str(size), str(ratio))):
        os.makedirs(os.path.join(folder_path, "images", str(size), str(ratio)))
    plt.savefig(
        "{}/{}.png".format(os.path.join(folder_path, "images", str(size), str(ratio)), "entropy-{}".format("-".join(names))), cmap="gray")
    plt.close()


def find_all_first_seen(paths, ignore_until=0.0):
    first_seens = []
    for path in paths:
        data = get_data_from_target_prediction(path, False)
        first_seens.append(find_seen(data, ignore_until))
    return first_seens


def parse_file_name(name):
    a = re.findall(".*_([0-9]*)-0_([0-9]*)-.*-[n|f]-.*-[n|y]-(.*).json.*", name)
    return a[0][0], a[0][1], a[0][2]


def paths_for_first_seen():
    return paths_for_testing("target_prediction_")


def paths_for_entropy():
    return paths_for_testing("drone_")


def paths_for_testing(starting_word):
    basepath = "/home/patik/.ros/searching_test/"
    groups = {}
    for file in os.listdir(basepath):
        if file.startswith(starting_word):
            path = os.path.join(basepath, file)
            size, ratio, index = parse_file_name(file)
            ratio = int(str(np.round(float("0." + str(ratio)), 1)).split(".")[1]) * 10
            if size not in groups:
                groups[size] = {}
            if ratio not in groups[size]:
                groups[size][ratio] = []
            groups[size][ratio].append(path)
    return groups


def find_index(paths, index, end):
    for i in range(len(paths)):
        path = paths[i]
        if str(path).find(index) != -1 and str(path).find(end) != -1:
            return i
    return -1

def create_subsets(i):
    a = []
    def findsubsets(s, n):
        return list(itertools.combinations(s, n))
    s = set(range(i))
    for j in range(i):
        a.extend(findsubsets(s, j + 1))
    return a

def pick_paths(subset, paths):
    a = []
    for s in subset:
        a.append(paths[s])
    return a

def test_time_entropy_plot():
    first_seen_paths_groups = paths_for_first_seen()
    paths_groups = paths_for_entropy()
    folder_path = "/home/patik/.ros/searching_test"
    for size in paths_groups:
        for ratio in paths_groups[size]:
            first_seen_paths = sorted(first_seen_paths_groups[size][ratio])
            paths = sorted(paths_groups[size][ratio])
            subsets = create_subsets(len(paths))
            for subset in subsets:
                chosen_paths = pick_paths(subset, paths)
                chosen_first_seen_paths = pick_paths(subset, first_seen_paths)
                first_seen = find_all_first_seen(chosen_first_seen_paths, 60.0)
                first_seen_maximum = np.copy(first_seen) + 1
                first_seen_maximum[first_seen_maximum == np.inf] = 0.0
                first_seen_maximum = np.max(first_seen_maximum)
                data_list = []
                legends = legend(chosen_paths)
                names = [leg[0] for leg in legends]
                indices = [leg[3] for leg in legends]
                ends = [leg[4] for leg in legends]
                for i in range(len(indices)):
                    index = indices[i]
                    end = ends[i]
                    j = find_index(chosen_paths, index, end)
                    if j == -1:
                        continue
                    k = find_index(chosen_first_seen_paths, index, end)
                    if k == -1:
                        continue
                    path = chosen_paths[j]
                    data = get_data_from_target_prediction(path)
                    data = fill_unfilled(data, first_seen[k], first_seen_maximum)
                    data_list.append(data)
                create_time_entropy_plot(data_list, names, size, ratio, folder_path)


def divide_to_groups(paths):
    groups = {}
    for path in paths:
        name = os.path.basename(path)
        a = re.findall(".*-([0-9]*)-0_([0-9]*)-.*-[n|f]-.*-[n|y]-.*", name)
        size = a[0][0]
        ratio = int(str(np.round(float("0." + str(a[0][1])), 1)).split(".")[1]) * 10
        if size not in groups:
            groups[size] = {}
        if ratio not in groups[size]:
            groups[size][ratio] = []
        groups[size][ratio].append(path)
    return groups


def map_statistics_paths(folder_path):
    paths = []
    for folder in os.listdir(folder_path):
        folder_join_path = os.path.join(folder_path, folder)
        if os.path.isdir(folder_join_path) and folder.startswith("sworld") and folder.find("-f-") != -1:
            paths.append(folder_join_path)
    return paths


def test_compare_map_statistics():
    world_folder_path = "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/"
    paths = map_statistics_paths(world_folder_path)
    folder_path = "/home/patik/.ros/searching_test"
    groups = divide_to_groups(paths)
    for size in groups:
        for ratio in groups[size]:
            all_statistics, all_name = compare_map_statistics(groups[size][ratio])
            d = section_statistics(groups[size][ratio])
            all_statistics.update(d)
            d = world_statistics(groups[size][ratio])
            all_statistics.update(d)
            s, n = compute_statistics(all_statistics)
            create_statistics_graph(s, n, folder_path, size, ratio)
            create_statistics_error_graph(all_statistics, folder_path, size, ratio)


def section_statistics(paths):
    visibility = []
    for path in paths:
        path = os.path.join(path, "section_file.json")
        d = SectionUtils.unjson_sections(path)
        try:
            objects = d["objects"]
            for i, o in objects.items():
                for j, p in o.objects.items():
                    for k, q in p.objects.items():
                        visibility.append(len(q.objects) * 0.25)
        except Exception as e:
            print(e)
    return {"points_visibility": visibility}


def world_statistics(paths):
    areas = []
    heights = []
    for path in paths:
        files = os.listdir(path)
        for file in files:
            if file.startswith("sworld") and file.endswith(".json"):
                path = os.path.join(path, file)
                area, height = read_world_map(path)
                areas.extend(area)
                heights.extend(height)
    return {"obstacles_areas": areas, "obstacles_heights": heights}


def read_world_map(path):
    areas = []
    heights = []
    with open(path, "r") as file:
        information = json.load(file)
        objects = information["objects"]["statical"]
        for o in objects:
            areas.append(o["x_size"] * o["y_size"])
            heights.append(o["z_size"])
    return areas, heights


def rename_name(name):
    if name == "visibility":
        return "Viditelnost"
    if name == "borders_counts":
        return "Počet přechodů"
    if name == "borders_sums":
        return "Délka přechodů"
    if name == "reduction_effectivity":
        return "Efektivita redukce"
    if name == "points_distances":
        return "Vzdálenost bodů"
    if name == "average_height":
        return "Průměrná výška bodů"
    if name == "clusters_distances":
        return "Vzdálenost regionů"
    if name == "clusters_on_size":
        return "Poměr počtu regionů na velikost"
    if name == "obstacles":
        return "Počet překážek na sekci"
    if name == "areas":
        return "Obsah"
    if name == "obstacles_heights":
        return "Výška budov"
    if name == "points_visibility":
        return "Plocha viditelná z bodů"
    if name == "obstacles_areas":
        return "Obsah půdorysu budov"
    return name


def approriate_metrics(name):
    if name == "visibility":
        return "Podíl viditelných pixelů a jejich celkového počtu"
    if name == "borders_counts":
        return "Jednotky"
    if name == "borders_sums":
        return "Podíl délky přechodů a obvodu sekce"
    if name == "reduction_effectivity":
        return "Podíl množství bodů před a po redukci"
    if name == "points_distances":
        return "m"
    if name == "average_height":
        return "m"
    if name == "clusters_distances":
        return "m"
    if name == "clusters_on_size":
        return "Poměr"
    if name == "obstacles":
        return "Jednotky"
    if name == "areas":
        return "m^2"
    if name == "obstacles_heights":
        return "m"
    if name == "points_visibility":
        return "m^2"
    if name == "obstacles_areas":
        return "m^2"
    return name

def ignore_column_for_given_key(name):
    if name == "visibility":
        return [0,1]
    if name == "borders_counts":
        return []
    if name == "borders_sums":
        return []
    if name == "reduction_effectivity":
        return []
    if name == "points_distances":
        return []
    if name == "average_height":
        return [0,1]
    if name == "clusters_distances":
        return []
    if name == "clusters_on_size":
        return []
    if name == "obstacles":
        return []
    if name == "areas":
        return [2]
    if name == "obstacles_heights":
        return []
    if name == "points_visibility":
        return [0,1]
    if name == "obstacles_areas":
        return [0,1]
    return []

def create_statistics_graph(data, names, folder_path, size, ratio):
    # ["Minimum", "Maximum", "Průměr"]
    for key in data[0]:
        values = []
        chosen_names = []
        ignored = ignore_column_for_given_key(key)
        for j in range(len(data)):
            if j in ignored:
                continue
            values.append(data[j][key])
            chosen_names.append(names[j])
        fig, ax = plt.subplots()
        plt.box(False)
        rects = ax.bar(chosen_names, values, width=((0.8 / len(names)) * len(chosen_names)))
        ax.set_title("{}".format(rename_name(key)), fontsize=18, pad=15.0)
        # ax.set_xlim(0, 0.8)
        ax.set_frame_on(False)
        ax.set_ylabel(approriate_metrics(key), fontsize=16)
        for rect in rects:
            height = np.round(rect.get_height(), 3)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=16)
        if not os.path.exists(os.path.join(folder_path, "images")):
            os.makedirs(os.path.join(folder_path, "images", str(size), str(ratio)))
        plt.savefig(
            "{}/{}.png".format(os.path.join(folder_path, "images", str(size), str(ratio)), str(key)), cmap="gray")
        plt.close()


def graph_groups():
    groups = [
        ["average_height", "obstacles_heights"],
        ["points_visibility", "obstacles_areas"],
        ["borders_counts", "obstacles"]
    ]
    return groups


def create_statistics_error_graph(data, folder_path, size, ratio):
    groups = graph_groups()
    for group in groups:
        means = []
        stds = []
        names = []
        fig, ax = plt.subplots()
        plt.box(False)
        for member in group:
            means.append(np.mean(data[member]))
            stds.append(np.std(data[member]))
            names.append(rename_name(member))
        x_pos = np.arange(0, len(names))
        rects = ax.bar(x_pos, means,
                       yerr=stds,
                       align='center',
                       alpha=0.5,
                       ecolor='black',
                       capsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names)
        ax.set_frame_on(False)
        ax.set_ylabel(approriate_metrics(group[0]), fontsize=16)
        if not os.path.exists(os.path.join(folder_path, "images")):
            os.makedirs(os.path.join(folder_path, "images", str(size), str(ratio)))
        plt.savefig(
            "{}/{}.png".format(os.path.join(folder_path, "images", str(size), str(ratio)), ",".join(group)), cmap="gray")
        plt.close()



def compare_map_statistics(paths):
    mean_statistics = {}
    min_statistics = {}
    max_statistics = {}
    all_data_statistics = {}
    for path in paths:
        path = os.path.join(path, "section_statics.json")
        with open(path, "r") as file:
            d = json.load(file)
            for key in d:
                if key not in mean_statistics:
                    mean_statistics[key] = []
                    min_statistics[key] = []
                    max_statistics[key] = []
                if key not in all_data_statistics:
                    all_data_statistics[key] = []
                value = d[key]
                if isinstance(value, list):
                    all_data_statistics[key].extend(value)
                    mean_statistics[key].append(np.mean(value))
                    min_statistics[key].append(np.min(value))
                    max_statistics[key].append(np.max(value))
                else:
                    all_data_statistics[key].append(value)
                    mean_statistics[key].append(value)
                    max_statistics[key].append(value)
                    min_statistics[key].append(value)
    return all_data_statistics, "all"


def compute_statistics(data):
    min_data = {}
    max_data = {}
    mean_data = {}
    for d in data:
        min_data[d] = np.min(data[d])
        max_data[d] = np.max(data[d])
        mean_data[d] = np.mean(data[d])
    return [min_data, max_data, mean_data], ["Minimum", "Maximum", "Průměr"]


test_compare_map_statistics()
test_time_entropy_plot()
exit(32)

# texture_join_files()
json_file = "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/sworld-100-0_0996--1-n--1-n-123456789/sworld-100-0_0996--1-n--1-n-123456789.json"
test_folder = "/home/patik/.ros/tracking_test"
result_file = test_folder + "/final_result.csv"
image_path = test_folder + "/image.png"
data = draw_target_tracks()
show_and_save_statistics(data, result_file, test_folder)
