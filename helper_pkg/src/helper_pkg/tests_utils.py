import numpy as np
import matplotlib.pyplot as plt
import yaml
from pesat_utils.pesat_math import Math
from pesat_utils.base_map import BaseMap
import copy
from scipy import stats
import os


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
            "{}/{}.png".format(os.path.join(folder_path,  "images"), str(index)))
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


# texture_join_files()
json_file = "/home/patik/Diplomka/dp_ws/src/pesat_resources/worlds/sworld-100-0_0996--1-n--1-n-123456789/sworld-100-0_0996--1-n--1-n-123456789.json"
test_folder = "/home/patik/.ros/tracking_test"
result_file = test_folder + "/final_result.csv"
image_path = test_folder + "/image.png"
data = draw_target_tracks()
show_and_save_statistics(data, result_file, test_folder)
