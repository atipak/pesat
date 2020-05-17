import numpy as np
import time
import cv2
import matplotlib.pyplot as plt


class Plotting:
    import time
    axs = {}

    @staticmethod
    def plot_hist(hist, map, title="Unnamed"):
        hist = np.nan_to_num(hist)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(hist, cmap=plt.cm.Reds, interpolation='none',
                  extent=[-map.real_width / 2.0, map.real_width / 2.0, -map.real_height / 2.0, map.real_height / 2.0])
        ax.set_aspect(2)  # you may also use am.imshow(..., aspect="auto") to restore the aspect ratio
        ax.set_title(title)
        plt.show()
        plt.close()

    @staticmethod
    def plot_points(points, map, title="Unnamed", blocking=True):
        if blocking:
            fig, ax = plt.subplots()
        else:
            if title in Plotting.axs:
                ax = Plotting.axs[title]
            else:
                fig, ax = plt.subplots()
                Plotting.axs[title] = ax
                plt.show(block=blocking)
        ax.clear()
        ax.scatter(points[:, 0], points[:, 1], s=20, marker='o', c="b")
        ax.set_ylim(-map.real_width / 2.0, map.real_width / 2.0)
        ax.set_xlim(-map.real_height / 2.0, map.real_height / 2.0)
        plt.draw()
        plt.pause(0.001)
        if blocking:
            plt.show(block=blocking)
            time.sleep(1)
            plt.close()

    @staticmethod
    def plot_probability(probability, name="pb_map.png"):
        print(np.count_nonzero(probability))
        m = np.zeros(probability.shape, np.uint8)
        avg = np.average(probability[probability > 0])
        m[probability > 0] = 155
        m[probability > avg] = 255
        cv2.imwrite(name, m)

    @staticmethod
    def plot_map(coordinates_xy, mapa):
        m = np.zeros(mapa.shape, np.uint8)
        m[coordinates_xy[:, 1], coordinates_xy[:, 0]] = 255
        cv2.imwrite("coor_map.png", m)
