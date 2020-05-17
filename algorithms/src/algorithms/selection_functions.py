from abc import abstractmethod
import numpy as np

class SelectionFunction(object):
    def __init__(self):
        super(SelectionFunction, self).__init__()

    @abstractmethod
    def select(self, objects, properties, scored_objects):
        pass

class BigDrop(SelectionFunction):

    def select(self, objects, properties, scored_objects):
        print("Big drop")
        sorted = np.sort(scored_objects)
        diffs = sorted - np.roll(sorted, -1)
        maximum = np.max(diffs)
        median = np.median(diffs)
        if maximum > 3 * median:
            index = np.argmin(diffs > 3 * median)
            value = sorted[index]
            return np.array([a[0] for a in np.argwhere(scored_objects >= value)])
        else:
            return np.arange(0, len(scored_objects))


class AboveMinimum(SelectionFunction):
    def __init__(self, normalized_minimum):
        super(AboveMinimum, self).__init__()
        self.normalized_minimum = normalized_minimum

    def select(self, objects, properties, scored_objects):
        print("Above minimum")
        s = np.sum(scored_objects)
        sc = np.copy(scored_objects)
        sc /= s
        return np.array([a[0] for a in np.argwhere(sc > self.normalized_minimum)])


class AboveAverage(SelectionFunction):

    def select(self, objects, properties, scored_objects):
        print("Above average")
        average = np.average(scored_objects)
        return np.array([a[0] for a in np.argwhere(scored_objects > average)])


class NElements(SelectionFunction):

    def __init__(self, n):
        super(NElements, self).__init__()
        self.n = n

    def select(self, objects, properties, scored_objects):
        print("N element")
        sorted = np.sort(scored_objects)
        value = sorted[self.n]
        return np.array([a[0] for a in np.argwhere(scored_objects > value)])


class RatioElements(SelectionFunction):

    def __init__(self, ratio):
        super(RatioElements, self).__init__()
        self.ratio = ratio

    def select(self, objects, properties, scored_objects):
        print("Ratio element")
        sorted = np.sort(scored_objects)
        value = sorted[int(self.ratio * len(scored_objects))]
        return np.array([a[0] for a in np.argwhere(scored_objects > value)])
