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
        sorted = np.sort(scored_objects)[::-1]
        print("sorted", sorted)
        diffs = np.abs(sorted - np.roll(sorted, -1))[:-1]
        print("Diffs", diffs)
	if len(diffs) == 0:
	    return [0]
        maximum = np.max(diffs)
        mean = np.mean(diffs)
        print("max, mean", maximum, mean)
        if maximum > mean:
            i = 1
            while True:
                if np.count_nonzero(diffs > i * mean) <= 1:
                    break
                i += 1
            index = np.clip(np.argmax(diffs > i * mean), 0, None)
            value = sorted[index]
            print("i, index, value", i, index, value)
            return np.array([a[0] for a in np.argwhere(scored_objects >= value)])
        else:
            return np.array([a[0] for a in np.argwhere(scored_objects > 0)])


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
