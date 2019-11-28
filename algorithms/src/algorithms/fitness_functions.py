from abc import abstractmethod
import numpy as np


class FitnessFunction(object):
    def __init__(self):
        super(FitnessFunction, self).__init__()

    @abstractmethod
    def score(self, objects, properties):
        pass

class Median(FitnessFunction):

    def score(self, objects, properties):
        ids = np.sort(np.array([i for (i, object) in objects.items()]))
        scores = [objects[i].score for i in ids]
        median = np.median(scores)
        fitness = np.empty((len(objects)))
        i = 0
        for index in ids:
            object = objects[index]
            fitness[i] = (abs(object.score - median))
            i += 1
        fitness = np.max(fitness) - fitness
        return fitness


class Maximum(FitnessFunction):

    def score(self, objects, properties):
        ids = np.sort(np.array([i for (i, object) in objects.items()]))
        return np.array([objects[i].score for i in ids])


class Neighbour(FitnessFunction):

    def score(self, objects, properties):
        ids = np.sort(np.array([i for (i, object) in objects.items()]))
        fitness = np.empty((len(objects)))
        i = 0
        for index in ids:
            object = objects[index]
            score = object.score
            for n in object.neighbors:
                if n in objects:
                    score += objects[n].score
            fitness[i] = score
            i += 1
        return fitness


class DistanceScore(FitnessFunction):

    def score(self, objects, properties):
        fitness = np.empty((len(objects)))
        distances = properties["distances"]
        ids = np.sort(np.array([i for (i, object) in objects.items()]))
        scores = np.array([objects[i].score for i in ids])
        for index in range(len(objects)):
            fitness[index] = np.sum(scores / np.clip(distances[index, :], 1, None))
        return fitness