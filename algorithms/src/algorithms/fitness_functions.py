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
        scores = [object.score for object in objects]
        median = np.median(scores)
        fitness = np.empty((len(objects)))
        for index in range(len(objects)):
            object = objects[index]
            fitness[index] = (abs(object.score - median))
        fitness = np.max(fitness) - fitness
        return fitness


class Maximum(FitnessFunction):

    def score(self, objects, properties):
        return np.array([object.score for object in objects])


class Neighbour(FitnessFunction):

    def score(self, objects, properties):
        fitness = np.empty((len(objects)))
        for index in range(len(objects)):
            object = objects[index]
            score = object.score
            for n in object.neighbors:
                score += objects[n].score
            fitness[index] = score
        return fitness


class DistanceScore(FitnessFunction):

    def score(self, objects, properties):
        fitness = np.empty((len(objects)))
        scores = np.array([object.score for object in objects])
        for index in range(len(objects)):
            fitness[index] = np.sum(scores / np.clip(properties[index, :], 1, None))
        return fitness