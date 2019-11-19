from abc import abstractmethod


class ReplaningFunction(object):
    def __init__(self):
        super(ReplaningFunction, self).__init__()

    def get_neighbours(self, objects, chosen_objects):
        new_objects = set([])
        for object_id in chosen_objects:
            new_objects.add(object_id)
            object = objects[object_id]
            for n in object.neighbors:
                new_objects.add(n)
        return objects

    def get_all(self, objects):
        ids = [o.id for o in objects]
        return ids

    @abstractmethod
    def replaning(self, objects, chosen_objects, properties, iteration):
        pass


class NAllReplaning(ReplaningFunction):
    def __init__(self, n):
        super(NAllReplaning, self).__init__()
        self.n = n

    def replaning(self, objects, chosen_objects, properties, iteration):
        if iteration % self.n:
            return True, None
        return False, None


class NNeighboursReplaning(ReplaningFunction):
    def __init__(self, n):
        super(NNeighboursReplaning, self).__init__()
        self.n = n

    def replaning(self, objects, chosen_objects, properties, iteration):
        if iteration % self.n:
            return True, self.get_neighbours(objects, chosen_objects)
        return False, None


class NNeighboursMAllReplaning(ReplaningFunction):
    def __init__(self, n, m):
        super(NNeighboursMAllReplaning, self).__init__()
        self.n = n
        self.m = m

    def replaning(self, objects, chosen_objects, properties, iteration):
        if iteration % self.n:
            return True, self.get_neighbours(objects, chosen_objects)
        if iteration % self.m:
            return True, None
        return False, None


class NoReplaning(ReplaningFunction):

    def __init__(self):
        super(NoReplaning, self).__init__()

    def replaning(self, objects, chosen_objects, properties, iteration):
        return False, None
