import abc


class DataLoader:
    def __iter__(self):
        return self

    def __next__(self):
        while self.stop_iteration():
            return self.load_minibatch()
        raise StopIteration

    @abc.abstractmethod
    def stop_iteration(self):
        """
        Condition called every iterations (if false, will stop iteration)
        :return:
        """

    @abc.abstractmethod
    def load_minibatch(self):
        '''
        implement this part to return images
        '''
