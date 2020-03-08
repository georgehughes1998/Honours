from math import exp


class LearningRate:
    def __init__(self, initial_lr, epoch, creep_rate=0.25):
        self.creep_rate = creep_rate

        self.initial_lr = initial_lr
        self.last_saved = epoch
        self.epoch = epoch

        self.counter = 0
        self.running_counter = 2

    def model_was_saved(self, epoch):
        self.last_saved = epoch

    def get_learning_rate(self, epoch):
        self.epoch = epoch
        self.counter = epoch - self.last_saved

        if self.counter == 0:
            self.running_counter -= 1
        else:
            self.running_counter += 1 + self.creep_rate

        learning_rate = self.initial_lr * exp(-(epoch - self.running_counter)/4)
        return learning_rate
