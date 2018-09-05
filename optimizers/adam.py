import torch.optim as O

class Adam:
    def __init__(self, config, model):
        self.lr = config.lr
        self.optimizer = O.Adam(model.parameters(), self.lr)
        self._step = 0

    def step(self):
        self.optimizer.step()