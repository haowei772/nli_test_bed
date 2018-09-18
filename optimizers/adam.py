import torch.optim as O

class Adam:
    def __init__(self, config, model):
        self.lr = config.lr
        self.parameters = filter(lambda p: p.requires_grad, model.parameters())
        self.optimizer = O.Adam(self.parameters, self.lr)
        self._step = 0

    def zero_grad():
        self.optimizer.zero_grad()
        
    def step(self):
        self.optimizer.step()