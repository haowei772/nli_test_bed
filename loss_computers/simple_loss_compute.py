
class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, config, criterion, opt=None):
        self.config = config
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, out, batch):
        loss = self.criterion(out, batch.label)
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss