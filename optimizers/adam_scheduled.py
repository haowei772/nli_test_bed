import torch.optim as O
import numpy as np

class AdamScheduled:
    def __init__(self, config, model):
        self.lr = config.lr
        self.parameters = filter(lambda p: p.requires_grad, model.parameters())
        self._optimizer = O.Adam(self.parameters, self.lr)
        self.n_warmup_steps = config.n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(config.d_embed, -0.5)
    
    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr