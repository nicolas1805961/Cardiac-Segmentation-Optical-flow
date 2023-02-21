from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):

    def __init__(self, optimizer, warmup_percent, after_scheduler):
        super(WarmupScheduler, self).__init__(optimizer)
        self.warmup_percent = warmup_percent
        self.after_scheduler = after_scheduler
        self.total_nb_iterations = after_scheduler.state_dict()['T_max']
        self.warmup_iter = int(self.warmup_percent * self.total_nb_iterations)
        self.current_iteration = 0
        self.add = after_scheduler.state_dict()['base_lrs'][0] / self.warmup_iter

    def get_lr(self):
        self.after_scheduler.base_lrs = [base_lr + self.add for base_lr in self.base_lrs]

    def step(self):
        if self.current_iteration < self.warmup_iter:
            return super(WarmupScheduler, self).step()
        else:
            self.after_scheduler.step()
            self._last_lr = self.after_scheduler.get_last_lr()
        