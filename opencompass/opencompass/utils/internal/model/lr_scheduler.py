

"""
测试一下一些 lr scheduler 吧 
"""

from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR, LinearWarmupLR


class CutomizedCosineAnnealingWarmupLR(CosineAnnealingWarmupLR):
    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                # 这个 True 的切换是为了避开warmup到达预设的值切换的时候爆出warning
                self.after_scheduler._get_lr_called_within_step = True
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        return [(self.last_epoch + 1) / self.warmup_epochs * lr for lr in self.base_lrs]
    
class FineTuneCosineAnnealingWarmupLR(CosineAnnealingWarmupLR):
    def __init__(self, optimizer, total_steps: int, init_steps:int = 0, warmup_steps: int = 0, eta_min: float = 0., last_epoch: int = -1):
        self._init_steps = init_steps
        self._warmup_steps = warmup_steps # 使用这个值计算warmup的lr, 因为 warmup_epochs = init_steps + warmup_steps
        super().__init__(optimizer, total_steps, warmup_steps + init_steps, eta_min, last_epoch)
        
    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:
                # 这个 True 的切换是为了避开warmup到达预设的值切换的时候爆出warning
                self.after_scheduler._get_lr_called_within_step = True
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()

        elif self.last_epoch >= self._init_steps:
            return [(self.last_epoch + 1 - self._init_steps) / self._warmup_steps * lr for lr in self.base_lrs]
        else:
            return [0 for lr in self.base_lrs]

def get_init_warmup_decay_ratio(step, init_steps, warmup_steps, total_steps):
    if step < init_steps:
        return 0
    if step > warmup_steps + init_steps:
        ratio = (step - warmup_steps - init_steps) / (total_steps - warmup_steps - init_steps)
        return 1 - ratio
    else:
        ratio = (step - init_steps) / warmup_steps
        return ratio

class ResetFineTuneCosineAnnealingWarmupLR(CosineAnnealingWarmupLR):
    def __init__(self, optimizer, total_steps: int, init_steps:int = 0, warmup_steps: int = 0, mask_scale_intervals=[], eta_min: float = 0., last_epoch: int = -1):
        self._init_steps = init_steps
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps # 使用这个值计算warmup的lr, 因为 warmup_epochs = init_steps + warmup_steps
        self.mask_scale_intervals = [0] + mask_scale_intervals
        # self.mask_reset_steps = mask_reset_steps
        self._cur_scale_interval = 0
        self._base_lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, total_steps, self.mask_scale_intervals[-1] + self._init_steps, warmup_steps, eta_min, last_epoch)
        super().__init__(optimizer, total_steps, warmup_steps + init_steps, eta_min, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self._init_steps:
            return [0 for lr in self.base_lrs]
        for i, interval in enumerate(self.mask_scale_intervals):
            if self.last_epoch < interval:
                step = self.last_epoch - self.mask_scale_intervals[i-1]
                ratio = get_init_warmup_decay_ratio(step, self._init_steps, self._warmup_steps, self._total_steps)
                return [lr * ratio for lr in self.base_lrs]
        self._base_lr_scheduler.step(self.last_epoch)
        return self._base_lr_scheduler.get_lr()
    
    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
        del state_dict['after_scheduler']
        state_dict['_base_lr_scheduler_dict'] = state_dict['_base_lr_scheduler'].state_dict()
        del state_dict['_base_lr_scheduler']
        return state_dict
    
    def load_state_dict(self, state_dict):
        for key in list(self.__dict__.keys()):
            if key in state_dict:
               self.__dict__[key] = state_dict[key]
        self._base_lr_scheduler.load_state_dict(state_dict['_base_lr_scheduler_dict'])
        return state_dict

if __name__ == '__main__':
    from torch import optim
    from torch import nn
    fc = nn.Linear(3, 3)
    optimizer = optim.Adam(fc.parameters(), lr=1)
    # import matplotlib.pyplot as plt

    total_steps = 20000
    ckpt_step = 1000000

    # lr_scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=total_steps, warmup_steps = 100, eta_min = 1e-6, last_epoch = -1)
    # lrs = []
    # for i in range(total_steps):
    #     lrs.append(lr_scheduler.get_lr()[0])
    #     lr_scheduler.step()
    #     if i == ckpt_step:
    #         states = lr_scheduler.state_dict()
    #         optim_states = optimizer.state_dict()
    # lrs.append(lr_scheduler.get_lr()[0])

    # lr_scheduler.load_state_dict(states)
    # optimizer.load_state_dict(optim_states)
    # for i in range(ckpt_step+1, total_steps):
    #     lrs.append(lr_scheduler.get_lr()[0])
    #     lr_scheduler.step()
    # lrs.append(lr_scheduler.get_lr()[0])
    # print(lrs)

    # lr_scheduler = CosineAnnealingWarmupLR(optimizer, total_steps=total_steps, warmup_steps = 5, eta_min = 1e-6, last_epoch = -1)
    # lrs = []
    # for i in range(total_steps):
    #     lrs.append(lr_scheduler.get_lr()[0])
    #     lr_scheduler.step()
    # lrs.append(lr_scheduler.get_lr()[0])
    # print(lrs)
    
    # lr_scheduler = FineTuneCosineAnnealingWarmupLR(optimizer, total_steps=total_steps, init_steps=10, warmup_steps = 20, eta_min = 1e-6, last_epoch = -1)
    lr_scheduler = ResetFineTuneCosineAnnealingWarmupLR(
        optimizer, total_steps=total_steps, init_steps=100, warmup_steps = 1000, 
        mask_scale_intervals=[1000, 2000, 3000, 4000],
         eta_min = 1e-6, last_epoch = -1)
    lrs = []
    for i in range(998):
        lrs.append(round(lr_scheduler.get_lr()[0], 2))
        lr_scheduler.step()
    # lrs.append(round(lr_scheduler.get_lr()[0], 2))
    print(lrs)
    print('cur lr', lr_scheduler.get_lr())
    state = lr_scheduler.state_dict()
    print(state)
    # lr_scheduler.load_state_dict()
    lr_scheduler = ResetFineTuneCosineAnnealingWarmupLR(
        optimizer, total_steps=total_steps, init_steps=100, warmup_steps = 1000, 
        mask_scale_intervals=[1000, 2000, 3000, 4000],
         eta_min = 1e-6, last_epoch = -1)
    lr_scheduler.load_state_dict(state)
    print(lr_scheduler.get_lr())
    # print(list(enumerate(lrs)))
    # plt.plot(range(total_steps), lrs)
    # plt.savefig('./lr_steps.png')
