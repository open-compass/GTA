from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR


class FineTuneCosineAnnealingWarmupLR(CosineAnnealingWarmupLR):
    """
    FineTune Cosine Annealing Warmup LR.

    Args:
        optimizer: The optimizer object.
        total_steps (int): The number of total steps.
        init_steps (int): The number of init steps, default is 0.
        warmup_steps (int): The number of warm up steps, default is 0.
        eta_min (float): The minimum learning rate, default is 0.0.
        last_epoch: Last epoch, default is -1.

    """

    def __init__(
        self,
        optimizer,
        total_steps: int,
        init_steps: int = 0,
        warmup_steps: int = 0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self._init_steps = init_steps
        self._warmup_steps = warmup_steps
        # Use this value to calculate the lr of warmup, because warmup_epochs = init_steps + warmup_steps
        super().__init__(optimizer, total_steps, warmup_steps + init_steps, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.warmup_epochs:
            if not self.finished:  # pylint: disable=E0203
                # This True switch is to avoid warning when the warmup reaches the preset value switch
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
    """
    Reset FineTune Cosine Annealing Warmup LR.

    Args:
        optimizer: The optimizer object.
        total_steps (int): The number of total steps.
        init_steps (int): The number of init steps, default is 0.
        warmup_steps (int): The number of warm up steps, default is 0.
        mask_scale_intervals: The list of mask scale intervals.
        eta_min (float): The minimum learning rate, default is 0.0.
        last_epoch: Last epoch, default is -1.

    """

    def __init__(  # pylint: disable=W0102
        self,
        optimizer,
        total_steps: int,
        init_steps: int = 0,
        warmup_steps: int = 0,
        mask_scale_intervals=[],
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ):
        self._init_steps = init_steps
        self._total_steps = total_steps
        self._warmup_steps = warmup_steps
        # Use this value to calculate the lr of warmup, because warmup_epochs = init_steps + warmup_steps
        self.mask_scale_intervals = [0] + mask_scale_intervals
        self._cur_scale_interval = 0
        self._base_lr_scheduler = FineTuneCosineAnnealingWarmupLR(
            optimizer, total_steps, self.mask_scale_intervals[-1] + self._init_steps, warmup_steps, eta_min, last_epoch
        )
        super().__init__(optimizer, total_steps, warmup_steps + init_steps, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch < self._init_steps:
            return [0 for lr in self.base_lrs]
        for i, interval in enumerate(self.mask_scale_intervals):
            if self.last_epoch < interval:
                step = self.last_epoch - self.mask_scale_intervals[i - 1]
                ratio = get_init_warmup_decay_ratio(step, self._init_steps, self._warmup_steps, self._total_steps)
                return [lr * ratio for lr in self.base_lrs]
        self._base_lr_scheduler.step(self.last_epoch)
        return self._base_lr_scheduler.get_lr()

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key != "optimizer"}
        del state_dict["after_scheduler"]
        state_dict["_base_lr_scheduler_dict"] = state_dict["_base_lr_scheduler"].state_dict()
        del state_dict["_base_lr_scheduler"]
        return state_dict

    def load_state_dict(self, state_dict):
        for key in list(self.__dict__.keys()):
            if key in state_dict:
                self.__dict__[key] = state_dict[key]
        self._base_lr_scheduler.load_state_dict(state_dict["_base_lr_scheduler_dict"])
        return state_dict
