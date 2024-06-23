import math
import torch


def cosine_mask_weight_v1(cur_iter, max_iter):
    # cur_iter: [1, max_iter]
    lr = (1 - math.cos(min(cur_iter, max_iter) / max_iter * math.pi * 0.5))
    return lr

def cosine_mask_weight_v2(cur_iter, max_iter):
    # cur_iter: [1, max_iter]
    lr = (1 - math.cos(min(cur_iter, max_iter) / max_iter * math.pi)) * 0.5
    return lr

def cosine_mask_weight_v3(cur_iter, max_iter):
    # cur_iter: [1, max_iter]
    lr = ((1 - math.cos(min(cur_iter, max_iter) / max_iter * math.pi)) * 0.5) ** 2
    return lr

def generate_scaleup_mask(cur_iter, warumup_iter, max_iter, ori_ch, scaleup_ch, max_ch, mode='v1', device='cuda'):
    '''
    cur_iter (int): [1, max_iter]
    ori_ch (int): channel of original model
    scaleup_ch (int): channel to scale up
    max_ch (int): max channel
    mode (str): cosine or sine
    '''
    assert warumup_iter <= max_iter
    if mode == 'v1':
        cur_weight = cosine_mask_weight_v1(max(cur_iter - warumup_iter, 0), max_iter - warumup_iter)
    elif mode == 'v2':
        cur_weight = cosine_mask_weight_v2(max(cur_iter - warumup_iter, 0), max_iter - warumup_iter)
    elif mode == 'v3':
        cur_weight = cosine_mask_weight_v3(max(cur_iter - warumup_iter, 0), max_iter - warumup_iter)
    else:
        raise NameError
    assert cur_weight >= 0 and cur_weight <= 1

    assert ori_ch <= max_ch
    assert scaleup_ch <= max_ch
    assert ori_ch <= scaleup_ch
    mask = torch.zeros((1, 1, max_ch), dtype=torch.float, device=device)
    mask[:, :, :ori_ch] = 1.0
    mask[:, :, ori_ch:scaleup_ch] = cur_weight

    # mask = mask.to(device)
    return mask, cur_weight

def get_hidden_dim(dim):
    multiple_of = 256
    hidden_dim = int(8 * dim / 3)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim

class ScaleMaskGenerator:
    def __init__(self, mode, warmup_iter, max_iter, ori_dim, new_dim, ori_layers, new_layers, scaleup_steps=None) -> None:
        self.mode = mode
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.ori_dim = ori_dim
        self.ori_ffn_dim = get_hidden_dim(ori_dim)
        self.new_dim = new_dim
        self.new_ffn_dim = get_hidden_dim(new_dim)
        self.ori_layers = ori_layers
        self.new_layers = new_layers
        self.scaleup_steps = scaleup_steps
        self._cur_iter = 0
        if scaleup_steps is not None:
            self.feat_mask_gen = MaskDiscreteGenerator_internal(self.warmup_iter, self.max_iter, self.ori_dim, self.new_dim, self.new_dim, self.scaleup_steps)
            self.ffn_mask_gen = MaskDiscreteGenerator_internal(self.warmup_iter, self.max_iter, self.ori_ffn_dim, self.new_ffn_dim, self.new_ffn_dim, self.scaleup_steps)
            self.layer_mask_gen = MaskDiscreteGenerator_internal(self.warmup_iter, self.max_iter, self.ori_layers, self.new_layers, self.new_layers, self.scaleup_steps)
        
    def get_feat_mask(self, cur_iter):
        if self.scaleup_steps is not None:
            mask = self.feat_mask_gen.forward(cur_iter)
            return mask, mask.sum().item()
        return generate_scaleup_mask(
            cur_iter=cur_iter,
            warumup_iter=self.warmup_iter,
            max_iter=self.max_iter,
            ori_ch=self.ori_dim,
            max_ch=self.new_dim,
            scaleup_ch=self.new_dim,
            mode=self.mode
        )
    
    def get_ffn_mask(self, cur_iter):
        if self.scaleup_steps is not None:
            mask = self.ffn_mask_gen.forward(cur_iter)
            return mask, mask.sum().item()
        return generate_scaleup_mask(
            cur_iter=cur_iter,
            warumup_iter=self.warmup_iter,
            max_iter=self.max_iter,
            ori_ch=self.ori_ffn_dim,
            max_ch=self.new_ffn_dim,
            scaleup_ch=self.new_ffn_dim,
            mode=self.mode
        )
        
    def get_layer_mask(self, cur_iter):
        if self.scaleup_steps is not None:
            mask = self.layer_mask_gen.forward(cur_iter)
            return mask.view(1, -1)
        mask = generate_scaleup_mask(
            cur_iter=cur_iter,
            warumup_iter=self.warmup_iter,
            max_iter=self.max_iter,
            ori_ch=self.ori_layers,
            max_ch=self.new_layers,
            scaleup_ch=self.new_layers,
            mode=self.mode
        )[0]
        return mask.view(1, -1)
    
    def get_scaleup_intervals(self):
        if self.scaleup_steps is None:
            return None
        interval = (float)(self.max_iter - self.warmup_iter) // self.scaleup_steps
        results = []
        for i in range(self.scaleup_steps):
            results.append(int(i * interval) + self.warmup_iter)
        return results

class MaskDiscreteGenerator_internal():
    def __init__(self, warmup_iter:int, max_iter:int, ori_ch:int, scaleup_ch:int, max_ch:int, scaleup_steps=1, device='cuda'):
        super().__init__()
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.ori_ch = ori_ch
        self.scaleup_ch = scaleup_ch
        self.max_ch = max_ch
        self.scaleup_steps = scaleup_steps
        self.device = device


        # assert ori_ch % group_ch == 0
        # assert scaleup_ch % group_ch == 0
        # assert max_ch % group_ch == 0

        assert warmup_iter <= max_iter
        assert ori_ch <= max_ch
        assert scaleup_ch <= max_ch
        assert ori_ch <= scaleup_ch
        assert (self.max_iter - self.warmup_iter) >= self.scaleup_steps
        
        self.group_ch = math.ceil((self.scaleup_ch - self.ori_ch) / self.scaleup_steps)
        self.scaleup_interval = (float)(self.max_iter - self.warmup_iter) // self.scaleup_steps
        
        # print(self.scaleup_steps, self.scaleup_interval, self.group_ch, self.ori_ch, self.max_ch)

    def forward(self, cur_iter):
        
        if cur_iter < self.warmup_iter:
            cur_ch = self.ori_ch
        else:
            scaleup_num = math.ceil((cur_iter - self.warmup_iter) / self.scaleup_interval)
            cur_ch = min(self.scaleup_ch, self.ori_ch + scaleup_num * self.group_ch)
            # print(scaleup_num)
        mask = torch.zeros((1, 1, self.max_ch), dtype=torch.float, device=self.device)
        mask[..., :cur_ch] = 1.0

        # mask = mask.to(self.device)

        return mask

class MaskDiscreteGenerator:
    def __init__(self, warmup_iter:int, max_iter:int, ori_ch:int, scaleup_ch:int, max_ch:int):
        self.warmup_iter = warmup_iter
        self.max_iter = max_iter
        self.ori_ch = ori_ch
        self.scaleup_ch = scaleup_ch
        self.max_ch = max_ch

        assert warmup_iter <= max_iter
        assert ori_ch <= max_ch
        assert scaleup_ch <= max_ch
        assert ori_ch <= scaleup_ch

    def get_feat_mask(self, cur_iter):
        scaleup_interval = (float)(self.max_iter - self.warmup_iter) // (self.scaleup_ch - self.ori_ch)

        if cur_iter < self.warmup_iter:
            cur_ch = self.ori_ch
        else:
            scaleup_num = int((cur_iter - self.warmup_iter) // scaleup_interval)
            cur_ch = min(self.scaleup_ch, self.ori_ch + scaleup_num)
        mask = torch.zeros((1, 1, self.max_ch), dtype=torch.float)
        mask[:, :, :cur_ch] = 1.0

        return mask, cur_ch/self.max_ch

    def get_ffn_mask(self, cur_iter):
        ori_ffn_dim = get_hidden_dim(self.ori_ch)
        new_ffn_dim = get_hidden_dim(self.max_ch)
        scaleup_interval = (float)(self.max_iter - self.warmup_iter) // (new_ffn_dim - ori_ffn_dim)

        if cur_iter < self.warmup_iter:
            cur_ch = ori_ffn_dim
        else:
            scaleup_num = int((cur_iter - self.warmup_iter) // scaleup_interval)
            cur_ch = min(self.scaleup_ch, ori_ffn_dim + scaleup_num)
        mask = torch.zeros((1, 1, new_ffn_dim), dtype=torch.float)
        mask[:, :, :cur_ch] = 1.0

        return mask, cur_ch/new_ffn_dim


def main():

    max_iter = 1000
    warmup_iter = 200
    ori_ch = 10
    scaleup_ch = 25
    max_ch = 30
    iters_list = []
    weight_v1_list = []
    weight_v2_list = []
    weight_v3_list = []
    for i in range(max_iter + 500):
        cur_iter = i + 1
        iters_list.append(cur_iter)

        mask_v1, weight_v1 = generate_scaleup_mask(cur_iter, warmup_iter, max_iter, ori_ch, scaleup_ch, max_ch, mode='v1', device='cuda')
        assert mask_v1.size() == (1, 1, 30)
        weight_v1_list.append(weight_v1)

        mask_v2, weight_v2 = generate_scaleup_mask(cur_iter, warmup_iter, max_iter, ori_ch, scaleup_ch, max_ch, mode='v2', device='cuda')
        weight_v2_list.append(weight_v2)

        mask_v3, weight_v3 = generate_scaleup_mask(cur_iter, warmup_iter, max_iter, ori_ch, scaleup_ch, max_ch, mode='v3', device='cuda')
        weight_v3_list.append(weight_v3)

    plt.plot(iters_list, weight_v1_list, label='v1')
    plt.plot(iters_list, weight_v2_list, label='v2')
    plt.plot(iters_list, weight_v3_list, label='v3')
    plt.legend(['v1', 'v2', 'v3'])
    plt.savefig('mask_weight_cos.png')
    plt.close()


if __name__ == '__main__':
    # import matplotlib
    # import matplotlib.pyplot as plt
    # main()
    ws = []
    mg = ScaleMaskGenerator('v3', 1000, 5000, 4096, 5120, 32, 40, scaleup_steps=4)
    for i in range(5000):
        _, w = mg.get_feat_mask(i)
        ws.append(w)
    # print(ws)
    print(mg.get_scaleup_intervals())
    