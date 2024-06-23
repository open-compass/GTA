from colossalai.nn.layer.base_layer import ParallelLayer
from colossalai.nn.layer.parallel_1d import VocabParallelEmbedding1D, VocabParallelClassifier1D, Linear1D_Col, Linear1D_Row
from colossalai.nn.layer.colossalai_layer._utils import ColossalaiModule
from colossalai.nn.layer.colossalai_layer import Dropout
from colossalai.core import global_context as gpc
from colossalai.context.parallel_mode import ParallelMode
from typing import Union, Dict, Tuple, List
import re
from colossalai.logging import disable_existing_loggers, get_dist_logger
import json
import torch
import os
import io 
from collections import defaultdict
import copy
import time
import socket
from .storage_manager import get_storage_manager
from functools import partial
from tqdm import tqdm
from opencompass.utils.internal.model2.converted_llama.import_helper import try_import_petrel_client

Client = try_import_petrel_client()

def get_dp_size():
    """
    获取data parallel的数量，没有的话就是1
    """
    try:
        return gpc.get_world_size(ParallelMode.DATA)
    except KeyError:
        return 1

def get_tp_size():
    """
    获取tensor parallel的数量，没有的话就是1
    """
    try:
        return gpc.get_world_size(ParallelMode.TENSOR)
    except KeyError:
        return 1

def get_pp_size():
    """
    获取pipeline parallel的数量，没有的话就是1
    """
    try:
        return gpc.get_world_size(ParallelMode.PIPELINE)
    except KeyError:
        return 1


def get_dp_rank():
    try:
        return gpc.get_local_rank(ParallelMode.DATA)
    except KeyError:
        return 0


def get_tp_rank():
    try:
        return gpc.get_local_rank(ParallelMode.TENSOR)
    except KeyError:
        return 0


def get_pp_rank():
    try:
        return gpc.get_local_rank(ParallelMode.PIPELINE)
    except KeyError:
        return 0

def _check_folder(folder):
    get_storage_manager().check_folder(folder)

def _save_states(step, fp, states):
    get_storage_manager().save(step, fp, states)

def _set_finish_flag(step):
    get_storage_manager().set_finish_flag(step)

def _get_fns(folder):
    return get_storage_manager().get_fns(folder)


def _load_states(fp):
    return get_storage_manager().load(fp)


def wait_upload_checkpoint():
    get_storage_manager().wait()


def get_model_topology(model):
    """
    返回的内容为
    {
        '{name}': {'dim': int} 
    }
    其中 name 是 module 的名称，在这个 module 下面的所有参数都通过 dim 这个维度进行拼接

    """
    from model.head import VocabParallelClassifier1D as CVocabParallelClassifier1D
    from model.packed_fat.packed_module import ScaleColumnParallelLinear, ColumnParallelLinear, RowParallelLinear
    from flash_attn.modules.embedding import VocabParallelEmbedding
    from flash_attn.ops.fused_dense import ColumnParallelLinear, RowParallelLinear
    # TODO 这里应该再新增一个关于 model 的组合信息，这样可以使得之后仅需要通过权重文件就可以组合出完整的整体model，
    #  实现的思路大概是通过（1）通过 named_modules 迭代循环；然后设定一些 colossalai 的 module 名称，这样可以自动判定需要concat的dimension；
    topos = {}
    # TODO 切换到flash attention
    for name, module in model.named_modules():
        if isinstance(module, ColossalaiModule) and not isinstance(module, Dropout):
            raise TypeError(f"ColossalaiModule is not supported yet, the module `{name}` is of ColossalaiModule type.")
        if isinstance(module, ParallelLayer):  # 这里就针对 Colossalai的ParallelLayer做一些特殊的处理就行
            if isinstance(module, (VocabParallelEmbedding1D, VocabParallelClassifier1D, CVocabParallelClassifier1D)):
                topos[name] = {'dim': 0}
            elif isinstance(module, Linear1D_Col):
                topos[name] = {'dim': 0}
            elif isinstance(module, Linear1D_Row):  
                topos[name + '.weight'] = {'dim': 1}
                topos[name + '.bias'] = {'dim': -1}

            # elif isinstance(module, type(model)):
            #     continue 
            # else:
            #     raise TypeError(f"{type(module)} is not supported yet, the module `{name}` is of `{type(module)}` type.")
        # 如果不满足这几个条件的，就是在各个tp/dp之间共享的，需要assert它们是一致的
        elif isinstance(module, VocabParallelEmbedding):
            topos[name] = {'dim': 0}
        elif isinstance(module, (ScaleColumnParallelLinear, ColumnParallelLinear)):
            topos[name] = {'dim': 0}
        elif isinstance(module, RowParallelLinear):  
            topos[name + '.weight'] = {'dim': 1}
            topos[name + '.bias'] = {'dim': -1}
    return topos


def _find_topo(name:str, topo:Dict) -> Tuple[str, Dict]:
    """
    给定 state_dict() 中的key名称，从 topo 中找出当前 key 对应的merge方式 ，会check是否只有一个 topo 和它是 match 的
    
    返回内容示例：{'dim': int}
    如果当前这个 name 不属于 tensor parallelism的，则会返回 None 

    """
    find_topo = None
    for key, _topo in topo.items():
        match = re.match(key+'(\.|$)', name)  # 要么是下一个点，要么就结束了
        if match is not None:
            assert match.span()[0] == 0
            assert find_topo is None, (find_topo, key)
            find_topo = (key, _topo)
    if find_topo is not None:
        return find_topo
    else:
        return None, None


def merge_weight(folder, no_pp_shard_name=('embedding.', 'head.', 'norm.weight', 'norm.bias'))->Dict:
    """
    会将在 folder 下的权重进行 merge 操作，其中出现在 no_pp_shard_name 中的权重名称将被 check 是否只在某一个 pipeline 部分存在，并且出现在了这个 tuple 中的
     权重将不会对名称有任何调整，其它的权重名称可能会被自增layer的index。

    返回 states 的 dict
    """
    _check_folder(folder)
    fns = _get_fns(folder)

    fps = []
    topo_fns = []
    model_fps = []
    model_fns = []
    for fn in fns:
        if fn.startswith('topo_') and not fn.endswith('md5'):
            fps.append(os.path.join(folder, fn))
            topo_fns.append(fn)
        elif fn.startswith('model_t') and not fn.endswith('md5'):  # 加入 _t 是为了避免和model_config.py冲突
            model_fps.append(os.path.join(folder, fn))
            model_fns.append(fn)
    assert len(model_fns) == len(topo_fns)  # 它们应该是一一对应的

    max_tp, max_pp = -1, -1
    for fn in topo_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_tp = max(max_tp, int(tp[2:])+1)
        max_pp = max(max_pp, int(pp[2:])+1)

    # 两个角度，在pp的角度，需要纵向；在tp的角度，需要横向

    # TODO 应该需要保证除了 no_pp_shard_name， 一定是有数字在中间的
    full_states = {}
    no_pp_shard_name_count = defaultdict(list)
    cur_layer_idx = 0
    all_state_keys = set()
    for _pp in range(max_pp):
        topo = None
        states = None
        for _tp in range(max_tp):
            _topo_fp = os.path.join(folder, f'topo_tp{_tp}_pp{_pp}.json')
            # with open(_topo_fp, 'r') as f:
            #     _topo = json.load(f)
            _topo = _load_states(_topo_fp)
            if topo is None:
                topo = _topo
            else:
                # 需要 assert 完全一致
                assert topo == _topo, (topo, _topo)

            _model_fp = os.path.join(folder, f'model_tp{_tp}_pp{_pp}.pt')
            _states = torch.load(_model_fp, map_location='cpu')
            if states is None:
                states = _states
            else:
                for name, state in _states.items():
                    find_key, _topo = _find_topo(name, topo)
                    if _topo is None or _topo['dim'] == -1:
                        # assert torch.allclose(state, states[name]), (name, find_key, state, states[name])
                        pass  # 在 flash attention 中只有rank=0有bias，所以无法进行check
                    else:
                        if len(state.shape)==1:
                            dim = 0
                        else:
                            dim = _topo['dim']
                        states[name] = torch.cat([states[name], state], dim=dim)
        # 这里完成对于 pp 的转换
        _cur_layer_idx = 0
        for name, state in states.items():
            all_state_keys.add(name)
            no_pp_flag = False
            for _name in no_pp_shard_name:
                match = re.search(_name, name)
                if match is not None:
                    no_pp_shard_name_count[_name].append((name, _pp))  
                    no_pp_flag = True
                    break
            if not no_pp_flag:
                match = re.search('\.\d+\.', name)  # 只 match 第一个满足的就行
                assert match is not None, (name, )  # 如果不是特殊的部分，那么一定是会满足类似于 'xxx.0.xxx'的形式的
                s, e= match.span()
                layer_idx = int(name[s+1:e-1])
                name = name[:s] + f'.{cur_layer_idx+layer_idx}.' + name[e:]
                _cur_layer_idx = max(layer_idx, _cur_layer_idx)
            full_states[name] = state
        cur_layer_idx += _cur_layer_idx + 1

    for name, finds in no_pp_shard_name_count.items():
        assert len(finds) == 1, f"The no_pp_shard_name:`{name}` should only be found once, but find {len(finds)} times in finds:{finds}"

    for name in no_pp_shard_name:
        assert name in no_pp_shard_name_count, f"no_pp_shard_name:`{name}` find no match in states keys:{all_state_keys}"

    return full_states


def _auto_load(fp):
    if 's3://' in fp:
        client = Client()
        with io.BytesIO(client.get(fp)) as f:
            states = torch.load(f, map_location='cpu')
    else:
        states = torch.load(fp, map_location='cpu')
    return states

def _auto_load_with_bar(fp, disable=True):
    if 's3://' in fp:
        client = Client()
        with load_with_progress_bar(fp, disable=disable) as f:
            states = torch.load(f, map_location='cpu')
    else:
        states = torch.load(fp, map_location='cpu')
    return states

def merge_weight_to_tp(folder, no_pp_shard_name=('embedding.', 'head.', 'norm.weight', 'norm.bias'), output_tp=None)->List[Dict]:
    """
    会将folder下的权重，首先进行merge，然后进行split到output_tp的需求, 当为None时，表示使用训练时的 tp ，这种情况下精度保持最佳，如果推理使用fp16的话，建议默认为None

    会将在 folder 下的权重进行 merge 操作，其中出现在 no_pp_shard_name 中的权重名称将被 check 是否只在某一个 pipeline 部分存在，并且出现在了这个 tuple 中的
     权重将不会对名称有任何调整，其它的权重名称可能会被自增layer的index。

    只能使用在flash attention的模块中，否则bias term会有问题

    返回 List[states]，每个序号内分别表示的是对应的 states 
    """
    _check_folder(folder)
    fns = _get_fns(folder)

    fps = []
    topo_fns = []
    model_fps = []
    model_fns = []
    for fn in fns:
        if fn.startswith('topo_') and not fn.endswith('md5'):
            fps.append(os.path.join(folder, fn))
            topo_fns.append(fn)
        elif fn.startswith('model_t') and not fn.endswith('md5'): # 加入 _t 是为了避免和model_config.py冲突
            model_fps.append(os.path.join(folder, fn))
            model_fns.append(fn)
    assert len(model_fns) == len(topo_fns)  # 它们应该是一一对应的

    max_tp, max_pp = -1, -1
    for fn in topo_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_tp = max(max_tp, int(tp[2:])+1)
        max_pp = max(max_pp, int(pp[2:])+1)
    if output_tp is None:
        output_tp = max_tp
    elif output_tp != max_tp:
        import warnings
        warnings.warn(f"The tp during training is {max_tp}, while inference is {output_tp}, this may cause results divergence.")
    # 两个角度，在pp的角度，需要纵向；在tp的角度，需要横向

    # TODO 应该需要保证除了 no_pp_shard_name， 一定是有数字在中间的
    full_states = [{} for i in range(output_tp)]
    no_pp_shard_name_count = defaultdict(list)
    cur_layer_idx = 0
    all_state_keys = set()
    for _pp in range(max_pp):
        topo = None
        states = None
        split_dims = {}
        for _tp in range(max_tp):
            _topo_fp = os.path.join(folder, f'topo_tp{_tp}_pp{_pp}.json')
            # with open(_topo_fp, 'r') as f:
            #     _topo = json.load(f)
            _topo = _auto_load(_topo_fp)
            if topo is None:
                topo = _topo
            else:
                # 需要 assert 完全一致
                assert topo == _topo, (topo, _topo)

            _model_fp = os.path.join(folder, f'model_tp{_tp}_pp{_pp}.pt')
            # _states = torch.load(_model_fp, map_location='cpu')
            _states = _auto_load(_model_fp)
            if states is None:
                states = _states
                for name in states:
                    split_dims[name] = -1  # 暂时都设置为 -1，主要是为了让flash attention中rank=0的才有col_linear的bias
            else:
                for name, state in _states.items():
                    find_key, _topo = _find_topo(name, topo)
                    if _topo is None or _topo['dim'] == -1:
                        # assert torch.allclose(state, states[name]), (name, find_key, state, states[name])
                        split_dims[name] = -1  # 在 flash attention 中只有rank=0有bias
                    else:
                        if len(state.shape)==1:
                            dim = 0
                        else:
                            dim = _topo['dim']
                        states[name] = torch.cat([states[name], state], dim=dim)
                        split_dims[name] = dim

        # 按照 output_tp 进行拆分
        states_list = [{} for _ in range(output_tp)]
        for name, tensor in states.items():
            if split_dims[name] == -1:
                if '.bias' in name and '.norm' not in name:
                    states_list[0][name] = tensor  # 只适用于flash attention这种 rank=0 才有bias的设定
                else:
                    for _ in range(output_tp):
                        states_list[_][name] = tensor
            else:
                assert tensor.shape[split_dims[name]]%output_tp == 0, f"The parameter:{name}'s {split_dims[name]} dimension is {tensor.shape[split_dims[name]]}, cannot split into {output_tp} parts."
                for t_idx, _tensor in enumerate(tensor.chunk(chunks=output_tp, dim=split_dims[name])):
                    states_list[t_idx][name] = _tensor
        
        # 这里完成对于 pp 的转换
        for s_idx, states in enumerate(states_list):
            _cur_layer_idx = _merge_pp(states, full_states[s_idx], all_state_keys, no_pp_shard_name, cur_layer_idx, no_pp_shard_name_count, _pp)
        cur_layer_idx += _cur_layer_idx + 1

    for name, finds in no_pp_shard_name_count.items():
        assert len(finds) == output_tp, f"The no_pp_shard_name:`{name}` should be found `{output_tp}` times, but find {len(finds)} times in finds:{finds}"

    for name in no_pp_shard_name:
        assert name in no_pp_shard_name_count, f"no_pp_shard_name:`{name}` find no match in states keys:{all_state_keys}"

    return full_states


def _merge_pp(states, full_states, all_state_keys, no_pp_shard_name, cur_layer_idx, no_pp_shard_name_count, _pp):
    _cur_layer_idx = 0
    for name, state in states.items():
        all_state_keys.add(name)
        no_pp_flag = False
        for _name in no_pp_shard_name:
            match = re.search(r''+_name, name)
            if match is not None:
                no_pp_shard_name_count[_name].append((name, _pp))  
                no_pp_flag = True
                break
        if not no_pp_flag:
            match = re.search('\.\d+\.', name)  # 只 match 第一个满足的就行
            assert match is not None, (name, )  # 如果不是特殊的部分，那么一定是会满足类似于 'xxx.0.xxx'的形式的
            s, e= match.span()
            layer_idx = int(name[s+1:e-1])
            name = name[:s] + f'.{cur_layer_idx+layer_idx}.' + name[e:]
            _cur_layer_idx = max(layer_idx, _cur_layer_idx)
        full_states[name] = state
    return _cur_layer_idx


def merge_pp(folder):
    """
    给定一个 folder ，merge 下面的 pipeline model

    """
    _check_folder(folder)
    fns = _get_fns(folder)

    model_fns = []
    for fn in fns:
        if fn.startswith('model_t') and not fn.endswith('md5'): # 加入 _t 是为了避免和model_config.py冲突
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_tp = max(max_tp, int(tp[2:])+1)
        max_pp = max(max_pp, int(pp[2:])+1)

    from tqdm import tqdm

    full_states = []
    for tp in tqdm(range(max_tp)):
        layer_shift = 0

        tp_states = {}
        for pp in tqdm(range(max_pp)):
            _layer_shift = 0
            model_name = f'model_tp{tp}_pp{pp}.pt'
            states = _auto_load(os.path.join(folder, model_name))
            keys = list(states.keys())
            for key in keys:
                match = re.search('\.\d+\.', key)
                if match is not None:  # 说明是 layer 相关的, 需要shift
                    s, e = match.span()
                    layer_idx = int(key[s+1:e-1]) + layer_shift
                    _layer_shift = max(_layer_shift, int(key[s+1:e-1]))
                    name = key[:s] + f'.{layer_idx}.' + key[e:]
                    tp_states[name] = states[key]
                else:
                    tp_states[key] = states[key]
            layer_shift += _layer_shift + 1
        full_states.append(tp_states)
    return full_states  # List[{}]，其中元素的长度是 tp 的数量



def save_model_checkpoint(batch_count, folder, model):
    """
    根据tp和dp的关系对model进行保存。其原理是，每个tp的数据不会进行gather，各自保存，这样相当于实际上进行了分片。保存的权重命名为
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt

    如果在之后的使用中tp与保存时不一致的话，需要对权重进行转换后才能进行 load 
    """
    from colossalai.core import global_context as gpc
    from colossalai.context.parallel_mode import ParallelMode

    with ParallelLayer.use_local_state_dict():  # 这样就不会进行 global 的 gather操作
        states = model.state_dict()
    topo = get_model_topology(model)

    if folder is not None:
        dp_size = get_dp_size()
        tp_size = get_tp_size()
        dp_rank = get_dp_rank()
        tp_rank = get_tp_rank()
        pp_rank = get_pp_rank()
        logger = get_dist_logger()
        # _check_folder(folder)

        # TODO 按道理应该还需要在 pp 级别上考虑下， 但是由于pp一般一定是跨机器的状态，所以即使不考虑pp，也一定不会在同一个机器写出了。
        should_save_rank_pair = set()  # (tp_rank, dp_rank)
        for i in range(tp_size):
            should_save_rank_pair.add((i, i%dp_size))

        if (tp_rank, dp_rank) in should_save_rank_pair:
            logger.info(f"Saving from global rank: {gpc.get_global_rank()}")
            fn = f'model_tp{tp_rank}_pp{pp_rank}.pt'
            fp = os.path.join(folder, fn)
            _save_states(batch_count, fp, states=states)
            topo_fn = f'topo_tp{tp_rank}_pp{pp_rank}.json'
            topo_fp = os.path.join(folder, topo_fn)
            _save_states(batch_count, topo_fp, topo)
    else:
        return {name:param.clone() for name, param in states.items()}  # state_dict是没有clone参数的，所以会导致修改是无法独立的

    torch.distributed.barrier()


def load_model_checkpoint(folder, model, states=None):
    """
    folder 下方应该有类似于以下名称的权重。
    - folder
        - model_tp{tp_rank}_pp{pp_rank}.pt 

    如果在之后的使用中tp与保存时不一致的话，需要对权重进行转换后才能进行 load 
    """
    if states is None:
        fns = _get_fns(folder)
        max_pp, max_tp = 0, 0
        for fn in fns:
            if fn.startswith('model_t') and not fn.endswith('.md5'):
                _, tp, pp = os.path.splitext(fn)[0].split('_')
                max_pp = max(max_pp, int(pp[2:]))
                max_tp = max(max_tp, int(tp[2:]))
        if not folder.startswith('/nvme'):
            assert get_pp_size() == max_pp + 1, f"The weights are save for {max_pp+1} pipelines, while current has {get_pp_size()} pipelines" 
            assert get_tp_size() == max_tp + 1, f"The weights are save for {max_tp+1} parallelism, while current has {get_tp_size()} tensor parallelism" 

        should_load_name = f'model_tp{get_tp_rank()}_pp{get_pp_rank()}.pt'
        fp = os.path.join(folder, should_load_name)
        states = _load_states(fp)
    with ParallelLayer.use_local_state_dict():  # 这样就是只 load 属于自己的那部分
        model.load_state_dict(states)
    torch.distributed.barrier()


def merge_pp_within_tp(folder, local_rank=None):
    """
    给定一个 folder ，merge 下面的 pipeline model

    """
    _check_folder(folder)
    fns = _get_fns(folder)

    model_fns = []
    for fn in fns:
        if fn.startswith('model_t') and not fn.endswith('md5'): # 加入 _t 是为了避免和model_config.py冲突
            model_fns.append(fn)

    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_pp = max(max_pp, int(pp[2:])+1)
        max_tp = max(max_tp, int(tp[2:])+1)
        
    if local_rank is None:
        assert max_tp == gpc.get_world_size(ParallelMode.TENSOR), f"The model trained with tp:{max_tp}, but current tp:{gpc.get_world_size(ParallelMode.TENSOR)}"
        tp = gpc.get_local_rank(ParallelMode.TENSOR)
    else:
        tp = local_rank
        
    layer_shift = 0

    tp_states = {}
    for pp in range(max_pp):
        _layer_shift = 0
        model_name = f'model_tp{tp}_pp{pp}.pt'
        states = _auto_load_with_bar(os.path.join(folder, model_name), disable = tp!=0 )
        keys = list(states.keys())
        for key in keys:
            match = re.search('\.\d+\.', key)
            if match is not None:  # 说明是 layer 相关的, 需要shift
                s, e = match.span()
                layer_idx = int(key[s+1:e-1]) + layer_shift
                _layer_shift = max(_layer_shift, int(key[s+1:e-1]))
                name = key[:s] + f'.{layer_idx}.' + key[e:]
                tp_states[name] = states[key]
            else:
                tp_states[key] = states[key]
        layer_shift += _layer_shift + 1

    return {(key[6:] if key.startswith('model.') else key):value for key,value in tp_states.items()}


def save_optimizer_ckeckpoint(batch_count, folder, optimizer):
    """
    每个 shard 自己存自己的
    - folder
        - optimizer_tp{tp_rank}_pp{pp_rank}_dp{dp_rank}.pt
    """
    # TODO sanity check for optimizer type
    dp_rank = get_dp_rank()
    states = optimizer.state_dict()
    if folder is not None:
        if 's3://' not in folder:
            assert os.path.exists(folder)
        # _check_folder(folder)
        fp = os.path.join(folder, f'optimizer_tp{get_tp_rank()}_pp{get_pp_rank()}_dp{dp_rank}.pt')
        _save_states(batch_count, fp, states)
    else:
        states = copy.deepcopy(states)
        return states


def load_optimizer_checkpoint(folder, optimizer, states=None):
    if states is None:
        fns = _get_fns(folder)
        max_tp, max_pp, max_dp = 0, 0, 0 
        for fn in fns:
            if fn.startswith('optimizer_') and not fn.endswith('.md5'):
                _, tp, pp, dp = os.path.splitext(fn)[0].split('_')
                max_dp = max(max_dp, int(dp[2:]))
                max_tp = max(max_tp, int(tp[2:]))
                max_pp = max(max_pp, int(pp[2:]))

        assert get_dp_size() == max_dp + 1, f"The weights are save for {max_dp+1} data parallel, while current has {get_dp_size()} data parallel."
        assert get_pp_size() == max_pp + 1, f"The weights are save for {max_pp+1} pipelines, while current has {get_pp_size()} pipelines" 
        assert get_tp_size() == max_tp + 1, f"The weights are save for {max_tp+1} parallelism, while current has {get_tp_size()} tensor parallelism" 

        should_load_fp = os.path.join(folder, f'optimizer_tp{get_tp_rank()}_pp{get_pp_rank()}_dp{get_dp_rank()}.pt')
        states = _load_states(should_load_fp)
    optimizer.load_state_dict(states)


def save_checkpoint(folder, batch_count, model, optimizer, scheduler, sampler, num_consumed_samples_in_epoch, other_stuffs:Dict, model_config:Dict=None):
    from colossalai.utils.megatron_timers import timer

    folder = os.path.join(folder, str(batch_count))
    # os.makedirs(folder, exist_ok=True)
    # if gpc.get_global_rank() == 0:
    #     assert len(os.listdir(folder)) == 0, f"`{folder}` is not empty, found {os.listdir(folder)}"
    start_time = time.time()
    torch.distributed.barrier()
    logger = get_dist_logger()
    logger.info(f"Saving checkpoint to `{folder}` at batch count:{batch_count} from rank:{gpc.get_global_rank()}...", ranks=[0])
    timer('save-model').start()
    save_model_checkpoint(batch_count, folder=folder, model=model)
    timer('save-model').stop()
    timer('save-optimizer').start()
    save_optimizer_ckeckpoint(batch_count, folder=folder, optimizer=optimizer)
    timer('save-optimizer').stop()
    if gpc.get_global_rank() == 0:
        scheduler_states = scheduler.state_dict()
        _save_states(batch_count, os.path.join(folder, 'schedulder.pt'), scheduler_states)

        sampler_state = sampler.state_dict()
        sampler_state['num_consumed_samples_in_epoch'] = num_consumed_samples_in_epoch  # 
        sampler_state['batch_count'] = batch_count
        _save_states(batch_count, os.path.join(folder, 'sampler.pt'), sampler_state)

        # torch.save(sampler_state, os.path.join(folder, 'sampler.pt'))
        _save_states(batch_count, os.path.join(folder, 'context.pt'), other_stuffs)
        # torch.save(other_stuffs, os.path.join(folder, 'context.pt'))

        if model_config is not None:
            _save_states(batch_count, os.path.join(folder, 'model_config.pt'), model_config)

    torch.distributed.barrier()
    duration = time.time() - start_time
    logger.info(f"Using {duration} seconds to save the checkpoint...", ranks=[0])
    timer.log(['save-model', 'save-optimizer'], logger=logger)
    _set_finish_flag(batch_count)


def load_checkpoint(folder, model, optimizer, scheduler, sampler, load_content='all', learning_rate=1e-4) -> Dict:
    """
    直接传入想要加载的 folder 路径，进行加载

    """
    start_time = time.time()
    torch.distributed.barrier()
    # assert os.path.exists(folder) and len(os.listdir(folder))>0, "The checkpoint folder should exist and have contents in it."
    _check_folder(folder=folder)
    get_dist_logger().info(f"Resume checkpoint from `{folder}`...", ranks=[0])
    load_model_checkpoint(folder=folder, model=model)
    if load_checkpoint not in ('model',):  # 如果只加载 model 就不需要 optimizer 了
        load_optimizer_checkpoint(folder=folder, optimizer=optimizer)
    if load_content not in ('model', 'model_optimizer') or 'optimizer' in load_content:
        scheduler_states = _load_states(os.path.join(folder, 'schedulder.pt'))
        # scheduler_states = torch.load(os.path.join(folder, 'schedulder.pt'), map_location='cpu')
        if load_content == 'no_lr':  # 使用新的 learning rate 覆盖之前的 learning rate， opt-175B中的经验
            get_dist_logger().warning(f"Using new learning rate {learning_rate} to replace old learn rate {scheduler_states['base_lrs'][0]}.")
            scheduler_states['base_lrs'] = [learning_rate]*len(scheduler_states['base_lrs'])
            if 'after_scheduler_dict' in scheduler_states:
                scheduler_states['after_scheduler_dict']['base_lrs'] = [learning_rate]*len(scheduler_states['after_scheduler_dict']['base_lrs'])
        scheduler.load_state_dict(scheduler_states)
        # sampler_states = torch.load(os.path.join(folder, 'sampler.pt'), map_location='cpu')
    sampler_states = _load_states(os.path.join(folder, 'sampler.pt'))
    sampler.load_state_dict(sampler_states)
    # other_stuffs = torch.load(os.path.join(folder, 'context.pt'), map_location='cpu')
    other_stuffs = _load_states(os.path.join(folder, 'context.pt'))
    torch.distributed.barrier()
    duration = time.time() - start_time
    get_dist_logger().info(f"Using {duration} seconds to load the checkpoint...", ranks=[0])

    return other_stuffs

def get_s3_object_file_size(fp):
    """
    给定一个具体的 S3 的文件路径，返回这个文件的bytes大小

    :param fp: _description_
    :return: _description_
    """
    import subprocess
    client = Client()
    assert client.contains(fp), f"`{fp}` is not exist"
    folder, fn = os.path.split(fp)
    # TODO: This is known to be buggy when the endopoint changes
    # need to figure out a better way to get the size of the file,
    # before which load_with_progress_bar will not have a progress bar
    output = subprocess.check_output(f'/usr/bin/aws --endpoint-url=http://10.140.14.254:80  s3 ls {folder}/', shell=True).decode('utf8')
    lines = output.split('\n')
    client.generate_presigned_url
    size = None
    for line in lines:
        parts = line.split()
        if parts:
            if parts[-1] == fn:
                size = int(parts[-2])
                break
    # print(size, flush=True)        
    assert size is not None, "size should be an int"
    return size


def load_with_progress_bar(fp, disable=True):
    # size = get_s3_object_file_size(fp)
    # pbar = tqdm(total=size, leave=False, disable=disable)
    client = Client()
    stream = client.get(fp, enable_stream=True)
    # chunk_size = min(size, 8192)
    f = io.BytesIO()
    for chunk in stream.iter_chunks(chunk_size=8192):
        f.write(chunk)
        # pbar.update(chunk_size)
    f.seek(0)
    return f