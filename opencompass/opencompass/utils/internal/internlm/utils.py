import io
import os
import re

import torch
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc

from internlm.solver.pipeline_utils import partition_uniform

from .storage_manager import get_storage_manager
from opencompass.utils import get_logger


basic_config = dict(num_chunks=1,
                    checkpoint=False,
                    dtype=torch.half,
                    embed_split_hidden=False,
                    num_layers=40,
                    hidden_size=5120,
                    vocab_size=150494,
                    embed_grad_scale=1,
                    parallel_output=False,
                    num_attention_heads=40,
                    mlp_ratio=8 / 3,
                    apply_post_layer_norm=False,
                    residual_in_fp32=False,
                    norm_type='rmsnorm',
                    drop_rate=0,
                    attn_drop_rate=0)

backup = {}


def try_import_petrel_client():
    """
    Overview:
        Try import petrel_client module, if failed, return ``None``

    Returns:
        - (:obj:`Module`): Imported module,
            or ``None`` when petrel_client not found
    """
    try:
        from petrel_client.client import Client

        Client()
        return Client
    except Exception as e:
        print(f'petrel_client.client import error! {e}', flush=True)
        return lambda *args, **kwargs: None


Client = try_import_petrel_client()


def _check_folder(folder):
    get_storage_manager().check_folder(folder)


def _get_fns(folder):

    fns = get_storage_manager().get_fns(folder)
    fns2 = [fn.split("/")[-1] for fn in fns]

    return fns2

def load_with_progress_bar(fp, disable=True):
    client = Client()
    # stream = client.get(fp, enable_stream=True)
    # f = io.BytesIO()
    # for chunk in stream.iter_chunks(chunk_size=8192):
    #     f.write(chunk)
    # f.seek(0)
    res = client.get(fp)
    f = io.BytesIO(res)
    return f


def _auto_load_with_bar(fp, disable=True):
    if 's3://' in fp:
        client = Client()
        with load_with_progress_bar(fp, disable=disable) as f:
            states = torch.load(f, map_location='cpu')
    else:
        states = torch.load(fp, map_location='cpu')
    return states


def merge_pp_within_tp(folder, local_rank=None, w2w3_bug=False):
    _check_folder(folder)
    fns = _get_fns(folder)

    model_fns = []
    for fn in fns:
        if fn.startswith('model_t') and not fn.endswith('md5'):
            model_fns.append(fn)
    max_tp, max_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split('_')
        max_pp = max(max_pp, int(pp[2:]) + 1)
        max_tp = max(max_tp, int(tp[2:]) + 1)

    if local_rank is None:
        assert max_tp == gpc.get_world_size(
            ParallelMode.TENSOR
        ), f'The model trained with tp:{max_tp}, but current tp:{gpc.get_world_size(ParallelMode.TENSOR)}'  # noqa: E501
        tp = gpc.get_local_rank(ParallelMode.TENSOR)
    else:
        tp = local_rank

    layer_shift = 0
    tp_states = {}
    for pp in range(max_pp):
        _layer_shift = 0
        model_name = f'model_tp{tp}_pp{pp}.pt'
        states = _auto_load_with_bar(os.path.join(folder, model_name),
                                     disable=tp != 0)
        keys = list(states.keys())
        for key in keys:
            match = re.search('\.\d+\.', key)  # noqa: W605
            if match is not None:
                s, e = match.span()
                layer_idx = int(key[s + 1:e - 1]) + layer_shift
                _layer_shift = max(_layer_shift, int(key[s + 1:e - 1]))
                name = key[:s] + f'.{layer_idx}.' + key[e:]
                # a potential bug in some train internlm model, may get
                # w2 w3 missmatch error
                if w2w3_bug and 'w2' in name:
                    tp_states[name.replace('w2', 'w3')] = states[key]
                elif w2w3_bug and 'w3' in name:
                    tp_states[name.replace('w3', 'w2')] = states[key]
                else:
                    tp_states[name] = states[key]
            else:
                tp_states[key] = states[key]
        layer_shift += _layer_shift + 1

    return {(key[6:] if key.startswith('model.') else key): value
            for key, value in tp_states.items()}

def load_with_dynamic_tp_pp(folder):
    """
    Only internlm2 is allowed.
    """
    assert folder is not None, "Please specify the folder of the pretrained model"
    logger = get_logger()
    if gpc.is_rank_for_log():
        logger.info(f"Loading pretrained model from {folder}")

    fns = _get_fns(folder)
    model_fns = []
    for fn in fns:
        # filter with `_t` is for avoiding conflict with model_config.py
        if fn.startswith("model_t") and not fn.endswith("md5"):
            model_fns.append(fn)

    old_tp, old_pp = -1, -1
    for fn in model_fns:
        _, tp, pp = os.path.splitext(fn)[0].split("_")
        old_tp = max(old_tp, int(tp[2:]) + 1)
        old_pp = max(old_pp, int(pp[2:]) + 1)

    assert old_tp > 0 and old_pp > 0, f"ckpt with tp:{old_tp} and pp:{old_pp} is illegal"

    tp = gpc.get_world_size(ParallelMode.TENSOR)
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    assert old_tp % tp == 0 or tp % old_tp == 0, (
        f"Expected TP size in loaded checkpoint to be fit with TP size in current config, but got {old_tp} in "
        f"checkpoint and {tp} in current config"
    )

    correspond_tps = []

    if old_tp <= tp:
        correspond_tps.append(tp_rank // (tp // old_tp))
        ratio = tp // old_tp
        rank = tp_rank % ratio
    else:
        for i in range(old_tp // tp):
            correspond_tps.append(tp_rank * (old_tp // tp) + i)
        rank = 0
        ratio = 1

    current_states = {}

    pp = gpc.get_world_size(ParallelMode.PIPELINE)

    assert gpc.config.model.num_chunks == 1, "May cause future collisions, ignore this if necessary"

    old_pp_partition = partition_uniform(gpc.config.model.num_layers, old_pp, 1)

    for idx, parts in enumerate(old_pp_partition):
        start, end = parts[0]

        tmp_states = {}

        for correspond_tp in correspond_tps:
            model_name = f"model_tp{correspond_tp}_pp{idx}.pt"
            states = _auto_load_with_bar(os.path.join(folder, model_name))
            for i in range(start, end):
                for name in list(states.keys()):
                    if f".{i-start}." in name:
                        to_name = name.replace(f".{i-start}.", f".{i}.")
                        if "norm" in name:
                            tmp_states[to_name] = [states.pop(name)]
                        elif any(x in name for x in ("wo", "w2")):
                            tmp_states[to_name] = tmp_states.get(to_name, [])
                            tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=1)[rank])
                        elif any(x in name for x in ("w1", "w3", "wqkv")):
                            tmp_states[to_name] = tmp_states.get(to_name, [])
                            tmp_states[to_name].append(states.pop(name).chunk(ratio, dim=0)[rank])
                        else:
                            raise KeyError(f"Unknown key {name}.")

            if "tok_embeddings.weight" in states:
                tmp_states["tok_embeddings.weight"] = tmp_states.get("tok_embeddings.weight", [])
                tmp_states["tok_embeddings.weight"].append(states["tok_embeddings.weight"].chunk(ratio, dim=1)[rank])
            if "model.tok_embeddings.weight" in states:
                tmp_states["tok_embeddings.weight"] = tmp_states.get("tok_embeddings.weight", [])
                tmp_states["tok_embeddings.weight"].append(states["model.tok_embeddings.weight"].chunk(ratio, dim=1)[rank])
            if "output.weight" in states:
                tmp_states["norm.weight"] = [states["norm.weight"]]
                tmp_states["output.weight"] = tmp_states.get("output.weight", [])
                tmp_states["output.weight"].append(states["output.weight"].chunk(ratio, dim=0)[rank])
            if "model.output.weight" in states:
                tmp_states["norm.weight"] = [states["model.norm.weight"]]
                tmp_states["output.weight"] = tmp_states.get("output.weight", [])
                tmp_states["output.weight"].append(states["model.output.weight"].chunk(ratio, dim=0)[rank])

            states = {}

        for name in list(tmp_states.keys()):
            data = tmp_states.pop(name)
            if len(data) == 1:
                current_states[name] = data[0]
            else:
                current_states[name] = torch.concat(
                    data, dim=1 if name == "tok_embeddings.weight" or any(x in name for x in ("wo", "w2")) else 0
                )

    return {(key[6:] if key.startswith('model.') else key): value
            for key, value in current_states.items()}

def proxy_off():
    global backup
    if 'http_proxy' in os.environ:
        backup['http_proxy'] = os.environ['http_proxy']
        del os.environ['http_proxy']
    if 'https_proxy' in os.environ:
        backup['https_proxy'] = os.environ['https_proxy']
        del os.environ['https_proxy']
    if 'HTTP_PROXY' in os.environ:
        backup['HTTP_PROXY'] = os.environ['HTTP_PROXY']
        del os.environ['HTTP_PROXY']
    if 'HTTPS_PROXY' in os.environ:
        backup['HTTPS_PROXY'] = os.environ['HTTPS_PROXY']
        del os.environ['HTTPS_PROXY']


def proxy_on():
    global backup
    if 'http_proxy' in backup:
        os.environ['http_proxy'] = backup['http_proxy']
    if 'https_proxy' in backup:
        os.environ['https_proxy'] = backup['https_proxy']
    if 'HTTP_PROXY' in backup:
        os.environ['HTTP_PROXY'] = backup['HTTP_PROXY']
    if 'HTTPS_PROXY' in backup:
        os.environ['HTTPS_PROXY'] = backup['HTTPS_PROXY']


def convert2run(model_config, tokenizer_type, model_dtype=None):
    logger = get_logger()
    if  model_dtype is None or model_dtype == "":
        logger.info(f'model_dtype not set, model_config["dtype"]: {model_config["dtype"]} is used in this case!\n'
                    'This log needs extra attention if you are using a legacy config file, as models were running in `torch.float16` by default!')
        model_dtype = model_config["dtype"]
    else:
        logger.info(f'model_dtype: {model_dtype} is used!')

    if model_dtype == "torch.bfloat16":
        model_config["dtype"] = torch.bfloat16
    elif model_dtype in ("torch.float16", "torch.half"):
        model_config["dtype"] = torch.float16
    elif model_dtype == "torch.float32":
        model_config["dtype"] = torch.float32
    elif model_dtype == "torch.tf32":
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        model_config["dtype"] = torch.float32
    elif not isinstance(model_dtype, torch.dtype):
        raise NotImplementedError(f"Unknown dtype {model_dtype}")
    return model_config
