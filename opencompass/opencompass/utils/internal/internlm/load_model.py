import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import internlm
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from internlm.core.context import ParallelMode
from internlm.core.context import global_context as gpc
from internlm.core.context import Config
from internlm.utils.storage_manager import init_storage_manager, SingletonMeta
try:
    from model.utils.model_checkpoint import DefaultLoadFuncDict
except:
    DefaultLoadFuncDict = None
from llama import Llama, ModelArgs, Tokenizer, Transformer
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer

from .generation import LLAMAGenerator, LLMGenerator
from .tokenizer import LLMTokenizer
from .utils import (basic_config, convert2run, merge_pp_within_tp, proxy_off,
                    try_import_petrel_client, load_with_dynamic_tp_pp)

Client = try_import_petrel_client()


def setup_model_parallel(init_seed=1):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    torch.distributed.init_process_group('nccl')
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(init_seed)
    return local_rank, world_size


def load_llm(checkpoint,
             max_seq_len=2048,
             tokenizer_path: Optional[str] = None,
             tokenizer_type: Optional[str] = None,
             module=None,
             model_type: Optional[str] = None,
             ckpt_type: Optional[str] = None,
             model_config_path=None,
             model_dtype=None,
             w2w3_bug=False,
             mode='none'):
    proxy_off()
    client = Client()
    WORLD_SIZE = os.getenv('WORLD_SIZE')
    if WORLD_SIZE is None:
        print('Supposed to launch with torchrun!')
        exit()
    TP = int(WORLD_SIZE)
    ckpts = checkpoint.split(';')
    assert ckpts
    ckpt = str(ckpts[0])
    model_context = {
        'model': {
            'checkpoint': False,
            'num_chunks': 1,
            'num_attention_heads': 32,
            'embed_split_hidden': False,
            'vocab_size': 103168,
            'embed_grad_scale': 1,
            'parallel_output': False,
            'hidden_size': 4096,
            'num_layers': 32,
            'mlp_ratio': 2.6666666666666665,
            'apply_post_layer_norm': False,
            'no_bias': True,
            'deepnorm': False,
            'dtype': 'torch.bfloat16',
            'norm_type': 'rmsnorm',
            'layer_norm_epsilon': 1e-05,
            'use_flash_attn': True,
            'embed_fp32': True
        },
        'parallel': dict(zero1=0, pipeline=dict(size=1), tensor=TP),
    }
    if hasattr(internlm.initialize, 'initialize_distributed_env'):
        if model_config_path is None:
            internlm.initialize.initialize_distributed_env(config=model_context, launcher="torch", args_check=False)
        else:
            os.environ["JOB_NAME"] = 'test'
            os.environ["CLUSTER_NAME"] = 'A800'
            os.environ["TRAIN_FOLDER"] = 'test'
            os.environ["VALID_FOLDER"] = 'test'
            os.environ["TOTAL_STEP"] = '1000'
            os.environ["SAVE_CKPT_FOLDER"] = "test"
            config = Config.from_file(model_config_path)
            config.parallel.zero1 = 0
            config.parallel.pipeline.size = 1
            config.parallel.sequence_parallel = False
            config.parallel.tensor = TP
            config.model.parallel_output = False

            internlm.initialize.initialize_distributed_env(config=config, launcher="torch", args_check=False)
    else:
        if model_config_path is None:
            internlm.launch_from_torch(config=model_context, seed=42, args_check=False)
        else:
            os.environ["JOB_NAME"] = 'test'
            os.environ["CLUSTER_NAME"] = 'A800'
            os.environ["TRAIN_FOLDER"] = 'test'
            os.environ["VALID_FOLDER"] = 'test'
            os.environ["TOTAL_STEP"] = '1000'
            os.environ["SAVE_CKPT_FOLDER"] = "test"
            config = Config.from_file(model_config_path)
            config.parallel.zero1 = 0
            config.parallel.tensor = TP
            config.parallel.pipeline.size = 1
            config.parallel.sequence_parallel = False
            config.model.parallel_output = False
            internlm.launch_from_torch(config=config, seed=42, args_check=False)
    # print args info
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    if tp_rank == 0:
        print(f'Args: ckpt={checkpoint}')

    if os.path.isdir(tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, add_bos_token=False, add_eos_token=False,
            trust_remote_code=True
        )
    else:
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
    tokenizer = LLMTokenizer(tokenizer,
                             max_seq_len=max_seq_len,
                             tokenizer_type=tokenizer_type,
                             mode=mode)
    if model_config_path is not None:
        print('Beginning to load model_config', flush=True)
        update_config = gpc.config.model
        print('Config done!', flush=True)
    else:
        print('Config loading', flush=True)
        config_file = os.path.join(ckpt, 'model_config.pt')
        if 's3://' in ckpt:
            if client.contains(config_file):
                with io.BytesIO(client.get(config_file)) as f:
                    update_config = torch.load(f)
            else:
                config_file = os.path.join(ckpt, 'params.json')
                update_config = json.loads(client.get(config_file).decode())
        else:
            with open(config_file, 'rb') as f:
                update_config = torch.load(f)
        print('Config done!', flush=True)

    # model_config = basic_config
    model_config = update_config
    model_config = convert2run(model_config, tokenizer_type, model_dtype)
    model = module(**model_config)
    if ckpt_type is None:
        if model_type == "INTERNLM2":
            states = load_with_dynamic_tp_pp(ckpt)
        else:
            states = merge_pp_within_tp(ckpt, tp_rank, w2w3_bug=w2w3_bug)
            if len(ckpts) > 1:
                for ckpt_ in ckpts[1:]:
                    states_ = merge_pp_within_tp(ckpt_, w2w3_bug=w2w3_bug)
                    for k in states_.keys():
                        states[k] += states_[k]

                for k in states.keys():
                    states[k] /= len(ckpts)
        load_info = model.load_state_dict(states, strict=False)
        if tp_rank == 0:
            i = 0
            while i < len(load_info.missing_keys):
                if 'inv_freq' in load_info.missing_keys[i]:
                    load_info.missing_keys.pop(i)
                else:
                    i += 1
            print(load_info)
            if load_info.missing_keys:
                exit(-1)
    elif ckpt_type in DefaultLoadFuncDict:
        SingletonMeta._instances = {}
        init_storage_manager(False, None, None)
        load_func = DefaultLoadFuncDict[ckpt_type]
        load_func(ckpt, model)
    else:
        raise NotImplementedError(f"ckpt_type {ckpt_type} is not found, please check.")

    model = model.to(model_config["dtype"]).cuda().eval()

    torch.distributed.barrier()
    use_mask = False
    generator = LLMGenerator(model,
                             tokenizer,
                             use_mask,
                             forward_kwargs={
                                 'feat_mask': None,
                                 'ffn_mask': None,
                                 'layer_mask': None
                             })

    return model, tokenizer, generator, tp_rank


def load_from_llama(
    ckpt_dir: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    tokenizer_path: str,
    tokenizer_type: str,
) -> Llama:
    proxy_off()
    client = Client()
    start_time = time.time()
    if 's3://' in ckpt_dir:
        checkpoints = []
        for file in client.list(ckpt_dir):
            file = os.path.join(ckpt_dir, file)
            if file.endswith('.pt') or file.endswith('.pth'):
                checkpoints.append(file)
        checkpoints = sorted(checkpoints)
    else:
        checkpoints = sorted([str(p) for p in Path(ckpt_dir).glob('*.pt*')])
    assert world_size == len(
        checkpoints
    ), f'Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}'

    ckpt_path = checkpoints[local_rank]
    print('Loading')
    if 's3://' in ckpt_path:
        with io.BytesIO(client.get(ckpt_path)) as f:
            checkpoint = torch.load(f, map_location='cpu')
        params = json.loads(
            client.get(os.path.join(ckpt_dir, 'params.json')).decode())
    else:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        with open(Path(ckpt_dir) / 'params.json', 'r') as f:
            params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len,
                                      max_batch_size=max_batch_size,
                                      **params)
    if tokenizer_type in ['llama', 'v7']:
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
    else:
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        tokenizer = LLMTokenizer(tokenizer,
                                 max_seq_len=max_seq_len,
                                 tokenizer_type=tokenizer_type)
        model_args.vocab_size = tokenizer.tokenizer.vocab_size()

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLAMAGenerator(model, tokenizer)
    print(f'Loaded in {time.time() - start_time:.2f} seconds')
    return model, tokenizer, generator


def load_llama(checkpoint,
               max_seq_len=2048,
               batch_size=1,
               tokenizer_path: Optional[str] = None,
               tokenizer_type: Optional[str] = None):
    local_rank, world_size = setup_model_parallel()
    if tokenizer_path is None:
        tokenizer_path = os.path.join(
            checkpoint.rsplit('/', 1)[0], 'tokenizer.model')
    model, tokenizer, generator = load_from_llama(
        checkpoint,
        local_rank,
        world_size,
        max_seq_len,
        batch_size,
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type)
    return model, tokenizer, generator


class LLAMASentencePiece(SentencePieceProcessor):

    def __init__(self, max_seq_len=2048, tokenizer_type='llama'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenizer_type = tokenizer_type
        if self.tokenizer_type == 'v4':
            self.bos_token_id = self.pad_token_id = 0
            self.eos_token_id = 1
        elif self.tokenizer_type in ['llama', 'v7']:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 2
        else:
            self.bos_token_id = self.pad_token_id = 1
            self.eos_token_id = 0

    def __call__(self,
                 prompts,
                 padding=True,
                 return_tensors='pt',
                 truncation=True):
        if isinstance(prompts, list):
            raise TypeError('mixing int and strings in a list as a prompt'
                            'is not yet supported in llama')
        if self.tokenizer_type == 'v4':
            tokens = [[0] + self.encode(x) for x in prompts]
        elif self.tokenizer_type in ['llama', 'v7']:
            tokens = [[1] + self.encode(x) for x in prompts]
        else:
            tokens = [self.encode(x) for x in prompts]

        if truncation:
            tokens = [i[:self.max_seq_len] for i in tokens]

        if padding:
            max_len = max([len(i) for i in tokens])
            tokens = torch.LongTensor(
                [i + [self.pad_token_id] * (max_len - len(i)) for i in tokens])
        return {'tokens': tokens.cuda(), 'start_pos': 0}
