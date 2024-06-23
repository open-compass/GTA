import io
import re
import os
import os.path as osp
from typing import Optional, Union, List
from pathlib import Path

import io
import json
import os
import time

import colossalai
import torch
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Tokenizer, Transformer
from sentencepiece import SentencePieceProcessor

from opencompass.utils.internal.test.auto_test_dict import (basic_config, convert2run, gen_masks,
                            local_config_convert, maxdim2oridim, maxlay2orilay,
                            model_name2iter_info, model_name2module,
                            model_name2tokenizer_info)
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from .generation_tools import LLAMAGenerator, LLMGenerator
from opencompass.utils.internal.test.proxy import proxy_off
from sentencepiece import SentencePieceProcessor
from opencompass.utils.internal.train.checkpoint_utils import merge_pp_within_tp
from opencompass.utils.internal.model2.converted_llama.import_helper import try_import_petrel_client

Client = try_import_petrel_client()

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


class LLMTokenizer(object):

    def __init__(self, tokenizer, max_seq_len=2048, tokenizer_type='llama'):
        super().__init__()
        self.tokenizer = tokenizer
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

        # This is a hack to fit in with LLama type model
        self.bos_id = self.bos_token_id
        self.eos_id = self.eos_token_id
        self.pad_id = self.pad_token_id

    def __call__(self,
                 prompts,
                 padding=True,
                 right_align=False,
                 return_tensors='pt',
                 truncation=True):
        #import pdb; pdb.set_trace()
        if self.tokenizer_type == 'v4':
            tokens = [[0] + self.encode(x, False, False) for x in prompts]
        elif self.tokenizer_type in ['llama', 'v7']:
            tokens = [[1] + self.encode(x, False, False) for x in prompts]
        else:
            tokens = [self.encode(x, False, False) for x in prompts]

        if truncation:
            tokens = [i[:self.max_seq_len] for i in tokens]

        if padding:
            max_len = max([len(i) for i in tokens])
            if right_align:
                tokens = torch.LongTensor([[self.pad_token_id] *
                                           (max_len - len(i)) + i
                                           for i in tokens])
            else:
                tokens = torch.LongTensor([
                    i + [self.pad_token_id] * (max_len - len(i))
                    for i in tokens
                ])
        return {'tokens': tokens.cuda() if torch.cuda.is_available() else tokens}

    def encode(self, s: str, bos: bool, eos: bool):
        assert isinstance(s, str)
        s = self._process_meta_tokens(s)
        t = self._tokenize_list_str(s)
        if bos:
            t = [self.bos_token_id] + t
        if eos:
            t = t + [self.eos_token_id]
        return t

    def _process_meta_tokens(self, input_string: str) -> List[Union[str, int]]:
        # Create a pattern to match the META_TOKEN_{NUM} substrings
        pattern = re.compile(r'<META_TOKEN_(\d+)>')

        # Split the input string using the META_TOKEN_{NUM} substrings
        parts = pattern.split(input_string)

        # Combine the parts and tokens in the correct order
        result = []
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Regular text parts
                if part != '':
                    result.append(part)
            else:  # Meta token parts
                result.append(int(part))

        return result

    def _tokenize_list_str(self, s: Union[str, list]) -> List[int]:
        if isinstance(s, str):
            s = [s]
        assert isinstance(s, list)
        t = []
        for item in s:
            if isinstance(item, str):
                t += self.tokenizer.encode(item)
            elif isinstance(item, int):
                t.append(item)
            else:
                raise ValueError(f'Unsupported type {type(item)}')
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

def load_all(ckpt_dir, local_rank, world_size, max_seq_len,
             max_batch_size):  # No TP check.
    proxy_off()
    client = Client()

    if ckpt_dir.startswith(
            's3://model_weights/0331/'):  # pref_len=24, assert no merge
        assert not ckpt_dir.startswith(
            's3://model_weights/0331/evaluation/'), 'no merge, plz.'
        # s3 model must be in product line
        model_name, cur_iter = ckpt_dir[24:].split('/')[:2]
        try:
            cur_iter = int(cur_iter)
        except:
            print('Haha, read the code.')
            exit()

        assert client.isdir(ckpt_dir)
    elif ckpt_dir.startswith(
            '/mnt/petrelfs/share_data/llm_weight/'):  #36, unmerged
        assert not ckpt_dir.startswith(
            '/mnt/petrelfs/share_data/llm_weight/evaluation'), 'no merge, plz.'
        model_name, cur_iter = ckpt_dir[36:].split('/')[0].split('_')
        try:
            cur_iter = int(cur_iter)
        except:
            print('Haha, read the code.')
            exit()
        assert os.path.isdir(ckpt_dir)

    tokenizer_path, tokenizer_type = model_name2tokenizer_info(model_name)
    tokenizer = LLAMASentencePiece(max_seq_len=max_seq_len,
                                   tokenizer_type=tokenizer_type)
    tokenizer.load(tokenizer_path)

    print(model_name, tokenizer_type)

    config_file = os.path.join(ckpt_dir, 'model_config.pt')
    if ckpt_dir.startswith('s3://model_weights/0331/'):
        assert client.contains(config_file), 'Need config file!'
        print('Config loading', flush=True)
        with io.BytesIO(client.get(config_file)) as f:
            update_config = torch.load(f)
    elif ckpt_dir.startswith('/mnt/petrelfs/share_data/llm_weight/'):
        print('Config loading', flush=True)
        with open(config_file, 'rb') as f:
            update_config = torch.load(f)
        print('Config done!', flush=True)

    model_config = basic_config
    model_config.update(update_config)

    model_args = ModelArgs(max_seq_len=max_seq_len,
                           max_batch_size=max_batch_size,
                           dim=model_config['hidden_size'],
                           n_layers=model_config['num_layers'],
                           n_heads=model_config['num_attention_heads'],
                           vocab_size=tokenizer.vocab_size())

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)

    print('model weight loading', flush=True)

    states = merge_pp_within_tp(ckpt_dir, local_rank)

    print('model weight done!', flush=True)

    current_states = {}

    for i in range(model_args.n_layers):
        name = f'layers.{i}.attention.Wqkv.weight'
        wqkv = states.pop(name).reshape(model_args.n_heads // world_size, 3,
                                        -1, model_args.dim)

        current_states[f'layers.{i}.attention.wq.weight'] = wqkv[:, 0].reshape(
            -1, model_args.dim)
        current_states[f'layers.{i}.attention.wk.weight'] = wqkv[:, 1].reshape(
            -1, model_args.dim)
        current_states[f'layers.{i}.attention.wv.weight'] = wqkv[:, 2].reshape(
            -1, model_args.dim)

    current_states.update(states)
    load_info = model.load_state_dict(current_states, strict=False)
    print(load_info)

    return model, tokenizer, tokenizer_type


def setup_model_parallel(init_seed=1):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))

    torch.distributed.init_process_group('nccl')
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(init_seed)
    return local_rank, world_size


def load_from_llama(
    ckpt_dir: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
    tokenizer_path: str,
    tokenizer_type: str,
):
    proxy_off()
    client = Client()
    start_time = time.time()
    if "s3://" in ckpt_dir:
        checkpoints = []
        for file in client.list(ckpt_dir):
            file = os.path.join(ckpt_dir, file)
            if file.endswith('.pt') or file.endswith(".pth"):
                checkpoints.append(file)
        checkpoints = sorted(checkpoints)
    else:
        checkpoints = sorted([str(p) for p in Path(ckpt_dir).glob('*.pt*')])

    assert world_size == len(
        checkpoints
    ), f'Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}'

    ckpt_path = checkpoints[local_rank]
    print('Loading')
    if "s3://" in ckpt_path:
        with io.BytesIO(client.get(ckpt_path)) as f:
            checkpoint = torch.load(f, map_location='cpu')
        params = json.loads(client.get(os.path.join(ckpt_dir, 'params.json')).decode())
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
        tokenizer = LLMTokenizer(tokenizer, max_seq_len=max_seq_len, tokenizer_type=tokenizer_type)
        model_args.vocab_size = tokenizer.tokenizer.vocab_size()

    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLAMAGenerator(model, tokenizer)
    print(f'Loaded in {time.time() - start_time:.2f} seconds')
    return model, tokenizer, generator


def load_llama(checkpoint, max_seq_len=2048, batch_size=1, tokenizer_path: Optional[str]=None, tokenizer_type: Optional[str]=None):
    local_rank, world_size = setup_model_parallel()
    if tokenizer_path is None:
        tokenizer_path = os.path.join(
            checkpoint.rsplit('/', 1)[0], 'tokenizer.model')
    model, tokenizer, generator = load_from_llama(checkpoint,
                                                  local_rank, world_size,
                                                  max_seq_len, batch_size, tokenizer_path=tokenizer_path, tokenizer_type=tokenizer_type)
    return model, tokenizer, generator


def load_llm(checkpoint, max_seq_len=2048, tokenizer_path: Optional[str] = None, tokenizer_type: Optional[str] = None, module = None):
    proxy_off()
    client = Client()

    # init colossalai paralleling config
    WORLD_SIZE = os.getenv('WORLD_SIZE')
    if WORLD_SIZE == None:
        print('Supposed to launch with torchrun!')
        exit()
    TP = int(WORLD_SIZE)

    # parameter spliting:
    ckpts = checkpoint.split(';')
    assert ckpts
    ckpt = str(ckpts[0])

    if 's3://' in ckpt:
        # We assume that the path looks like the following:
        # opennlplab_hdd:s3://opennlplab_hdd/llm_it/0419/sft_7132k_flan64_8196/1399
        model_name, cur_iter = osp.realpath(ckpt).split('/')[-2:]
        try:
            cur_iter = int(cur_iter)
        except: # allow s3 path without iteration.
            cur_iter = 0
        assert client.isdir(ckpt)

    else:
        model_name, cur_iter = osp.realpath(ckpt).split('/')[-2:]
        try:
            cur_iter = int(cur_iter)
        except:
            if ckpt.startswith('/cpfs01/'):
                cur_iter = 0
            else:
                print('Haha, read the code.')
                exit()

        # 'model_tp0_pp*.pt' ~ 'model_tp7_pp*.pt'
        save_tp = 0
        for file in os.listdir(ckpt):
            if file.startswith('model_tp') and file.endswith('.pt'):
                save_tp = max(save_tp, int(file[8:].split('_')[0]))
        save_tp += 1

    colossalai.launch_from_torch(config={
        'parallel': {
            # 'pipeline': 2,
            'tensor': {
                'size': TP,
                'mode': '1d'
            }
        },
        'clip_grad_norm': 0.0
    }, seed=42)

    # print args info
    tp_rank = gpc.get_local_rank(ParallelMode.TENSOR)
    if tp_rank == 0:
        print(f'Args: ckpt={checkpoint}')

    tokenizer = SentencePieceProcessor()
    if not tokenizer_path:
        # TODO: delete these hardcoded mapping
        tokenizer_path, tokenizer_type = model_name2tokenizer_info(model_name)
        if "llamav4.model" in tokenizer_path:
            tokenizer_path = "/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model"# tmp change
    tokenizer.load(tokenizer_path)
    tokenizer = LLMTokenizer(tokenizer, max_seq_len=max_seq_len, tokenizer_type=tokenizer_type)

    if cur_iter == None:
        # local origin llama
        assert ckpt.startswith('/mnt/petrelfs/share_data/llm_llama/')
        config_path = os.path.join(ckpt, 'params.json')
        with open(config_path, 'r') as f:
            update_config = local_config_convert(json.load(f))
        assert update_config['vocab_size'] == -1, '?'
        update_config['vocab_size'] = tokenizer.vocab_size()
    else:
        print('Config loading', flush=True)
        config_file = os.path.join(ckpt, 'model_config.pt')
        if "s3://" in ckpt:
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

    model_config = basic_config
    model_config.update(update_config)
    model_config = convert2run(model_config)

    if module is None:
        model_module = model_name2module(model_name)
    else:
        model_module = module
        if tokenizer_type in ['llama', 'v6', 'v4']:
            model_config['embed_split_hidden'] = True

    if 'layer_norm_epsilon' in model_config:
        del model_config['layer_norm_epsilon']
    model = model_module(**model_config)
    states = merge_pp_within_tp(ckpt, tp_rank)
    if len(ckpts) > 1:
        for ckpt_ in ckpts[1:]:
            states_ = merge_pp_within_tp(ckpt_)
            for k in states_.keys():
                states[k] += states_[k]

        for k in states.keys():
            states[k] /= len(ckpts)

    load_info = model.load_state_dict(states, strict=False)
    if tp_rank == 0:
        print(load_info)
        if load_info.missing_keys:
            exit(-1)
    model = model.half().eval().cuda()

    warm_iter, max_iter = model_name2iter_info(model_name)
    use_mask = (warm_iter + max_iter) != 0

    if use_mask:
        feat_mask, ffn_mask, layer_mask = gen_masks(
            cur_iter = cur_iter,
            maxdim = model_config['hidden_size'],
            oridim = maxdim2oridim[model_config['hidden_size']],
            maxlay = model_config['num_layers'],
            orilay = maxlay2orilay[model_config['num_layers']],
            warm_iter = warm_iter,
            max_iter = max_iter
        )
    else:
        feat_mask, ffn_mask, layer_mask = None, None, None

    torch.distributed.barrier()

    generator = LLMGenerator(model, tokenizer, use_mask, forward_kwargs={"feat_mask": feat_mask, "ffn_mask": ffn_mask, "layer_mask": layer_mask})

    return model, tokenizer, generator, tp_rank
