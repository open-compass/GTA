# This config aims to
from mmengine.config import read_base

from opencompass.models.internal import LLMv2
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

with read_base():
    from .datasets.agieval.agieval_gen import agieval_datasets
    from .datasets.ceval.ceval_ppl import ceval_datasets
    from .datasets.CLUE_CMRC.CLUE_CMRC_gen import CMRC_datasets
    from .datasets.CLUE_DRCD.CLUE_DRCD_gen import DRCD_datasets
    from .datasets.crowspairs.crowspairs_ppl import crowspairs_datasets
    from .datasets.FewCLUE_chid.FewCLUE_chid_ppl import chid_datasets
    from .datasets.FewCLUE_cluewsc.FewCLUE_cluewsc_ppl import cluewsc_datasets
    from .datasets.FewCLUE_csl.FewCLUE_csl_ppl import csl_datasets
    from .datasets.FewCLUE_eprstmt.FewCLUE_eprstmt_ppl import eprstmt_datasets
    from .datasets.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets
    from .datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.math.math_gen import math_datasets
    from .datasets.mbpp.mbpp_gen import mbpp_datasets
    from .datasets.mmlu.mmlu_gen import mmlu_datasets
    from .datasets.nq.nq_gen import nq_datasets
    from .datasets.race.race_ppl import race_datasets
    from .datasets.realtoxicprompts.realtoxicprompts_gen import \
        realtoxicprompts_datasets
    from .datasets.triviaqa.triviaqa_gen import triviaqa_datasets
    from .datasets.truthfulqa.truthfulqa_gen import truthfulqa_datasets

openai_key = 'ENV'

infer = dict(
    partitioner=dict(type=SizePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLInferTask),
        retry=5),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=SlurmRunner,
        max_num_workers=64,
        task=dict(type=OpenICLEvalTask),
        retry=5),
)

datasets = []
datasets += chid_datasets
datasets += cluewsc_datasets
datasets += eprstmt_datasets
datasets += CMRC_datasets
datasets += DRCD_datasets
datasets += humaneval_datasets
datasets += mbpp_datasets
datasets += math_datasets
datasets += gsm8k_datasets
datasets += crowspairs_datasets
datasets += realtoxicprompts_datasets
datasets += truthfulqa_datasets
datasets += race_datasets
datasets += triviaqa_datasets
datasets += nq_datasets
datasets += csl_datasets
datasets += agieval_datasets
datasets += mmlu_datasets
datasets += ceval_datasets
datasets += GaokaoBench_datasets

models = [
    dict(
        abbr='PJLM-v0.2.0-Exam-v0.1.5',
        type=LLMv2,
        model_type='converted',
        path='model_weights:s3://model_weights/0331/1006_pr/5499/',
        tokenizer_path=
        '/mnt/petrelfs/share_data/yanhang/tokenizes/llamav4.model',
        tokenizer_type='v4',
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=8, num_procs=8)),
]
