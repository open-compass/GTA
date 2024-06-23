# This config aims to
from opencompass.models import LLMv2
from mmengine.config import read_base
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.runners import SlurmRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .datasets.ChineseUniversal.FewCLUE_chid.FewCLUE_chid_ppl import chid_datasets
    from .datasets.ChineseUniversal.FewCLUE_cluewsc.FewCLUE_cluewsc_ppl import cluewsc_datasets
    from .datasets.ChineseUniversal.FewCLUE_eprstmt.FewCLUE_eprstmt_ppl import eprstmt_datasets
    from .datasets.ChineseUniversal.CLUE_CMRC.CLUE_CMRC_gen import CMRC_datasets
    from .datasets.ChineseUniversal.CLUE_DRCD.CLUE_DRCD_gen import DRCD_datasets
    from .datasets.Coding.humaneval.humaneval_gen import humaneval_datasets
    from .datasets.Coding.mbpp.mbpp_gen import mbpp_datasets
    from .datasets.Reasoning.math.math_gen import math_datasets
    from .datasets.Reasoning.gsm8k.gsm8k_gen import gsm8k_datasets
    from .datasets.Security.crowspairs.crowspairs_ppl import crowspairs_datasets
    from .datasets.Security.realtoxicprompts.realtoxicprompts_gen import realtoxicprompts_datasets
    from .datasets.Security.truthfulqa.truthfulqa_gen import truthfulqa_datasets
    from .datasets.EnglishUniversal.race.race_ppl import race_datasets
    from .datasets.QA.triviaqa.triviaqa_gen import triviaqa_datasets
    from .datasets.QA.nq.nq_gen import nq_datasets
    from .datasets.ChineseUniversal.FewCLUE_csl.FewCLUE_csl_ppl import csl_datasets
    from .datasets.Exam.agieval.agieval_gen import agieval_datasets
    from .datasets.Exam.mmlu.mmlu_gen import mmlu_datasets
    from .datasets.Exam.ceval.ceval_ppl import ceval_datasets
    from .datasets.Exam.GaokaoBench.GaokaoBench_gen import GaokaoBench_datasets

# When it is set to "ENV", the key will be fetched from the environment
# variable $OPENAI_API_KEY.
openai_key = 'ENV'

datasets = []
# not important
# datasets += chid_datasets
# datasets += cluewsc_datasets
# datasets += eprstmt_datasets
# datasets += humaneval_datasets
# datasets += mbpp_datasets
# datasets += gsm8k_datasets
# datasets += crowspairs_datasets
# datasets += realtoxicprompts_datasets
# datasets += truthfulqa_datasets
# datasets += race_datasets
# datasets += flores_datasets
# # important
# datasets += triviaqa_datasets
# datasets += nq_datasets
# datasets += CMRC_datasets
# datasets += DRCD_datasets
# datasets += csl_datasets
# datasets += agieval_datasets
datasets += mmlu_datasets
# datasets += ceval_datasets
# datasets += GaokaoBench_datasets


infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=2000),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=dict(
            bashrc_path="/cpfs01/user/zhoufengzhe/.bashrc",
            conda_env_name='pjeval',
            dlc_config_path="/cpfs01/user/zhoufengzhe/.dlc/config",
            workspace_id='ws28twxwbsauodwa',
            worker_image='pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/pytorch:py3.6-torch1.8-cuda11.1-rdma5.2-sshd-ubuntu18.04',
        ),
        task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLEvalTask)),
)


models = [
    dict(abbr='ChatPJLM-v0.2.3rc43-womi',
        type=LLMv2, model_type='converted', path='/cpfs01/shared/public/llmit/ckpt/sft_1006_v023rc43',
        tokenizer_path='/cpfs01/shared/public/tokenizers/llamav4.model', tokenizer_type='v4',
        meta_template=_template_0_1_0_ChatPJLM_v0_2_1rc1,
        max_out_len=100, max_seq_len=2048, batch_size=8, run_cfg=dict(num_gpus=8)),
]
