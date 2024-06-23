from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM
from opencompass.models import GLM130B

with read_base():
    from .summarizers.medium import summarizer
    from .datasets.glm.agieval import agieval_datasets
    # from .datasets.glm.ceval import datasets
    from .datasets.glm.mmlu import mmlu_datasets
    from .datasets.glm.afqmc import afqmc_datasetsdatasets
    from .datasets.glm.chid import chid_datasets
    from .datasets.glm.GaokaoBench import gaokao_bench_datasets


models = [
    # GLM130B
    dict(
        type=GLM130B,
        pkg_root='/mnt/petrelfs/mazerun/Git/GLM-130B',
        ckpt_path='/mnt/petrelfs/share_data/yanhang/weights/glm130b//glm-130b-sat',
        max_seq_len=2048,
        max_out_len=50,
        batch_size=1,
        run_cfg=dict(num_gpus=8),
        abbr='glm130b',
    ),
]
