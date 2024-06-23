from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import DS1000Dataset, DS1000ServiceEvaluator

ds1000_reader_cfg = dict(
    input_columns=["prompt"],
    output_column="test_column",
    train_split='test',
    test_split='test')

ds1000_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt="{prompt}",
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

ds1000_eval_cfg_dict = {
    lib: dict(
        evaluator=dict(
            type=DS1000ServiceEvaluator,
            lib=lib,
            ip_address=
            # EXTERNAL "localhost",  # replace to your code_eval_server ip_address, port
            # INTERNAL_BEGIN
            # If the IP address is not accessible, 
            # follow the instructions below to launch a code evaluate service.
            # https://aicarrier.feishu.cn/docx/JpLAdWNh9oGC1fxH9Z9cTobLntb
            # use these ip as fallback if service is down
            # "http://10.140.60.10" for T cluster, http://10.140.0.133 for S cluster
            "http://service.opencompass.org.cn",  # Need to use proxy on cluster T or S
            # INTERNAL_END
            port=5000
            ),
        pred_role="BOT")
    for lib in [
        'Pandas',
        'Numpy',
        'Tensorflow',
        'Scipy',
        'Sklearn',
        'Pytorch',
        'Matplotlib',
    ]
}

# The DS-1000 dataset can be downloaded from
# https://github.com/HKUNLP/DS-1000/blob/main/ds1000_data.zip
ds1000_datasets = [
    dict(
        abbr=f"ds1000_{lib}",
        type=DS1000Dataset,
        path="./data/ds1000_data/",
        libs=f"{lib}",
        mode="Completion",
        reader_cfg=ds1000_reader_cfg,
        infer_cfg=ds1000_infer_cfg,
        eval_cfg=ds1000_eval_cfg_dict[lib],
    ) for lib in [
        'Pandas',
        'Numpy',
        'Tensorflow',
        'Scipy',
        'Sklearn',
        'Pytorch',
        'Matplotlib',
    ]
]
