from opencompass.partitioners import SizePartitioner, NaivePartitioner, InferTimePartitioner
from opencompass.runners import SlurmSequentialRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

# if users want to use other partition, use `-p {PARTITION}` 
# in the command line can change the partition.

# new infer setting, using split strategy, may speed up the infer time
infer = dict(
    partitioner=dict(
        type=InferTimePartitioner,
        max_task_time=3600,
        strategy='split'),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=64,
        retry=4,
        partition='llmit2',
        quotatype='auto',
        task=dict(type=OpenICLInferTask)),
)

# origin infer setting
infer_size = dict(
    partitioner=dict(
        type=SizePartitioner, 
        max_task_size=5000,     # default = 2000
        gen_task_coef=20,
    ),
    runner=dict(
        type=SlurmSequentialRunner,
        max_num_workers=64,
        retry=4,
        partition='llmit2',
        quotatype='auto',
        task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(
        type=SlurmSequentialRunner,
        partition='llmit2',
        quotatype='auto',
        max_num_workers=128,
        retry=2,
        task=dict(type=OpenICLEvalTask)),
)
