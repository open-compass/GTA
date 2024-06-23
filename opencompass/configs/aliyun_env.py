import getpass

from opencompass.runners import DLCRunner
from opencompass.partitioners import SizePartitioner, NaivePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask


# 大集群
llmeval_infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40000),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=dict(
            bashrc_path=f"/cpfs01/user/{getpass.getuser()}/.bashrc",  # bashrc 路径
            conda_env_name='opencompass',  # conda 环境名
            dlc_config_path=f"/cpfs01/user/{getpass.getuser()}/.dlc/config.llmeval",  # dlc 配置文件路径
            workspace_id='ws28twxwbsauodwa',  # 工作空间 id, 见 pai 网页
            worker_image='pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/pytorch:py3.6-torch1.8-cuda11.1-rdma5.2-sshd-ubuntu18.04',  # 默认镜像名，见 pai 网页
        ),
        task=dict(type=OpenICLInferTask)),
)

llmeval_eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=dict(
            bashrc_path=f"/cpfs01/user/{getpass.getuser()}/.bashrc",  # bashrc 路径
            conda_env_name='opencompass',  # conda 环境名
            dlc_config_path=f"/cpfs01/user/{getpass.getuser()}/.dlc/config.llmeval",  # dlc 配置文件路径
            workspace_id='ws28twxwbsauodwa',  # 工作空间 id, 见 pai 网页
            worker_image='pjlab-wulan-acr-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pjlab-eflops/pytorch:py3.6-torch1.8-cuda11.1-rdma5.2-sshd-ubuntu18.04',  # 默认镜像名，见 pai 网页
        ),
        task=dict(type=OpenICLEvalTask)),
)

# 小集群
llm_infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40000),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=dict(
            bashrc_path=f"/cpfs01/user/{getpass.getuser()}/.bashrc",  # bashrc 路径
            conda_env_name='opencompass',  # conda 环境名
            dlc_config_path=f"/cpfs01/user/{getpass.getuser()}/.dlc/config",  # dlc 配置文件路径
            workspace_id='ws18sdj44um64lxi',  # 工作空间 id, 见 pai 网页
            worker_image='master0:5000/eflops/pytorch:py3.6-torch1.8-cuda11.1-rdma5.2-sshd-ubuntu18.04',  # 默认镜像名，见 pai 网页
        ),
        task=dict(type=OpenICLInferTask)),
)

llm_eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=DLCRunner,
        aliyun_cfg=dict(
            bashrc_path=f"/cpfs01/user/{getpass.getuser()}/.bashrc",  # bashrc 路径
            conda_env_name='opencompass',  # conda 环境名
            dlc_config_path=f"/cpfs01/user/{getpass.getuser()}/.dlc/config",  # dlc 配置文件路径
            workspace_id='ws18sdj44um64lxi',  # 工作空间 id, 见 pai 网页
            worker_image='master0:5000/eflops/pytorch:py3.6-torch1.8-cuda11.1-rdma5.2-sshd-ubuntu18.04',  # 默认镜像名，见 pai 网页
        ),
        task=dict(type=OpenICLEvalTask)),
)
