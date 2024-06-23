import argparse
import copy  # INTERNAL
import getpass
import os
import os.path as osp
import time  # INTERNAL
from datetime import datetime

from mmengine.config import Config, DictAction

from opencompass.partitioners import MultimodalNaivePartitioner
from opencompass.registry import PARTITIONERS, RUNNERS, build_from_cfg
from opencompass.runners import SlurmRunner
from opencompass.summarizers import DefaultSummarizer
from opencompass.utils import LarkReporter, get_logger
from opencompass.utils.run import (exec_mm_infer_runner, fill_eval_cfg,
                                   fill_infer_cfg, get_config_from_arg)


def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('config', nargs='?', help='Train config file path')

    # add mutually exclusive args `--slurm` `--dlc`, defaults to local runner
    # if "infer" or "eval" not specified
    launch_method = parser.add_mutually_exclusive_group()
    launch_method.add_argument('--slurm',
                               action='store_true',
                               default=False,
                               help='Whether to force tasks to run with srun. '
                               'If True, `--partition(-p)` must be set. '
                               'Defaults to False')
    launch_method.add_argument('--dlc',
                               action='store_true',
                               default=False,
                               help='Whether to force tasks to run on dlc. If '
                               'True, `--aliyun-cfg` must be set. Defaults'
                               ' to False')
    # multi-modal support
    parser.add_argument('--mm-eval',
                        help='Whether or not enable multimodal evaluation',
                        action='store_true',
                        default=False)
    # Add shortcut parameters (models, datasets and summarizer)
    parser.add_argument('--models', nargs='+', help='', default=None)
    parser.add_argument('--datasets', nargs='+', help='', default=None)
    parser.add_argument('--summarizer', help='', default=None)
    # add general args
    parser.add_argument('--debug',
                        help='Debug mode, in which scheduler will run tasks '
                        'in the single process, and output will not be '
                        'redirected to files',
                        action='store_true',
                        default=False)
    parser.add_argument('--dry-run',
                        help='Dry run mode, in which the scheduler will not '
                        'actually run the tasks, but only print the commands '
                        'to run',
                        action='store_true',
                        default=False)
    parser.add_argument('-m',
                        '--mode',
                        help='Running mode. You can choose "infer" if you '
                        'only want the inference results, or "eval" if you '
                        'already have the results and want to evaluate them, '
                        'or "viz" if you want to visualize the results.',
                        choices=['all', 'infer', 'eval', 'viz'],
                        default='all',
                        type=str)
    parser.add_argument('-r',
                        '--reuse',
                        nargs='?',
                        type=str,
                        const='latest',
                        help='Reuse previous outputs & results, and run any '
                        'missing jobs presented in the config. If its '
                        'argument is not specified, the latest results in '
                        'the work_dir will be reused. The argument should '
                        'also be a specific timestamp, e.g. 20230516_144254'),
    parser.add_argument('-w',
                        '--work-dir',
                        help='Work path, all the outputs will be '
                        'saved in this path, including the slurm logs, '
                        'the evaluation results, the summary results, etc.'
                        'If not specified, the work_dir will be set to '
                        './outputs/default.',
                        default=None,
                        type=str)
    parser.add_argument(
        '--config-dir',
        default='configs',
        help='Use the custom config directory instead of config/ to '
        'search the configs for datasets, models and summarizers',
        type=str)
    parser.add_argument('-l',
                        '--lark',
                        help='Report the running status to lark bot',
                        action='store_true',
                        default=False)
    parser.add_argument('--max-partition-size',
                        help='The maximum size of an infer task. Only '
                        'effective when "infer" is missing from the config.',
                        type=int,
                        default=40000),
    parser.add_argument(
        '--gen-task-coef',
        help='The dataset cost measurement coefficient for generation tasks, '
        'Only effective when "infer" is missing from the config.',
        type=int,
        default=20)
    parser.add_argument('--max-num-workers',
                        help='Max number of workers to run in parallel. '
                        'Will be overrideen by the "max_num_workers" argument '
                        'in the config.',
                        type=int,
                        default=32)
    parser.add_argument('--max-workers-per-gpu',
                        help='Max task to run in parallel on one GPU. '
                        'It will only be used in the local runner.',
                        type=int,
                        default=1)
    parser.add_argument(
        '--retry',
        help='Number of retries if the job failed when using slurm or dlc. '
        'Will be overrideen by the "retry" argument in the config.',
        type=int,
        default=2)
    parser.add_argument(
        '--dump-eval-details',
        help='Whether to dump the evaluation details, including the '
        'correctness of each sample, bpb, etc.',
        action='store_true',
    )
    # INTERNAL_BEGIN
    parser.add_argument(
        '--wait',
        help='Wait and check whether model paths exist, then run the task.',
        action='store_true',
    )
    parser.add_argument(
        '-s',
        '--skip-upload',
        help='Skip uploading the results to the inferstore database',
        action='store_true',
    )
    # INTERNAL_END
    # set srun args
    slurm_parser = parser.add_argument_group('slurm_args')
    parse_slurm_args(slurm_parser)
    # set dlc args
    dlc_parser = parser.add_argument_group('dlc_args')
    parse_dlc_args(dlc_parser)
    # set hf args
    hf_parser = parser.add_argument_group('hf_args')
    parse_hf_args(hf_parser)
    # set custom dataset args
    custom_dataset_parser = parser.add_argument_group('custom_dataset_args')
    parse_custom_dataset_args(custom_dataset_parser)
    args = parser.parse_args()
    if args.slurm:
        assert args.partition is not None, (
            '--partition(-p) must be set if you want to use slurm')
    if args.dlc:
        assert os.path.exists(args.aliyun_cfg), (
            'When launching tasks using dlc, it needs to be configured '
            'in "~/.aliyun.cfg", or use "--aliyun-cfg $ALiYun-CFG_Path"'
            ' to specify a new path.')
    return args


def parse_slurm_args(slurm_parser):
    """These args are all for slurm launch."""
    slurm_parser.add_argument('-p',
                              '--partition',
                              help='Slurm partition name',
                              default=None,
                              type=str)
    slurm_parser.add_argument('-q',
                              '--quotatype',
                              help='Slurm quota type',
                              default=None,
                              type=str)
    slurm_parser.add_argument('--qos',
                              help='Slurm quality of service',
                              default=None,
                              type=str)


def parse_dlc_args(dlc_parser):
    """These args are all for dlc launch."""
    dlc_parser.add_argument('--aliyun-cfg',
                            help='The config path for aliyun config',
                            default='~/.aliyun.cfg',
                            type=str)


# INTERNAL_BEGIN
def check_model_accessibility(model_cfgs, logger):
    from mmengine.fileio import exists

    from opencompass.models.internal.llama import LLama
    from opencompass.models.internal.pjlm import LLM, LLMv2, LLMv3
    from opencompass.utils.internal.test.proxy import proxy_off, proxy_on
    proxy_off()
    for cfg in model_cfgs:
        if cfg.type in [LLM, LLMv2, LLMv3, LLama]:
            paths_to_check = [cfg.path]
            if 'tokenizer_path' in cfg:
                paths_to_check.append(cfg.tokenizer_path)
            for path in paths_to_check:
                if exists(path):
                    logger.info(f'{path} is accessible')
                else:
                    logger.error(
                        f'{path} is not accessible, please check the path '
                        'or if you are granted the permission to access.')
                    exit(-1)
    proxy_on()


def wait_and_check_model_accessibility(cfg, logger):
    """Wait and check model path accessibility.

    If has `wait` in cfg, first will wait `wait_time` minutes to begin run
    the script, then will check the accessibility of model path every
    `check_every` minutes. If all paths are accessible, will wait
    `wait_for_run` minutes to begin run. If waiting `max_waiting_time`
    minutes, but some model path are still not accessible will
    exit the script. If not specify `wait` in cfg, will only check the
    accessibility of model path.

    Demo in config:

    ```
    wait = dict(
        wait_time=0,
        check_every=2,
        max_waiting_time=30,
        wait_for_run=2,
    )
    ```
    """
    from mmengine.config import ConfigDict
    from mmengine.fileio import exists

    from opencompass.models.internal.intern_model import InternLMwithModule
    from opencompass.models.internal.llama import LLama
    from opencompass.models.internal.pjlm import LLM, LLMv2, LLMv3
    from opencompass.models.lagent import CodeAgent, LagentAgent
    from opencompass.utils.internal.test.proxy import proxy_off, proxy_on

    proxy_off()
    paths_to_check = []
    for m in cfg.models:
        if m.type in [LLM, LLMv2, LLMv3, LLama, InternLMwithModule]:
            # normal path
            paths_to_check.append(m.path)
        elif m.type in [CodeAgent, LagentAgent]:
            # agent path
            paths_to_check.append(m.llm.path)

    paths_to_check = set(paths_to_check)
    wait_cfg = cfg.get('wait', None)
    if wait_cfg is None:
        # if not specify `wait` in cfg, will
        # only check the accessibility of model path
        logger.warning('No `wait` setting in the config, will only check the '
                       'accessibility of model path.')
        wait_cfg = ConfigDict(
            wait_time=0,
            check_every=0.1,
            max_waiting_time=0.1,
            wait_for_run=0,
        )

    # get wait config, all time is in minutes
    wait_time = wait_cfg.get('wait_time', 0)
    check_every = wait_cfg.get('check_every', 0.1)
    wait_for_run = wait_cfg.get('wait_for_run', 0)
    max_waiting_time = wait_cfg.get('max_waiting_time', 0.1)

    # wait for `wait_time` minutes to begin run the script
    logger.info(f'Wait {wait_time} minutes to begin run the script.')
    time.sleep(wait_time * 60)

    # begin check the accessibility of model path
    logger.info('Begin check the accessibility of model path.')

    paths_exist = [False for _ in paths_to_check]
    all_exist = False

    start_time = time.time()
    current_time = time.time()
    while not all_exist:
        for i, path in enumerate(paths_to_check):
            if exists(path):
                paths_exist[i] = True
        all_exist = all(paths_exist)
        if all_exist:
            # if all model path are accessible, will wait
            # `wait_for_run` minutes
            logger.info('All model path are accessible, wait '
                        f'{wait_for_run} minutes to begin run the script.')
            time.sleep(wait_for_run * 60)
        else:
            # if not all model path are accessible, will wait
            # `check_every` minutes to check again
            not_exist_paths = [
                p for i, p in enumerate(paths_to_check) if not paths_exist[i]
            ]
            not_exist_paths_str = '\n '.join(not_exist_paths)
            logger.info('Some model path are not accessible. \n'
                        'Not exists models path:\n '
                        f'{not_exist_paths_str} \n'
                        f'wait {check_every} minutes to check again.')
            time.sleep(check_every * 60)

        # if waiting `max_waiting_time` minutes, but some model path are
        # still not accessible will exit the script
        current_time = time.time()
        waiting_time = current_time - start_time
        if waiting_time > (max_waiting_time * 60) and not all_exist:
            logger.error(f'Waited for {max_waiting_time} minutes, '
                         'but some model paths are still not accessible, '
                         'please check the path or if you are granted '
                         'the permission to access.')
            exit(-1)
    proxy_on()


# INTERNAL_END


def parse_hf_args(hf_parser):
    """These args are all for the quick construction of HuggingFace models."""
    hf_parser.add_argument('--hf-path', type=str)
    hf_parser.add_argument('--peft-path', type=str)
    hf_parser.add_argument('--tokenizer-path', type=str)
    hf_parser.add_argument('--model-kwargs',
                           nargs='+',
                           action=DictAction,
                           default={})
    hf_parser.add_argument('--tokenizer-kwargs',
                           nargs='+',
                           action=DictAction,
                           default={})
    hf_parser.add_argument('--max-out-len', type=int)
    hf_parser.add_argument('--max-seq-len', type=int)
    hf_parser.add_argument('--no-batch-padding',
                           action='store_true',
                           default=False)
    hf_parser.add_argument('--batch-size', type=int)
    hf_parser.add_argument('--num-gpus', type=int)
    hf_parser.add_argument('--pad-token-id', type=int)


def parse_custom_dataset_args(custom_dataset_parser):
    """These args are all for the quick construction of custom datasets."""
    custom_dataset_parser.add_argument('--custom-dataset-path', type=str)
    custom_dataset_parser.add_argument('--custom-dataset-meta-path', type=str)
    custom_dataset_parser.add_argument('--custom-dataset-data-type',
                                       type=str,
                                       choices=['mcq', 'qa'])
    custom_dataset_parser.add_argument('--custom-dataset-infer-method',
                                       type=str,
                                       choices=['gen', 'ppl'])


def main():
    args = parse_args()
    if args.dry_run:
        args.debug = True
    # initialize logger
    logger = get_logger(log_level='DEBUG' if args.debug else 'INFO')

    cfg = get_config_from_arg(args)
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', './outputs/default/')
    # INTERNAL_BEGIN
    if args.mode in ['all', 'infer']:
        if not args.wait:
            check_model_accessibility(cfg.models, logger)
        else:
            wait_and_check_model_accessibility(cfg, logger)
    # INTERNAL_END

    # cfg_time_str defaults to the current time
    cfg_time_str = dir_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.reuse:
        if args.reuse == 'latest':
            if not os.path.exists(cfg.work_dir) or not os.listdir(
                    cfg.work_dir):
                logger.warning('No previous results to reuse!')
            else:
                dirs = os.listdir(cfg.work_dir)
                dir_time_str = sorted(dirs)[-1]
        else:
            dir_time_str = args.reuse
        logger.info(f'Reusing experiements from {dir_time_str}')
    elif args.mode in ['eval', 'viz']:
        raise ValueError('You must specify -r or --reuse when running in eval '
                         'or viz mode!')

    # update "actual" work_dir
    cfg['work_dir'] = osp.join(cfg.work_dir, dir_time_str)
    os.makedirs(osp.join(cfg.work_dir, 'configs'), exist_ok=True)

    # dump config
    output_config_path = osp.join(cfg.work_dir, 'configs',
                                  f'{cfg_time_str}.py')
    cfg.dump(output_config_path)
    # Config is intentally reloaded here to avoid initialized
    # types cannot be serialized
    cfg = Config.fromfile(output_config_path, format_python_code=False)

    # report to lark bot if specify --lark
    if not args.lark:
        cfg['lark_bot_url'] = None
    elif cfg.get('lark_bot_url', None):
        # EXTERNAL content = f'{getpass.getuser()}\'s task has been launched!'
        # INTERNAL_BEGIN
        models = copy.deepcopy(cfg.models)
        total_model = len(models)
        model_abbr = []
        for model in models:
            model_abbr.append(model.abbr)
        text_model_abbr = ', '.join(model_abbr)
        content = f'{getpass.getuser()}新的测试任务已启动！\n' \
                  f'配置文件为：{args.config}\n'\
                  f'共有{total_model}个模型待测试，分别是：{text_model_abbr}\n'\
                  f'每个模型检测{len(cfg.datasets)}个测试数据。'
        # INTERNAL_END
        LarkReporter(cfg['lark_bot_url']).post(content)

    if args.mode in ['all', 'infer']:
        # When user have specified --slurm or --dlc, or have not set
        # "infer" in config, we will provide a default configuration
        # for infer
        if (args.dlc or args.slurm) and cfg.get('infer', None):
            logger.warning('You have set "infer" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "infer" configuration will be overridden by '
                           'your runtime arguments.')
        # Check whether run multimodal evaluation
        if args.mm_eval:
            partitioner = MultimodalNaivePartitioner(
                osp.join(cfg['work_dir'], 'predictions/'))
            tasks = partitioner(cfg)
            exec_mm_infer_runner(tasks, args, cfg)
            return

        if args.dlc or args.slurm or cfg.get('infer', None) is None:
            fill_infer_cfg(cfg, args)

        if args.partition is not None:
            if RUNNERS.get(cfg.infer.runner.type) == SlurmRunner:
                cfg.infer.runner.partition = args.partition
                cfg.infer.runner.quotatype = args.quotatype
        else:
            logger.warning('SlurmRunner is not used, so the partition '
                           'argument is ignored.')
        if args.debug:
            cfg.infer.runner.debug = True
        if args.lark:
            cfg.infer.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.infer.partitioner['out_dir'] = osp.join(cfg['work_dir'],
                                                    'predictions/')
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.infer.runner)
        # Add extra attack config if exists
        if hasattr(cfg, 'attack'):
            for task in tasks:
                cfg.attack.dataset = task.datasets[0][0].abbr
                task.attack = cfg.attack
        runner(tasks)

    # evaluate
    if args.mode in ['all', 'eval']:
        # When user have specified --slurm or --dlc, or have not set
        # "eval" in config, we will provide a default configuration
        # for eval
        if (args.dlc or args.slurm) and cfg.get('eval', None):
            logger.warning('You have set "eval" in the config, but '
                           'also specified --slurm or --dlc. '
                           'The "eval" configuration will be overridden by '
                           'your runtime arguments.')

        if args.dlc or args.slurm or cfg.get('eval', None) is None:
            fill_eval_cfg(cfg, args)
        if args.dump_eval_details:
            cfg.eval.runner.task.dump_details = True

        if args.partition is not None:
            if RUNNERS.get(cfg.eval.runner.type) == SlurmRunner:
                cfg.eval.runner.partition = args.partition
                cfg.eval.runner.quotatype = args.quotatype
            else:
                logger.warning('SlurmRunner is not used, so the partition '
                               'argument is ignored.')
        if args.debug:
            cfg.eval.runner.debug = True
        if args.lark:
            cfg.eval.runner.lark_bot_url = cfg['lark_bot_url']
        cfg.eval.partitioner['out_dir'] = osp.join(cfg['work_dir'], 'results/')
        partitioner = PARTITIONERS.build(cfg.eval.partitioner)
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.eval.runner)
        runner(tasks)

    # visualize
    if args.mode in ['all', 'eval', 'viz']:
        summarizer_cfg = cfg.get('summarizer', {})
        if not summarizer_cfg or summarizer_cfg.get('type', None) is None:
            summarizer_cfg['type'] = DefaultSummarizer
        summarizer_cfg['config'] = cfg
        summarizer = build_from_cfg(summarizer_cfg)
        summarizer.summarize(time_str=cfg_time_str)

    # INTERNAL_BEGIN
    # upload results
    if args.mode in ['all'] and not args.debug and not args.skip_upload:
        from opencompass.utils.internal.result_uploader import DatabaseReporter
        reporter = DatabaseReporter(work_dir=cfg['work_dir'])
        reporter.run_answer_dataset_id_mapping()
        reporter.run_result()
    # INTERNAL_END


if __name__ == '__main__':
    main()
