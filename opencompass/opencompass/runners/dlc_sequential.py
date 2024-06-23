import datetime
import os
import os.path as osp
import re
import subprocess
import sys
import time
import traceback
from functools import partial
from multiprocessing import Pipe, Pool
from typing import Any, Dict, List, Tuple

import mmengine
from mmengine.config import ConfigDict
from tqdm import tqdm

from opencompass.registry import RUNNERS, TASKS
from opencompass.utils import get_logger

from .base import BaseRunner


@RUNNERS.register_module()
class DLCSequentialRunner(BaseRunner):
    """DLC version of slurm_sequential.SlurmSequentialRunner.

    Please refer to the documentation of `dlc.DLCRunner` and
    `slurm_sequential.SlurmSequentialRunner` for more details.
    """

    # TODO: codes below are really ugly, need to refactor it

    def __init__(self,
                 task: ConfigDict,
                 aliyun_cfg: ConfigDict,
                 max_num_workers: int = 32,
                 retry: int = 2,
                 debug: bool = False,
                 lark_bot_url: str = None):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.aliyun_cfg = aliyun_cfg
        self.max_num_workers = max_num_workers
        self.retry = retry

        logger = get_logger()
        logger.warning(
            'To ensure the integrity of the log results, the log displayed '
            f'by {self.__class__.__name__} has a 10-second delay.')

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        if not self.debug:
            return self._launch_wo_debug(tasks)
        else:
            return [self._launch(task) for task in tasks]

    def _launch_wo_debug(self,
                         tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        launched_bar = tqdm(total=len(tasks), desc='Launched')
        finished_bar = tqdm(total=len(tasks), desc='Finished')
        job_ids = []
        status = []

        def _update(result):
            finished_bar.update()
            status.append(result)
            return result

        def _err_update(err):
            finished_bar.update()
            traceback.print_exc()
            status.append(('', -1))

        try:
            parent_conns = []
            num_workers = max(min(self.max_num_workers, len(tasks)), 1)
            with Pool(processes=num_workers) as pool:
                for task in tasks:
                    parent_conn, child_conn = Pipe()
                    _ = pool.apply_async(self._launch,
                                         kwds={
                                             'cfg': task,
                                             'child_conn': child_conn
                                         },
                                         callback=_update,
                                         error_callback=_err_update)
                    time.sleep(0.5)

                    job_id = parent_conn.recv()
                    launched_bar.update()
                    parent_conns.append(parent_conn)
                    job_ids.append(job_id)

                pool.close()
                pool.join()
            return status
        except KeyboardInterrupt:
            raise
        finally:
            launched_bar.close()
            finished_bar.close()
            for parent_conn in parent_conns:
                while parent_conn.poll():
                    try:
                        job_id = parent_conn.recv()
                        job_ids.append(job_id)
                    except EOFError:
                        break
                parent_conn.close()

            for job_id in tqdm(job_ids, desc='clear sruns'):
                if job_id is None:
                    continue
                cmd = f'scancel {job_id}'
                p = subprocess.Popen(cmd,
                                     shell=True,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT)
                p.wait()

    def _launch(self, cfg: ConfigDict, child_conn: Pipe = None):
        task = TASKS.build(dict(cfg=cfg, type=self.task_cfg['type']))
        num_gpus = task.num_gpus
        task_name = task.name

        # Dump task config to file
        mmengine.mkdir_or_exist('tmp/')
        param_file = f'tmp/{os.getpid()}_params.py'
        try:
            cfg.dump(param_file)

            # Build up DLC command
            pwd = os.getcwd()
            shell_cmd = (
                f'source {self.aliyun_cfg["bashrc_path"]}; '
                f'conda activate {self.aliyun_cfg["conda_env_name"]}; '
                f'cd {pwd}; '
                '{task_cmd}')

            tmpl = ('dlc create job'
                    f" --command '{shell_cmd}'"
                    f' --name {task_name[:512]}'
                    ' --kind BatchJob'
                    f" -c {self.aliyun_cfg['dlc_config_path']}"
                    f" --workspace_id {self.aliyun_cfg['workspace_id']}"
                    ' --worker_count 1'
                    f' --worker_cpu {max(num_gpus * 6, 8)}'
                    f' --worker_gpu {num_gpus}'
                    f' --worker_memory {max(num_gpus * 64, 48)}'
                    f" --worker_image {self.aliyun_cfg['worker_image']}"
                    ' --interactive')
            get_cmd = partial(task.get_command,
                              cfg_path=param_file,
                              template=tmpl)
            cmd = get_cmd()

            logger = get_logger()
            logger.debug(f'Running command: {cmd}')

            # Run command with retry
            if self.debug:
                stdout = sys.stdout
            else:
                out_path = task.get_log_path(file_extension='out')
                mmengine.mkdir_or_exist(osp.split(out_path)[0])
                stdout = open(out_path, 'w', encoding='utf-8')

            def _run_within_retry():
                try:
                    process = subprocess.Popen(cmd,
                                               shell=True,
                                               text=True,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE)
                    job_id = None
                    job_allocated = False
                    job_finished = False
                    last_end_time = datetime.datetime.now().strftime(
                        '%Y-%m-%dT%H:%M:%SZ')
                    while True:
                        if not job_allocated:
                            line = process.stdout.readline()
                            if not line:
                                break
                            match = re.search(r'(dlc[0-9a-z]+)', line)
                            if match and job_id is None:
                                job_id = match.group(1)
                            stdout.write(line)
                            match = re.search(r'Job .* is \[Running\]', line)
                            if match:
                                job_allocated = True
                                child_conn.send(job_id)
                        else:
                            try:
                                process.wait(10)
                            except subprocess.TimeoutExpired:
                                pass
                            else:
                                job_finished = True
                            if job_finished:
                                this_end_time = datetime.datetime.now(
                                ).strftime('%Y-%m-%dT%H:%M:%SZ')
                            else:
                                this_end_time = (
                                    datetime.datetime.now() -
                                    datetime.timedelta(seconds=10)
                                ).strftime('%Y-%m-%dT%H:%M:%SZ')
                            logs_cmd = (
                                'dlc logs'
                                f' {job_id} {job_id}-worker-0'
                                f' --start_time {last_end_time}'
                                f' --end_time {this_end_time}'
                                f" -c {self.aliyun_cfg['dlc_config_path']}")
                            log_process = subprocess.Popen(
                                logs_cmd,
                                shell=True,
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
                            log_output, log_err = log_process.communicate()
                            log_output = '\n'.join(log_output.split('\n')[2:])
                            stdout.write(log_output)
                            last_end_time = this_end_time
                        stdout.flush()
                        if job_finished:
                            break
                    process.wait()
                    return process.returncode
                finally:
                    if job_id is not None:
                        cancel_cmd = (
                            'dlc stop job'
                            f' {job_id}'
                            f" -c {self.aliyun_cfg['dlc_config_path']}"
                            ' -f')
                        subprocess.run(cancel_cmd,
                                       shell=True,
                                       text=True,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)

            return_code = _run_within_retry()
            retry = self.retry
            output_paths = task.get_output_paths()
            while self._job_failed(return_code, output_paths) and retry > 0:
                retry -= 1
                cmd = get_cmd()
                return_code = _run_within_retry()
        finally:
            # Clean up
            if child_conn is not None:
                child_conn.send(None)
                child_conn.close()
            os.remove(param_file)

        return task_name, return_code

    def _job_failed(self, return_code: int, output_paths: List[str]) -> bool:
        return return_code != 0 or not all(
            osp.exists(output_path) for output_path in output_paths)
