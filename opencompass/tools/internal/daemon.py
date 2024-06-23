import argparse
import getpass
import queue
import subprocess
import threading
import time
import uuid
from tempfile import NamedTemporaryFile
from typing import List

import mmengine
from flask import Flask, jsonify, request

app = Flask(__name__)

PROTOCOL_VERSION = '1.0'
task_queue = queue.Queue()
tasks = {}  # To store the status and output of tasks
tasks_lock = threading.Lock()
daemon_user = getpass.getuser()


def parse_args():
    parser = argparse.ArgumentParser(description='OpenCompass Evaluation '
                                     'service.')
    parser.add_argument('-p',
                        '--port',
                        default=10000,
                        type=int,
                        help='The '
                        'port that receives requests')
    parser.add_argument('-n',
                        '--nworkers',
                        default=4,
                        type=int,
                        help='The '
                        'maximum number of tasks to run in parallel')
    return parser.parse_args()


def get_tasks_with_prefix(prefix, username):
    # Use squeue to list jobs, filtering by user and output format for job ID
    # and job name
    cmd = [
        'squeue',
        '-u',
        username,
        '--format=%i|%j',  # %i for Job ID and %j for Job Name
        '--noheader'  # Exclude the header in the output
    ]

    try:
        result = subprocess.check_output(cmd, text=True)
        # Extract job ids with the given prefix
        task_names, task_ids = [], []
        for line in result.splitlines():
            if line.split('|')[1].startswith(prefix):
                task_names.append(line.split('|')[1])
                task_ids.append(line.split('|')[0])
        return task_ids, task_names
    except subprocess.CalledProcessError:
        print('Error executing squeue command.')
        return [], []


def kill_tasks_by_ids(task_ids: List[str]):
    for id, _ in zip(
            *get_tasks_with_prefix(prefix=task_ids, username=daemon_user)):
        subprocess.run(['scancel', id])
        time.sleep(0.2)


@app.before_request
def verify_request():
    """Verify request by the protocol version and tokens (to be implemented)"""
    if request.method == 'POST':
        protocol_version = request.json.get('protocol_version')
        if protocol_version != PROTOCOL_VERSION:
            return jsonify({'error': 'Invalid protocol version'}), 400


@app.route('/clear', methods=['GET'])
def kill_dangling_tasks():
    kill_ids = []
    for id, name in zip(
            *get_tasks_with_prefix(prefix='', username=daemon_user)):
        if not any(name.startswith(k) for k in tasks.keys()):
            kill_ids.append(id)
    for id in kill_ids:
        subprocess.run(['scancel', id])
        time.sleep(0.2)
    return jsonify({'Killed': kill_ids})


def worker():
    while True:
        task_id, code_content = task_queue.get()
        with tasks_lock:
            tasks[task_id] = {'status': 'running'}
        with NamedTemporaryFile('w', suffix='.py', dir='configs/') as f:
            f.write(f'task_id = "{task_id}"\n')
            f.write(code_content)
            f.flush()
            mmengine.mkdir_or_exist(f'outputs/daemon/{task_id}/')
            output = f'outputs/daemon/{task_id}/terminal.out'
            task_output = open(output, 'w', encoding='utf-8')
            try:
                result = subprocess.run(' '.join([
                    'python', 'run.py', f.name, '--slurm', '-p', 'llmeval',
                    '-q', 'auto', '-w', f'outputs/daemon/{task_id}'
                ]),
                                        shell=True,
                                        text=True,
                                        stdout=task_output,
                                        stderr=task_output)
                with tasks_lock:
                    tasks[task_id] = {'status': 'exited', 'output': output}
            except subprocess.CalledProcessError as e:
                result = e.output.decode()
                with tasks_lock:
                    tasks[task_id] = {
                        'status': 'error',
                        'output': output,
                        'error': result
                    }
            finally:
                kill_tasks_by_ids(task_id)
                task_queue.task_done()


@app.route('/execute', methods=['POST'])
def execute():
    data = request.json
    protocol_version = data.get('protocol_version')
    code_content = data.get('python_file')

    if protocol_version != PROTOCOL_VERSION:
        return jsonify({'error': 'Invalid protocol version'}), 400

    task_id = str(uuid.uuid4())
    with tasks_lock:
        tasks[task_id] = {'status': 'queueing', 'output': ''}
    task_queue.put((task_id, code_content))

    return jsonify({'task_id': task_id})


@app.route('/tasks/list', methods=['GET'])
def list_tasks():
    # List tasks that are either running or queueing
    relevant_tasks = {
        task_id: task_info
        for task_id, task_info in tasks.items()
        if task_info['status'] in ['queueing', 'running']
    }
    return jsonify(relevant_tasks)


@app.route('/tasks/kill', methods=['POST'])
def kill_task():
    data = request.json
    task_id = data.get('task_id')
    kill_tasks_by_ids([task_id])
    with tasks_lock:
        tasks[task_id]['status'] = 'killed'
    return jsonify(tasks[task_id])


@app.route('/tasks/<task_id>', methods=['GET'])
def get_task_status(task_id):
    task_info = tasks.get(task_id)
    if task_info:
        return jsonify(task_info)
    else:
        return jsonify({'error': 'Task not found'}), 404


if __name__ == '__main__':
    args = parse_args()

    # Start the worker threads
    for _ in range(args.nworkers):
        t = threading.Thread(target=worker, daemon=True)
        t.start()

    app.run(host='0.0.0.0', port=args.port)
