import argparse
import json

import requests

BASE_URL = 'http://127.0.0.1:10000'  # Change to your daemon's IP and port
PROTOCOL_VERSION = '1.0'


def post_execute(filename: str):
    with open(filename, 'r') as file:
        python_file_content = file.read()
    payload = {
        'protocol_version': PROTOCOL_VERSION,
        'python_file': python_file_content
    }
    response = requests.post(f'{BASE_URL}/execute', json=payload)
    if response.status_code == 200:
        print('Task created with ID:', response.json()['task_id'])
    else:
        print(response.json()['error'])


def list_tasks():
    response = requests.get(f'{BASE_URL}/tasks/list')
    print(response)
    tasks = response.json()
    if tasks:
        print(json.dumps(tasks, indent=2))
    else:
        print('No tasks found.')


def kill_task(task_id: str):
    payload = {
        'task_id': task_id,
        'protocol_version': PROTOCOL_VERSION,
    }
    response = requests.post(f'{BASE_URL}/tasks/kill', json=payload)
    if response.status_code == 200:
        print('Task killed:', response.json())
    else:
        print(response.json()['error'])


def clear_tasks():
    response = requests.get(f'{BASE_URL}/clear')
    killed = response.json().get('Killed', [])
    if killed:
        print('Tasks killed:', ', '.join(killed))
    else:
        print('No tasks to clear.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCompass Daemon Client')
    subparsers = parser.add_subparsers(dest='command')

    # Execute command
    execute_parser = subparsers.add_parser('run')
    execute_parser.add_argument('filename',
                                type=str,
                                help='Path to the python config file')

    # List tasks command
    list_parser = subparsers.add_parser('tasks')
    list_parser.add_argument('action', choices=['list', 'kill', 'clear'])
    list_parser.add_argument('task_id',
                             type=str,
                             nargs='?',
                             default='',
                             help='Task ID (for killing specific task)')

    args = parser.parse_args()

    if args.command == 'execute':
        post_execute(args.filename)
    elif args.command == 'tasks':
        if args.action == 'list':
            list_tasks()
        elif args.action == 'kill':
            if not args.task_id:
                print("Error: TASK_ID is required for 'kill' action.")
            else:
                kill_task(args.task_id)
        elif args.action == 'clear':
            clear_tasks()
