import argparse

from opencompass.utils.internal.result_uploader import DatabaseReporter

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('work_dir', type=str)
    parser.add_argument('-u', '--url', type=str, default='product')
    parser.add_argument('--retry', type=int, default=2)
    parser.add_argument('-p', '--nproc', type=int, default=32)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        default='result',
        choices=['all', 'meta', 'result', 'prediction', 'raw_dataset'])
    parser.add_argument('-c', '--config-name', type=str, default=None)

    args = parser.parse_args()

    if args.url == 'product':
        index_url = 'http://inferstore-eval.openxlab.org.cn'
    elif args.url == 'staging':
        index_url = 'http://106.14.134.80:10824'
    else:
        index_url = args.url

    reporter = DatabaseReporter(
        url=index_url,
        work_dir=args.work_dir,
        uploaded_result_path=None,
        retry=args.retry,
        nproc=args.nproc,
        debug=args.debug,
        dry_run=args.dry_run,
        config_name=args.config_name,
    )

    if args.mode in ['all', 'meta', 'result', 'prediction']:
        reporter.run_answer_dataset_id_mapping()
    if args.mode in ['all', 'result']:
        reporter.run_result()
    if args.mode in ['all', 'prediction']:
        reporter.run_prediction()
    if args.mode in ['raw_dataset']:
        reporter.run_raw_dataset()
