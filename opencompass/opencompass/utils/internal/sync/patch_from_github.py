"""A script that converts each commit in a branch to a patch file and applies
it to another branch.

"""
import argparse
import subprocess
import fnmatch
import os

from git import Actor, Repo
import sys

sys.path.append('opencompass/utils')
from lark import LarkReporter


def parse_args():
    parser = argparse.ArgumentParser(description='Converts each commit in src'
                                     'branch to a patch file and applies it to'
                                     'tgt branch. The starting commit is '
                                     'specified in .github_commit file. '
                                     'If any commit touches a'
                                     'blacklisted file in .github_blacklist, '
                                     'the script will exit.')
    parser.add_argument('src_branch', help='source branch')
    parser.add_argument('tgt_branch', help='target branch')
    parser.add_argument('lark', help='lark url')
    parser.add_argument('--manual', action='store_true', help='manual mode')
    args = parser.parse_args()
    return args

def patch_branch(source_branch, target_branch, lark_url, repo_path='.', manual=False):
    status = 0

    # Load the commit hash and blacklisted files
    with open(os.path.join(repo_path, '.github_commit'), 'r') as f:
        commit_hash = f.read().strip()

    with open(os.path.join(repo_path, '.github_blacklist'), 'r') as f:
        blacklisted_patterns = f.read().splitlines()

    # Initialize the repository
    repo = Repo(repo_path)
    repo.config_writer().set_value("user", "name", "bot").release()
    repo.config_writer().set_value("user", "email", "bot@bot.com").release()

    # Get the commits from .github_commit to the latest commit of "github" branch
    latest_commit = list(repo.iter_commits(source_branch, max_count=1))[0].hexsha
    commits = list(repo.iter_commits(f'{commit_hash}..{latest_commit}'))[::-1]

    if len(commits) == 0:
        exit(0)

    updated = False

    for commit in commits:
        print(f'Applying {commit.hexsha}...')
        # Check if commit touches any blacklisted file
        for file in commit.stats.files:
            if any(fnmatch.fnmatch(file, pattern) for pattern in blacklisted_patterns):
                status = f'{file} in commit {commit.hexsha} touches a blacklisted pattern. Exiting...'
                break
        if status != 0:
            break

        # Create a patch for the commit
        patch_files = repo.git.format_patch('-1', commit.hexsha).split('\n')

        # Switch to "dev" branch
        repo.git.checkout(target_branch)

        # Try to apply the patch
        for patch_file in patch_files:
            try:
                repo.git.am(patch_file, '--3way')
            except Exception as e:
                if commit.message.lower().startswith('[sync]'):
                    repo.git.am('--abort')
                    continue
                else:
                    if not manual:
                        repo.git.am('--abort')
                    status = f'Error applying patch for commit {commit.hexsha}. Exiting...'
                    status += '\n' + str(e)
                    break
            os.remove(patch_file)
        if status == 0 or manual:
            # Update .github_commit file
            with open(os.path.join(repo_path, '.github_commit'), 'w') as f:
                f.write(commit.hexsha + '\n')
        if status != 0:
            break
        else:
            updated = True

    if updated:
        actor = Actor("bot", "bot@bot.com")
        repo.index.add('.github_commit')
        repo.index.commit('Update .github_commit', author=actor, committer=actor, skip_hooks=True)

    if status == 0:
        print('All patches applied successfully!')
        if not manual:
            lark = LarkReporter(lark_url)
            lark.post(title='Gitlab 版本已顺利同步 Github 更改！', content=f'同步至 {commit.hexsha}\n{commit.message}')
    else:
        print(status)
        if not manual:
            lark = LarkReporter(lark_url)
            lark.post(title='Gitlab 版本同步 Github 失败', content=f'错误信息：{status}\n\n同步至 {commit.hexsha}\n{commit.message}')


if __name__ == '__main__':
    args = parse_args()
    patch_branch(args.src_branch, args.tgt_branch, args.lark, manual=args.manual)
