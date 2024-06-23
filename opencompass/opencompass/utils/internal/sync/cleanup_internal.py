import fnmatch
import os
import re
import subprocess

def clean(script):
    # Remove blocks between # INTERNAL_BEGIN and # INTERNAL_END
    pattern_blocks = r'[ \t]*?# INTERNAL_BEGIN.*?# INTERNAL_END.*?\n'
    cleaned_script = re.sub(pattern_blocks, '', script, flags=re.DOTALL)

    # Uncomment blocks between # EXTERNAL_BEGIN and # EXTERNAL_END
    pattern_external_blocks = r'# EXTERNAL_BEGIN(.*?)# EXTERNAL_END'
    # Find all blocks
    blocks = re.findall(pattern_external_blocks, cleaned_script, flags=re.DOTALL)
    # Uncomment each block and replace it in the script
    for block in blocks:
        uncommented_block = '\n'.join(line[2:] if line.startswith("# ") else line for line in block.split('\n'))
        cleaned_script = cleaned_script.replace(block, uncommented_block)

    # Remove lines with # EXTERNAL_BEGIN or # EXTERNAL_END
    pattern_internal = r'.*# EXTERNAL_BEGIN.*\n'
    cleaned_script = re.sub(pattern_internal, '', cleaned_script)
    pattern_internal = r'.*# EXTERNAL_END.*\n'
    cleaned_script = re.sub(pattern_internal, '', cleaned_script)

    # Remove lines with # INTERNAL
    pattern_internal = r'.*# INTERNAL.*\n'
    cleaned_script = re.sub(pattern_internal, '', cleaned_script)

    # Uncomment lines with # EXTERNAL
    pattern_external = r'# EXTERNAL '
    cleaned_script = re.sub(pattern_external, '', cleaned_script)

    return cleaned_script

basepath = '.'

cmd = 'git ls-files'
p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=basepath)
output, errors = p.communicate()
lines = output.splitlines()
filenames = [os.path.join(basepath, line.decode('utf-8').strip()) for line in lines]

with open(".github_blacklist", "r") as f:
    internal_patterns = f.readlines()
internal_patterns = [os.path.join(basepath, pattern.strip()) for pattern in internal_patterns]

for filename in filenames:
    if any(fnmatch.fnmatch(filename, ip) for ip in internal_patterns):
        # delete the file
        print("deleting file: {}".format(filename))
        os.remove(filename)
        continue

    with open(filename, "r") as f:
        # lines = f.readlines()
        content = f.read()
    with open(filename, "w") as f:
        f.write(clean(content))
