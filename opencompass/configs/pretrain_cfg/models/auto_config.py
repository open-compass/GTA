import os

models = """

"""

max_out_len = 100
max_seq_len = 2048
batch_size = 8
num_gpus = 1
num_procs = 1
output_dir = "model_01_08"
mode="none"

def get_normal_model(abbr, folder, step, tokenizer, module_path, model_config,
                     model_type, max_out_len, max_seq_len, batch_size,
                     num_gpus, num_procs, dtype, submit_time, mode):
    return "    " + f"dict(" + \
            f"\n        abbr=\"{abbr}_{step}\"," + \
            f"\n        type=\"opencompass.models.internal.InternLMwithModule\"," + \
            f"\n        path=\"{folder + '/' + str(step)}\"," + \
            f"\n        tokenizer_path=\"{tokenizer}\"," + \
            f"\n        module_path=\"{module_path}\"," + \
            f"\n        model_config=\"{model_config}\"," + \
            f"\n        model_type=\"{model_type}\"," + \
            f"\n        max_out_len={max_out_len}," + \
            f"\n        max_seq_len={max_seq_len}," + \
            f"\n        batch_size={batch_size}," + \
            f"\n        run_cfg=dict(num_gpus={num_gpus}, num_procs={num_procs})," + \
            f"\n        model_dtype=\"{dtype}\"," + \
            f"\n        submit_time=\"{submit_time}\"," + \
            f"\n        mode=\"{mode}\"," + \
            f"\n    ),"
    

class Request:
    def __init__(self, request: str, num_gpus, num_procs,
                 batch_size, max_out_len, max_seq_len, mode):
        self.num_gpus = num_gpus
        self.num_procs = num_procs
        self.max_out_len = max_out_len
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        assert mode in ["none", "mid"]
        self.mode = mode

        # parse request
        (path, info, datasets, submit_time, expected_time, res_doc, res_path, \
         status, model_type, dtype, cluster, abbr, tokenizer, module_path, \
         model_config, requester, evaluator) = request.split("\t")
        # remove prefix
        if path.startswith("local:"):
            path = path[len("local:"):]
        elif path.find("s3://") != -1:
            idx = path.find("s3://")
            path = path[idx:]

        # set model type to LLAMA if not set
        if model_type == "":
            model_type = "LLAMA"
        # Assume that path: /path/tofolder/step
        # remove the last /: /path/tofolder/step/ -> /path/tofolder/step
        if path.endswith("/"):
            path = path[:-1]
        folder, steps = os.path.split(path)
        # set abbr if abbr is not set
        if abbr == "":
            abbr = os.path.split(folder)[-1]

        self.abbr = abbr
        self.folder = folder
        self.steps = steps
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.module_path = module_path
        self.model_config = model_config
        self.cluster_name = cluster
        self.dtype = dtype
        self.submit_time = submit_time

    def cluster(self):
        return self.cluster_name

    @staticmethod
    def build(request: str, num_gpus, num_procs,
              batch_size, max_out_len, max_seq_len, mode):
        if "~" in os.path.split(request.split("\t")[0])[1]:
            return RangeStepRequest(request, num_gpus, num_procs,
                                    batch_size, max_out_len, max_seq_len,
                                    mode)
        else:
            return NormalRequest(request, num_gpus, num_procs,
                                 batch_size, max_out_len, max_seq_len,
                                 mode)


class NormalRequest(Request):

    def is_normal(self):
        return True

    def parse(self):
        models = []
        for step in self.steps.split(","):
            step = step.strip()
            models.append(get_normal_model(
                abbr=self.abbr, folder=self.folder, step=step,
                tokenizer=self.tokenizer, module_path=self.module_path,
                model_config=self.model_config, model_type=self.model_type,
                max_out_len=self.max_out_len, max_seq_len=self.max_seq_len,
                batch_size=self.batch_size, num_gpus=self.num_gpus,
                num_procs=self.num_procs, dtype=self.dtype,
                submit_time=self.submit_time, mode=self.mode
            ))
        return "\n".join(models)


class RangeStepRequest(Request):

    def is_normal(self):
        return False

    def parse(self):
        step_range, step_gap = self.steps.split(",")
        low, high = step_range.split("~")
        step_range = step_range.strip()
        low = int(low.strip())
        high = int(high.strip())
        step_gap = int(step_gap)
        n = (high - low) // step_gap + 1
        for_code = f"for i in range({n}):" + \
            f"\n    step = {low} + i * {step_gap}" + \
            f"\n    models.append(dict(" + \
            f"\n        abbr=f\"{self.abbr}_{{step}}\"," + \
            f"\n        type=\"opencompass.models.internal.InternLMwithModule\"," + \
            f"\n        path=os.path.join(\"{self.folder}\", str(step))," + \
            f"\n        tokenizer_path=\"{self.tokenizer}\"," + \
            f"\n        module_path=\"{self.module_path}\"," + \
            f"\n        model_config=\"{self.model_config}\"," + \
            f"\n        model_type=\"{self.model_type}\"," + \
            f"\n        max_out_len={self.max_out_len}," + \
            f"\n        max_seq_len={self.max_seq_len}," + \
            f"\n        batch_size={self.batch_size}," + \
            f"\n        model_dtype=\"{self.dtype}\"," + \
            f"\n        run_cfg=dict(num_gpus={self.num_gpus}, num_procs={self.num_procs})," + \
            f"\n        submit_time=\"{self.submit_time}\"," + \
            f"\n        mode=\"{self.mode}\"," + \
            f"\n))"

        # The last step
        if high != low + (n - 1) * step_gap:
            last = get_normal_model(
                abbr=self.abbr, folder=self.folder, step=high,
                tokenizer=self.tokenizer, module_path=self.module_path,
                model_config=self.model_config, model_type=self.model_type,
                max_out_len=self.max_out_len, max_seq_len=self.max_seq_len,
                batch_size=self.batch_size, num_gpus=self.num_gpus,
                num_procs=self.num_procs, dtype=self.dtype,
                submit_time=self.submit_time, mode=self.mode
            )
            for_code += "\nmodels += [\n" + last + "\n]"
        
        
        return for_code

class AutoGonfigGenerator:
    def __init__(self, output_dir, cluster, normal_requests, range_step_requests):
        self.normal = normal_requests
        self.range_step = range_step_requests
        self.output_dir = output_dir

        if cluster == "阿里云":
            self.filename = "models_aliyun.py"
        elif cluster == "火山云":
            self.filename = "models_volcano.py"
        elif cluster == "T":
            self.filename = "models_p.py"
        else:
            raise NotImplementedError

    def write_to_file(self):
        if len(self.normal) == 0 and len(self.range_step) == 0:
            return
        output = os.path.join(self.output_dir, self.filename)
        if os.path.exists(output):
            with open(output) as fp:
                content = fp.readlines()
                empty = len("\n".join(content).strip()) == 0
            has_os_import = False
            for line in content:
                if line == "import os\n":
                    has_os_import = True
        else:
            empty = True
            has_os_import = False

        # normal
        output_str = ""
        if len(self.normal) != 0:
            if empty:
                output_str += "models = " +"\n".join(["["] + self.normal + ["]"])
            else:
                output_str += "\n\nmodels += " +"\n".join(["["] + self.normal + ["]"])
            empty = False
                
        # range step
        if len(self.range_step) != 0:
            if empty:
                output_str += "import os\n\n" +"models = []\n\n" +"\n\n".join(self.range_step)
            else:
                if not has_os_import:
                    output_str += "\n\nimport os"
                output_str += "\n\n" + "\n\n".join(self.range_step)

        with open(output, "a") as fp:
            fp.write(output_str)


def check_param(requests, param):
    if isinstance(param, (int, str)):
        return [param for _ in range(len(requests))]
    elif isinstance(param, (list, tuple)):
        assert len(param) == len(requests)
        return param
    else:
        raise NotImplementedError
    
def parse_lark_requests(requests, output_dir, num_gpus = 1, num_procs = 1,
                       max_out_len = 100, max_seq_len = 2048, batch_size = 16,
                       mode="none"):
    
    requests = requests.split("\n")
    requests = [r for r in requests if len(r) != 0]

    num_gpus = check_param(requests, num_gpus)
    num_procs = check_param(requests, num_procs)
    max_out_len = check_param(requests, max_out_len)
    max_seq_len = check_param(requests, max_seq_len)
    batch_size = check_param(requests, batch_size)
    mode = check_param(requests, mode)
    normal_models = {"阿里云": [], "火山云": [], "T": []}
    range_step_models = {"阿里云": [], "火山云": [], "T": []}
    for i in range(len(requests)):
        request = Request.build(
            requests[i], num_gpus=num_gpus[i], num_procs=num_procs[i],
            max_out_len=max_out_len[i], max_seq_len=max_seq_len[i],
            batch_size=batch_size[i], mode=mode[i]
        )
        if request.is_normal():
            normal_models[request.cluster()].append(request.parse())
        else:
            range_step_models[request.cluster()].append(request.parse())

    for k in normal_models.keys():
        generator = AutoGonfigGenerator(output_dir, k, normal_models[k], range_step_models[k])
        generator.write_to_file()
        
def write(output_dir, filename, output_string):
    output = os.path.join(output_dir, filename)
    if os.path.exists(output):
        with open(output) as fp:
            content = fp.read()
            if content.strip() == "":
                prefix = "models = "
            else:
                prefix = "\n\nmodels += "
    else:
        folder, filename = os.path.split(output)
        os.makedirs(folder, exist_ok=True)
        prefix = "models = "
    with open(output, "a") as fp:
        fp.write(prefix + output_string)

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)
    parse_lark_requests(
        models, output_dir, num_gpus=num_gpus, num_procs=num_procs,
        batch_size=batch_size, max_out_len=max_out_len,
        max_seq_len=max_seq_len, mode=mode
    )
