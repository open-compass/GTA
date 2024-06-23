# GTA: A Benchmark for General Tool Agents

<!-- <div align="center">

<img src="https://github.com/open-compass/opencompass/assets/28834990/c285f051-f6cb-4425-8045-863bb94095ed" width="400">
  <div> </div>
    <b><font size="3">MathBench</font></b>
    <div> 
  </div>
</div> -->


<div align="center">

[📃 [Paper](https://xxx)]
[🌐 [Project Page](https://xxx)]
[🤗 [Hugging Face](https://xxx)]
[📌 [License](https://xxx)]
</div>

## 🌟 Introduction

GTA is a benchmark to evaluate the tool-use capability of LLM-based agents in real-world scenarios. It features three main aspects:
- **Real user queries.** The benchmark contains 229 human-written queries with simple real-world objectives but implicit tool-use, requiring the LLM to reason the suitable tools and plan the solution steps. 
- **Real deployed tools.** an evaluation platform equipped with tools across perception, operation, logic, and creativity categories to evaluate the agents' actual task execution performance.
- **Real multimodal inputs.** authentic image files, such as spatial scenes, web page screenshots, tables, code snippets, and printed/handwritten materials, used as the query contexts to align with real-world scenarios closely.
<div align="center">
 <img src="figs/dataset.jpg" width="800"/>
</div>

## 📣 What's New

- **[2024.x.xx]** Paper available on Arxiv. ✨✨✨
- **[2024.x.xx]** Release the evaluation and tool deployment code of GTA. 🔥🔥🔥
- **[2024.x.xx]** Release the GTA dataset on Hugging Face and OpenDataLab. 🎉🎉🎉

## 📚 Dataset Statistics
GTA comprises a total of 229 questions. The basic dataset statistics is presented below. 

<div align="center">
 <img src="figs/statistics.jpg" width="800"/>
</div>

## 🏆 Leader Board

We evaluate the language models in two modes:
- **Step-by-step mode.** It is designed to evaluate the model's fine-grained tool-use capabilities. In this mode, the model is provided with the initial $n$ steps of the reference tool chain as prompts, with the expectation to predict the action in step $n+1$. Four metrics are devised under step-by-step mode: ***InstAcc*** (instruction following accuracy), ***ToolAcc*** (tool selection accuracy), ***ArgAcc*** (argument prediction accuracy), and ***SummAcc*** (answer summarizing accuracy).

- **End-to-end mode.** It is designed to reflect the tool agent's actual task executing performance. In this mode, the model actually calls the tools and solves the problem by itself. We use ***AnsAcc*** (final answer accuracy) to measure the accuracy of the execution result. Besides, we calculate four ***F1 scores of tool selection: P, L, O, C*** in perception, operation, logic, and creativity categories, to measure the tool selection capability. 

Here is the performance of various LLMs on GTA. Inst, Tool, Arg, Summ, and Ans denote InstAcc, ToolAcc, ArgAcc SummAcc, and AnsAcc, respectively. P, O, L, C denote the F1 score of tool selection in Perception, Operation, Logic, and Creativity categories. ***Bold*** denotes the best score among all models. <u>*Underline*</u> denotes the best score under the same model scale. ***AnsAcc*** reflects the overall performance.

**Models** | **Inst** | **Tool** | **Arg** | **Summ** | **P** | **O** | **L** | **C** | **Ans**
---|---|---|---|---|---|---|---|---|---
***API-based*** | | | | | | | | |
gpt-4-1106-preview | 85.19 | 61.4 | <u>**37.88**</u> | 75 | 67.61 | 64.61 | 74.73 |89.55 |  <u>**46.59**</u>
gpt-4o | 61.7 | 80.0 | 55.7 | 38.7 | 20.7 | 51.3 
gpt-3.5-turbo | 76.0 | 82.3 | 59.0 | 41.3 | 35.3 | 58.8 
mistral-large | 82.7 | **89.3** | 59.0 | 39.3 | 29.3 | 59.9 
***Open-source*** | | | | | | | | |
qwen1.5-72b-chat | 35.3 | 36.3 | 7.0 | 3.0 | 4.3 | 17.2 
mixtral-8x7b-instruct | 38.0 | 41.0 | 13.7 | 5.3 | 1.7 | 19.9 
deepseek-llm-67b-chat | 48.3 | 47.7 | 8.7 | 4.3 | 2.7 | 22.3 
llama3-70b-instruct | 50.7 | 50.7 | 22.0 | 9.3 | 6.0 | 27.7 
yi-34b-chat | 52.0 | 66.3 | <u>30.0</u> | <u>13.7</u> | <u>8.7</u> | <u>34.1</u> 
---|---|---|---|---|---|---|---|---|---
qwen1.5-14b-chat | <u>54.7</u> | <u>71.0</u> | 25.0 | <u>19.0</u> | <u>14.0</u> | <u>36.7</u> 
qwen1.5-7b-chat | 40.0 | 44.7 | 13.7 | 4.7 | 1.7 | 20.9 
mistral-7b-instruct | 50.7 | 62.0 | 23.0 | 14.7 | 7.7 | 31.6 
deepseek-llm-7b-chat | <u>63.7</u> | 61.7 | <u>39.0</u> | 21.0 | 12.0 | 39.5 
llama3-8b-instruct | 62.3 | <u>72.7</u> | 37.7 | <u>24.7</u> | <u>13.0</u> | <u>42.1</u> 
yi-6b-chat | 62.0 | 72.7 | 33.3 | 21.3 | 12.0 | 40.3 



## 🚀 Get Started
[OpenCompass](https://github.com/open-compass/opencompass) is a toolkit for evaluating the performance of large language models (LLMs). There are steps for inference MathBench with OpenCompass:
1. Install OpenCompass
```
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
```
2. Prepare the dataset, you can download the data from [release file](https://github.com/open-compass/MathBench/releases/tag/v0.1.0)
```
# Download dataset from release file and copy to data/ folder
mkdir data
cp -rf mathbench_v1 ./data/ 
```
3. Inference MathBench
```
# Inference MathBench with hf_llama2_7b_chat model
python run.py --models hf_llama2_7b_chat --datasets mathbench_gen
```
You can also evaluate HuggingFace models via command line. 
```
python run.py --datasets mathbench_gen \
--hf-path meta-llama/Llama-2-7b-chat-hf \  # HuggingFace model path
--model-kwargs device_map='auto' \  # Arguments for model construction
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # Arguments for tokenizer construction
--max-seq-len 2048 \  # Maximum sequence length the model can accept
--batch-size 8 \  # Batch size
--no-batch-padding \  # Don't enable batch padding, infer through for loop to avoid performance loss
--num-gpus 1  # Number of minimum required GPUs
--summarizer summarizers.mathbench_v1 # Summarizer for MathBench
```




# 📝 Citation
If you use GTA in your research, please cite the following paper:
```
@misc{xxx,
      title={GTA: A Benchmark for General Tool Agents}, 
      author={xxx},
      year={2024},
      eprint={xxx},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
