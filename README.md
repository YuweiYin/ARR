<div align="center">

# ARR: Question Answering with LLMs via <br/> Analyzing, Retrieving, and Reasoning

<img src="https://yuweiyin.com/files/img/2025-02-15-ARR.jpg" alt="ARR" width="400" height="auto">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) &nbsp;
[![arXiv](https://img.shields.io/badge/arXiv-2502.04689-b31b1b.svg)](https://arxiv.org/abs/2502.04689)

</div>

<details open><summary>Paper Abstract</summary>

* **ARR**: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning
* **Authors**: [Yuwei Yin](https://www.yuweiyin.com/) and [Giuseppe Carenini](https://www.cs.ubc.ca/~carenini/)
* **Paper**: https://huggingface.co/papers/2502.04689

```text
Large language models (LLMs) achieve remarkable performance on challenging benchmarks 
that are often structured as multiple-choice question-answering (QA) tasks. Zero-shot 
Chain-of-Thought (CoT) prompting enhances reasoning in LLMs but provides only vague and 
generic guidance ("think step by step"). This paper introduces ARR, an intuitive and 
effective zero-shot prompting method that explicitly incorporates three key steps in QA 
solving: analyzing the intent of the question, retrieving relevant information, and 
reasoning step by step. Comprehensive experiments across diverse and challenging QA tasks 
demonstrate that ARR consistently improves the Baseline (without ARR prompting) and 
outperforms CoT. Ablation and case studies further validate the positive contributions of 
each component: analyzing, retrieving, and reasoning. Notably, intent analysis plays 
a vital role in ARR. Additionally, extensive evaluations across various model sizes, 
LLM series, and generation settings solidify the effectiveness, robustness, and 
generalizability of ARR.
```

</details>

## Development Environments

<details><summary>Environment Setup</summary>

- **Python**: Python 3.10
- **GPU**: A single NVIDIA V100-32GB or A100-40GB GPU
  - 1B/3B/7B/8B LLMs `float16` inference mode only

```bash
git clone https://github.com/YuweiYin/ARR
cd ARR/
# Now, "/path/to/ARR/" is the project root directory

# https://docs.conda.io/projects/miniconda/en/latest/
conda create -n arr python=3.10 -y
conda activate arr

pip install -r requirements.txt -i https://pypi.org/simple/
pip install -e . -i https://pypi.org/simple/

# We can set the Hugging Face cache directory. The following is for the dataset cache.
export HF_HOME="/path/to/your/.cache/huggingface/datasets"  # Default: "${HOME}/.cache/huggingface/datasets/"
```

</details>

## Datasets and Models (Paper Section 4)

- Download the datasets and models beforehand if the computing nodes have no Internet access or HOME storage is limited.
- Please ensure `CACHE_DIR` and `HF_TOKEN` in the script are correct directories.

### Datasets

```bash
# https://huggingface.co/datasets
HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
bash run_download_datasets.sh "${HF_TOKEN}" "${CACHE_DIR}"  # Download data to "${CACHE_DIR}/datasets/"
```

<details><summary>Multi Choice QA Datasets</summary>

- **Reading Comprehension**
  - [x] `boolq`: **BoolQ** - [Paper](https://aclanthology.org/N19-1300/); [Dataset](https://github.com/google-research-datasets/boolean-questions)
  - [x] `logiqa`: **LogiQA** - [Paper](https://arxiv.org/abs/2007.08124); [GitHub](https://github.com/lgw863/LogiQA-dataset)
- **Commonsense Reasoning**
  - [x] `commonsense_qa`: **CommonsenseQA** (CSQA) - [Paper](https://aclanthology.org/N19-1421/); [Dataset](https://huggingface.co/datasets/tau/commonsense_qa)
  - [x] `social_iqa`: **SocialIQA** (SIQA) - [Paper](https://arxiv.org/abs/1904.09728); [Dataset](https://huggingface.co/datasets/allenai/social_i_qa)
- **World Knowledge**
  - [x] `sciq`: **SciQ** - [Paper](https://aclanthology.org/W17-4413/); [Dataset](https://huggingface.co/datasets/allenai/sciq)
  - [x] `openbookqa`: **OpenBookQA** (OBQA) - [Paper](https://arxiv.org/abs/1809.02789); [Homepage](https://leaderboard.allenai.org/open_book_qa/submissions/get-started); [GitHub](https://github.com/allenai/OpenBookQA)
  - [x] `ai2_arc`: **ARC** - [Paper](https://arxiv.org/abs/1803.05457); [Homepage](https://leaderboard.allenai.org/arc/submissions/get-started); [GitHub](https://github.com/allenai/aristo-leaderboard)
- **Multitask Understanding**
  - [x] `bbh`: **BigBench Hard** (BBH) - [BigBench Paper](https://arxiv.org/abs/2206.04615); [BigBench GitHub](https://github.com/google/BIG-bench); [BBH Paper](https://arxiv.org/abs/2210.09261); [BBH Dataset](https://huggingface.co/datasets/lukaemon/bbh)
  - [x] `mmlu`: **MMLU** - [Paper](https://arxiv.org/abs/2009.03300); [Dataset](https://huggingface.co/datasets/cais/mmlu); [No-Train Data](https://huggingface.co/datasets/hails/mmlu_no_train)
  - [x] `mmlu_pro`: **MMLU-Pro** - [Paper](https://arxiv.org/abs/2406.01574); [Dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)

</details>

### Models

```bash
# https://huggingface.co/models
HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
bash run_download_models.sh "${HF_TOKEN}" "${CACHE_DIR}"  # Download models to "${CACHE_DIR}/"
```

## Main Experiments (Paper Section 5)

For each bash script, please ensure `CACHE_DIR` and `PROJECT_DIR` in the script are 
correct Hugging Face cache directory (default: `"~/.cache/huggingface/"`) and 
project root directory (`"/path/to/ARR/"`).

```bash
mkdir -p logs/  # where we save running logs
mkdir -p results/  # where we save experimental results
```

### QA Performance (Paper Table 2)

<details><summary>Experimental Settings</summary>

- **Comparison**: (Zero-shot Settings)
  - w/o Reason: directly selecting options without relying on rationales (skipping Reasoning Generation)
  - Baseline: `"Answer:"`
  - CoT: `"Answer: Let's think step by step."`
  - **ARR**: `"Answer: Let's analyze the intent of the question, find relevant information, and answer the question with step-by-step reasoning."`
- **Models**:
  - [x] `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/ARR/"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_FEWSHOT="0"

# [Reasoning Generation] **First**, freely generate answer with reasoning/rationale:
echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot"
bash run_gen_lm.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
echo -e "\n\n >>> bash run_gen_lm-cot.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot"
bash run_gen_lm-cot.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
echo -e "\n\n >>> bash run_gen_lm-arr.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot"
bash run_gen_lm-arr.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"

# [Option Selection] **Second**, answer the question (evaluate each option) (multi-choice QA -- Accuracy):
echo -e "\n\n >>> bash run_eval_lm-no_gen.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot"  # w/o Reason
bash run_eval_lm-no_gen.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot"  # Baseline
bash run_eval_lm.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
echo -e "\n\n >>> bash run_eval_lm-cot.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot"  # CoT
bash run_eval_lm-cot.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
echo -e "\n\n >>> bash run_eval_lm-arr.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot"  # ARR
bash run_eval_lm-arr.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
```

</details>

### Ablation Study (Paper Table 4)

<details><summary>Experimental Settings</summary>

- **Comparison**:
  - A / R / R: Analyzing / Retrieving / Reasoning
  - ARR = "000" "001" "010" "100" "111"
  - **ARR** 000 = Baseline: `"Answer:"`
  - **ARR** 001 = Reasoning-only: `"Answer: Let's answer the question with step-by-step reasoning."`
  - **ARR** 010 = Retrieving-only: `"Answer: Let's find relevant information, and answer the question."`
  - **ARR** 100 = Analyzing-only: `"Answer: Let's analyze the intent of the question, and answer the question."`
  - **ARR** 111 = ARR: `"Answer: Let's analyze the intent of the question, find relevant information, and answer the question with step-by-step reasoning."`
- **Models**:
  - [x] `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/ARR/"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
NUM_FEWSHOT="0"

# [Reasoning Generation] **First**, freely generate answer with reasoning/rationale:
for ABLATION in "001" "010" "100"  # "000" = baseline; "111" = ARR
do
  echo -e "\n\n >>> bash run_gen_lm-arr.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot (ABLATION: ARR = ${ABLATION})"
  bash run_gen_lm-arr.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT};${ABLATION}" "${CACHE_DIR}" "${PROJECT_DIR}"
done

# [Option Selection] **Second**, answer the question (evaluate each option) (multi-choice QA -- Accuracy):
for ABLATION in "001" "010" "100"  # "000" = baseline; "111" = ARR
do
  echo -e "\n\n >>> bash run_eval_lm-arr.sh --hf_id ${MODEL} QA_ALL ${NUM_FEWSHOT}-shot (ABLATION: ARR = ${ABLATION})"
  bash run_eval_lm-arr.sh "1;${MODEL};1;QA_ALL;0.0;${NUM_FEWSHOT};${ABLATION}" "${CACHE_DIR}" "${PROJECT_DIR}"
done
```

</details>

## Generalizability (Paper Section 6)

### Model Sizes (Paper Table 6)

<details><summary>Experimental Settings</summary>

- **Comparison**:
  - The effect of the proposed **ARR** method using LLMs of different sizes
- **Models**:
  - [x] Llama-3
    - `meta-llama/Llama-3.2-1B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct))
    - `meta-llama/Llama-3.2-3B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct))
    - `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/ARR/"

# [Reasoning Generation] **First**, freely generate answer with reasoning/rationale:
for MODEL in "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
do
  NUM_FEWSHOT="0"
  echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-cot.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-arr.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done

# [Option Selection] **Second**, answer the question (evaluate each option) (multi-choice QA -- Accuracy):
for MODEL in "meta-llama/Llama-3.2-1B-Instruct" "meta-llama/Llama-3.2-3B-Instruct"
do
  NUM_FEWSHOT="0"
  echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-cot.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-arr.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done
```

</details>

### LLM Series (Paper Table 7)

<details><summary>Experimental Settings</summary>

- **Comparison**:
  - The effect of the proposed **ARR** method using other LLM series
- **Models**:
  - [x] `Qwen/Qwen2.5-7B-Instruct` ([Link](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct))
  - [x] `google/gemma-7b-it` ([Link](https://huggingface.co/google/gemma-7b-it))
  - [x] `mistralai/Mistral-7B-Instruct-v0.3` ([Link](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/ARR/"

# [Reasoning Generation] **First**, freely generate answer with reasoning/rationale:
for MODEL in "Qwen/Qwen2.5-7B-Instruct" "google/gemma-7b-it" "mistralai/Mistral-7B-Instruct-v0.3"
do
  NUM_FEWSHOT="0"
  echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-cot.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-arr.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done

# [Option Selection] **Second**, answer the question (evaluate each option) (multi-choice QA -- Accuracy):
for MODEL in "Qwen/Qwen2.5-7B-Instruct" "google/gemma-7b-it" "mistralai/Mistral-7B-Instruct-v0.3"
do
  NUM_FEWSHOT="0"
  echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-cot.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-arr.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done
```

</details>

### Generation Temperatures (Paper Table 8)

<details><summary>Experimental Settings</summary>

- **Comparison**:
  - Observe the effect of the proposed **ARR** method using different generation temperatures
  - Temperature: 0 (default), 0.5, 1.0, 1.5
- **Models**:
  - [x] `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/ARR/"

MODEL="meta-llama/Llama-3.1-8B-Instruct"

# [Reasoning Generation] **First**, freely generate answer with reasoning/rationale:
NUM_FEWSHOT="0"
for TEMPERATURE in "0.5" "1.0" "1.5"  # default: "0.0"
do
  echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm.sh "1;${MODEL};1;QA_GEN;${TEMPERATURE};${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-cot.sh "1;${MODEL};1;QA_GEN;${TEMPERATURE};${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-arr.sh "1;${MODEL};1;QA_GEN;${TEMPERATURE};${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done

# [Option Selection] **Second**, answer the question (evaluate each option) (multi-choice QA -- Accuracy):
NUM_FEWSHOT="0"
for TEMPERATURE in "0.5" "1.0" "1.5"  # default: "0.0"
do
  echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm.sh "1;${MODEL};1;QA_GEN;${TEMPERATURE};${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-cot.sh "1;${MODEL};1;QA_GEN;${TEMPERATURE};${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-arr.sh "1;${MODEL};1;QA_GEN;${TEMPERATURE};${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done
```

</details>

### Few-shot Generation (Paper Table 9)

<details><summary>Experimental Settings</summary>

- **Comparison**:
  - Observe the effect of the proposed **ARR** method in few-shot settings (using different shots)
  - QA Performance comparison among baseline (in-context learning), few-shot CoT, and few-shot **ARR** methods
  - N shots: 0 (default), 1, 3, and 5
- **Models**:
  - [x] `meta-llama/Llama-3.1-8B-Instruct` ([Link](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))

</details>

<details><summary>Obtain CoT/ARR Rationales for Few-shot Examples</summary>

```bash
# Before LLM generation and evaluation, we first obtain 1/3/5 few-shot examples from the dev/train set,
#   where the CoT and ARR reasoning/rationale for each few-shot example is constructed by GPT-4o.
# Now, the `few_shot.json` file under each evaluation task already has CoT and ARR reasoning/rationale.

#CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
#PROJECT_DIR="/path/to/ARR/"
#bash run_gen_gpt.sh "1;QA_GEN;0;111;1.0" "${CACHE_DIR}" "${PROJECT_DIR
```

</details>

<details><summary>Experiment Script</summary>

```bash
CACHE_DIR="YOUR_HF_CACHE_DIR"  # E.g., "${HOME}/.cache/huggingface/"
PROJECT_DIR="/path/to/ARR/"

MODEL="meta-llama/Llama-3.1-8B-Instruct"

# [Reasoning Generation] **First**, freely generate answer with reasoning/rationale:
for NUM_FEWSHOT in "1" "3" "5"  # default: "0"
do
  echo -e "\n\n >>> bash run_gen_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-cot.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_gen_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_gen_lm-arr.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done

# [Option Selection] **Second**, answer the question (evaluate each option) (multi-choice QA -- Accuracy):
for NUM_FEWSHOT in "1" "3" "5"  # default: "0"
do
  echo -e "\n\n >>> bash run_eval_lm.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-cot.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-cot.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
  echo -e "\n\n >>> bash run_eval_lm-arr.sh --hf_id ${MODEL} QA_GEN ${NUM_FEWSHOT}-shot"
  bash run_eval_lm-arr.sh "1;${MODEL};1;QA_GEN;0.0;${NUM_FEWSHOT}" "${CACHE_DIR}" "${PROJECT_DIR}"
done
```

</details>

## License

Please refer to the [LICENSE](./LICENSE) file for more details.

## Citation

* **Paper** (arXiv): https://arxiv.org/abs/2502.04689
* If you find our work helpful, please kindly star this GitHub repo and cite our paper. 🤗

```bibtex
@article{yin2025arr,
  title   = {ARR: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning},
  author  = {Yin, Yuwei and Carenini, Giuseppe},
  journal = {arXiv preprint arXiv:2502.04689},
  year    = {2025},
  url     = {https://arxiv.org/abs/2502.04689},
}
```

---
