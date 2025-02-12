#!/bin/bash

PARAM=$1

IFS=";"  # Set ";" as the delimiter
read -ra PARAM_ARRAY <<< "${PARAM}"

#echo ${#PARAM_ARRAY[@]}
idx=0
for val in "${PARAM_ARRAY[@]}";
do
  idx=$(( $((idx)) + 1 ))
  # echo -e ">>> idx = ${idx}; val = ${val}"
  if [[ "${idx}" == "1" ]]; then
    TASK=${val}
  elif [[ "${idx}" == "2" ]]; then
    EVAL_TASKS=${val}
  elif [[ "${idx}" == "3" ]]; then
    NUM_FEW_SHOT=${val}
  elif [[ "${idx}" == "4" ]]; then
    ARR_ABLATION=${val}
  elif [[ "${idx}" == "5" ]]; then
    GPT_TEMP=${val}
  fi
done

if [[ -z ${TASK} ]]; then
  TASK="1"
fi

if [[ -z ${EVAL_TASKS} ]]; then
  echo -e "!!! Error EVAL_TASKS input: \"${EVAL_TASKS}\"\n"
  exit 1
fi

if [[ -z ${NUM_FEW_SHOT} ]]; then
  NUM_FEW_SHOT="0"
fi

if [[ -z ${ARR_ABLATION} ]]; then
  ARR_ABLATION="111"
fi

if [[ -z ${GPT_TEMP} ]]; then
  GPT_TEMP="1.0"
fi

echo -e "TASK: ${TASK}"
echo -e "EVAL_TASKS: ${EVAL_TASKS}"
echo -e "NUM_FEW_SHOT: ${NUM_FEW_SHOT}"
echo -e "ARR_ABLATION: ${ARR_ABLATION}"
echo -e "GPT_TEMP: ${GPT_TEMP}"

CACHE_DIR=$2
PROJECT_DIR=$3
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi
if [[ -z ${PROJECT_DIR} ]]; then
  PROJECT_DIR="/path/to/ARR/"
fi
echo -e "CACHE_DIR: ${CACHE_DIR}"
echo -e "PROJECT_DIR: ${PROJECT_DIR}"

SEED_FEW_SHOT=42
MAX_NUM_FEW_SHOT=10

OPENAI_MODEL=$4
OPENAI_API_KEY=$5
if [[ -z ${OPENAI_MODEL} ]]; then
  OPENAI_MODEL="gpt-4o"
fi
if [[ -z ${OPENAI_API_KEY} ]]; then
  OPENAI_API_KEY="YOUR_API_KEY"  # https://platform.openai.com/
fi
echo -e "OPENAI_MODEL: ${OPENAI_MODEL}"

if [[ ${EVAL_TASKS} == "QA_ALL" ]]; then
  EVAL_TASK_NAME="boolq,logiqa,commonsense_qa,social_iqa,sciq,openbookqa,ai2_arc,bbh,mmlu,mmlu_pro"
elif [[ ${EVAL_TASKS} == "QA_GEN" ]]; then
  EVAL_TASK_NAME="bbh,mmlu,mmlu_pro"
else
  EVAL_TASK_NAME="${EVAL_TASKS}"
fi

echo -e "\n\n >>> python3 run_gen_gpt.py --use_cot --eval_task_name ${EVAL_TASK_NAME} --openai_model ${OPENAI_MODEL}"
OUTPUT_DIR="results/gen_lm-gpt4o_fewshot_cot/"
mkdir -p "${OUTPUT_DIR}"
python3 run_gen_gpt.py \
  --task "${TASK}" \
  --eval_task_name "${EVAL_TASK_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --openai_model "${OPENAI_MODEL}" \
  --openai_api_key "${OPENAI_API_KEY}" \
  --temperature "${GPT_TEMP}" \
  --num_few_shot "${NUM_FEW_SHOT}" \
  --seed_few_shot "${SEED_FEW_SHOT}" \
  --max_num_few_shot "${MAX_NUM_FEW_SHOT}" \
  --use_cot \
  --arr_ablation "${ARR_ABLATION}" \
  --run_few_shot \
  --verbose

echo -e "\n\n >>> python3 run_gen_gpt.py --use_arr --eval_task_name ${EVAL_TASK_NAME} --openai_model ${OPENAI_MODEL}"
OUTPUT_DIR="results/gen_lm-gpt4o_fewshot_arr/"
mkdir -p "${OUTPUT_DIR}"
python3 run_gen_gpt.py \
  --task "${TASK}" \
  --eval_task_name "${EVAL_TASK_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --openai_model "${OPENAI_MODEL}" \
  --openai_api_key "${OPENAI_API_KEY}" \
  --temperature "${GPT_TEMP}" \
  --num_few_shot "${NUM_FEW_SHOT}" \
  --seed_few_shot "${SEED_FEW_SHOT}" \
  --max_num_few_shot "${MAX_NUM_FEW_SHOT}" \
  --use_arr \
  --arr_ablation "${ARR_ABLATION}" \
  --run_few_shot \
  --verbose
