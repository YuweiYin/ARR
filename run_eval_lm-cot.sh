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
    MODEL=${val}
  elif [[ "${idx}" == "3" ]]; then
    BSZ=${val}
  elif [[ "${idx}" == "4" ]]; then
    EVAL_TASKS=${val}
  elif [[ "${idx}" == "5" ]]; then
    GEN_TEMP=${val}
  elif [[ "${idx}" == "6" ]]; then
    NUM_FEW_SHOT=${val}
  elif [[ "${idx}" == "7" ]]; then
    ARR_ABLATION=${val}
  elif [[ "${idx}" == "8" ]]; then
    CKPT_PATH=${val}
  fi
done

if [[ -z ${TASK} ]]; then
  TASK="1"
fi

if [[ -z ${BSZ} ]]; then
  BSZ="1"
fi

if [[ -z ${EVAL_TASKS} ]]; then
  echo -e "!!! Error EVAL_TASKS input: \"${EVAL_TASKS}\"\n"
  exit 1
fi

if [[ -z ${GEN_TEMP} ]]; then
  GEN_TEMP="0.0"
fi

if [[ -z ${NUM_FEW_SHOT} ]]; then
  NUM_FEW_SHOT="0"
fi

if [[ -z ${ARR_ABLATION} ]]; then
  ARR_ABLATION="111"
fi

if [[ -z ${CKPT_PATH} ]]; then
  CKPT_PATH="___NONE___"
fi

MODEL_NAME="${MODEL//[\/]/_}"

echo -e "TASK: ${TASK}"
echo -e "MODEL: ${MODEL}"
echo -e "MODEL_NAME: ${MODEL_NAME}"
echo -e "BSZ: ${BSZ}"
echo -e "EVAL_TASKS: ${EVAL_TASKS}"
echo -e "GEN_TEMP: ${GEN_TEMP}"
echo -e "NUM_FEW_SHOT: ${NUM_FEW_SHOT}"
echo -e "ARR_ABLATION: ${ARR_ABLATION}"
echo -e "CKPT_PATH: ${CKPT_PATH}"

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

SEED=42

if [[ ${EVAL_TASKS} == "QA_ALL" ]]; then
  EVAL_TASK_NAME="boolq,logiqa,commonsense_qa,social_iqa,sciq,openbookqa,ai2_arc,bbh,mmlu,mmlu_pro"
elif [[ ${EVAL_TASKS} == "QA_GEN" ]]; then
  EVAL_TASK_NAME="bbh,mmlu,mmlu_pro"
else
  EVAL_TASK_NAME="${EVAL_TASKS}"
fi

echo -e "\n\n >>> python3 run_eval_lm.py --use_cot --eval_task_name ${EVAL_TASK_NAME} --hf_id ${MODEL}"
OUTPUT_DIR="results/eval_results-temp_${GEN_TEMP}-${NUM_FEW_SHOT}-shot--cot/"  # ${MODEL_NAME} is subdir
mkdir -p "${OUTPUT_DIR}"
python3 run_eval_lm.py \
  --task "${TASK}" \
  --eval_task_name "${EVAL_TASK_NAME}" \
  --hf_id "${MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --project_dir "${PROJECT_DIR}" \
  --ckpt_path "${CKPT_PATH}" \
  --seed "${SEED}" \
  --ckpt_dir "${HOME}/ckpt/intent_analysis/" \
  --bsz "${BSZ}" \
  --use_cot \
  --use_gen_output \
  --verbose

# --overwrite
