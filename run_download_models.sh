#!/bin/bash

HF_TOKEN=$1
if [[ -z ${HF_TOKEN} ]]; then
  HF_TOKEN="YOUR_HF_TOKEN"  # https://huggingface.co/settings/tokens
fi

CACHE_DIR=$2
if [[ -z ${CACHE_DIR} ]]; then
  CACHE_DIR="${HOME}/.cache/huggingface/"
fi

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "meta-llama/Llama-3.2-1B-Instruct"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "meta-llama/Llama-3.2-3B-Instruct"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "meta-llama/Llama-3.1-8B-Instruct"

python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "Qwen/Qwen2.5-7B-Instruct"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "google/gemma-7b-it"
python3 utils/download_hf_model.py --hf_token "${HF_TOKEN}" --cache_dir "${CACHE_DIR}" --trust_remote_code --verbose \
  --hf_id "mistralai/Mistral-7B-Instruct-v0.3"
