#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import sys
import time
import json
import copy
from typing import Optional

import fire
import numpy as np

from datasets import Dataset
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

from tasks.boolq import EvalTaskBoolq
from tasks.logiqa import EvalTaskLogiqa
from tasks.commonsense_qa import EvalTaskCommonsenseqa
from tasks.social_iqa import EvalTaskSocialiqa
from tasks.sciq import EvalTaskSciq
from tasks.openbookqa import EvalTaskOpenbookqa
from tasks.ai2_arc import EvalTaskAi2Arc
from tasks.bbh import EvalTaskBbh
from tasks.mmlu import EvalTaskMmlu
from tasks.mmlu_pro import EvalTaskMmluPro

from utils.init_functions import logger_setup, cuda_setup, random_setup


class LMGenGPT:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            openai_model: str = "gpt-4o",
            openai_api_key: Optional[str] = None,
            api_call_sleep: float = 3.0,
            temperature: float = 1.0,
            debug: bool = False,
            output_dir: Optional[str] = None,
            num_few_shot: int = 0,
            seed_few_shot: int = 42,
            max_num_few_shot: int = 10,
            max_eval_num: int = -1,
            use_cot: bool = False,
            use_arr: bool = False,
            arr_ablation: str = "111",
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.debug = debug

        if isinstance(project_dir, str) and os.path.isdir(project_dir):
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd()
        assert os.path.isdir(project_dir)

        self.output_dir = output_dir
        self.num_few_shot = num_few_shot
        self.seed_few_shot = seed_few_shot
        self.max_num_few_shot = max_num_few_shot
        self.max_eval_num = max_eval_num
        self.use_cot = use_cot
        self.use_arr = use_arr
        self.arr_ablation = arr_ablation

        self.task_class_dict = {
            "boolq": EvalTaskBoolq,
            "logiqa": EvalTaskLogiqa,
            "openbookqa": EvalTaskOpenbookqa,
            "sciq": EvalTaskSciq,
            "social_iqa": EvalTaskSocialiqa,
            "ai2_arc": EvalTaskAi2Arc,
            "bbh": EvalTaskBbh,
            "commonsense_qa": EvalTaskCommonsenseqa,
            "mmlu": EvalTaskMmlu,
            "mmlu_pro": EvalTaskMmluPro,
        }

        # Cache directory
        self.home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
        else:
            self.cache_dir = os.path.join(self.home_dir, ".cache/huggingface")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            self.logger.info(f">>> cache_dir: {self.cache_dir}")

        # os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        # os.environ["HF_HOME"] = self.cache_dir

        # OpenAI settings
        # openai.organization = "YOUR_ORG_ID"
        # self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_key = openai_api_key
        assert isinstance(self.openai_api_key, str), f"Assertion Error: openai_api_key = {self.openai_api_key}"
        self.client = OpenAI(api_key=self.openai_api_key)
        self.openai_model = openai_model  # "gpt-4o"
        self.api_call_sleep = api_call_sleep
        self.temperature = temperature

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def gpt_generate(
            self,
            eval_task_name: str,
    ):
        assert isinstance(self.output_dir, str), "Please specify --output_dir"
        assert eval_task_name in self.task_class_dict, \
            f"AssertionError: task name {eval_task_name} not in task_class_dict"
        eval_task_class = self.task_class_dict[eval_task_name]

        eval_task_obj = eval_task_class(
            verbose=self.verbose,
            logger=self.logger,
            cache_dir=self.cache_dir,
            project_dir=self.project_dir,
            seed_few_shot=self.seed_few_shot,
            max_num_few_shot=self.max_num_few_shot,
        )

        self.logger.info(f">>> Evaluation Task: {eval_task_name}")
        task_info = eval_task_obj.load_task()
        dataset_list = task_info["data"]

        system_prompt = """
You are a helpful assistant. \
To answer the question, you need to select one from the given options. \
Your final answer must start with "Final Answer:"
        """.strip()

        # Deal with each task (and sub-tasks)
        all_results = {}
        show_cnt = 100
        for dataset_dict in dataset_list:
            cur_results = []
            ds_name, subset = dataset_dict["hf_dataset"], dataset_dict["hf_subset"]
            eval_split, eval_dataset = dataset_dict["eval_split"], dataset_dict["eval_dataset"]
            assert isinstance(eval_dataset, Dataset)
            len_dataset = len(eval_dataset)
            assert isinstance(ds_name, str) and len(ds_name) > 0
            if isinstance(subset, str) and len(subset) > 0:
                ds_id = f"{ds_name}---{subset}"
            else:
                ds_id = ds_name
            if self.verbose:
                self.logger.info(f">>> [Dataset: {ds_id}] [Eval: {eval_split}] # = {len_dataset}")

            # Run generation with batch_size = 1
            for idx, data_item in enumerate(eval_dataset):
                prompt_dict = eval_task_obj.set_prompt(
                    ds_name=ds_name,
                    subset=subset,
                    data_item=data_item,
                    num_few_shot=self.num_few_shot,
                    seed_few_shot=self.seed_few_shot,
                    use_cot=self.use_cot,
                    use_arr=self.use_arr,
                    arr_ablation=self.arr_ablation,
                )

                # Run generation - OpenAI request
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "developer", "content": system_prompt.strip()},
                        {"role": "user", "content": prompt_dict["prompt"]},
                    ],
                    temperature=self.temperature,
                )
                time.sleep(self.api_call_sleep)

                res_text = str(response.choices[0].message.content).strip()
                cur_gen_output = {
                    "index": idx,
                    "prompt": prompt_dict["prompt"],  # The input prompt
                    # "len_input": None,  # Number of tokens of the model input/prompt
                    "output_text": res_text,  # The LLM output (excluding the input)
                    "answer_str": prompt_dict["answer_str"],  # The golden answer (full string)
                    "answer_label": prompt_dict["answer_label"],  # The golden answer label (like 0 or "A")
                    "label_options": prompt_dict["label_options"],  # The options of all answer labels
                    "answer_options": prompt_dict["answer_options"],  # The options of all answer strings
                }

                cur_results.append(cur_gen_output)
                if self.verbose and len(cur_results) % show_cnt == 0:
                    self.logger.info(f">>> Progress: [{ds_id}] [{len(cur_results)} / {len_dataset}]")
                if len(cur_results) >= self.max_eval_num > 0:
                    break

            all_results[ds_id] = cur_results

        # Save the generation outputs and show logs
        output_dir = os.path.join(self.output_dir, eval_task_name, self.openai_model)
        os.makedirs(output_dir, exist_ok=True)
        output_fp = os.path.join(output_dir, "results_gen.json")
        if os.path.exists(output_fp):
            self.logger.info(f"Results will be overwritten: {output_fp}")
        else:
            self.logger.info(f"Results will be saved at: {output_fp}")
        dumped = json.dumps(
            all_results,
            indent=2,  # indent=None,
            default=self._handle_non_serializable,
            ensure_ascii=True,
        )
        with open(output_fp, "w", encoding="utf-8") as fp_out:
            fp_out.write(dumped)
        self.logger.info(
            f">>> DONE ALL. openai_model = {self.openai_model}; "
            f"num_few_shot: {self.num_few_shot}, use_cot: {self.use_cot}, use_arr: {self.use_arr}."
        )

    def gpt_generate_fewshot(
            self,
            eval_task_name: str,
    ):
        assert isinstance(self.output_dir, str), "Please specify --output_dir"
        assert eval_task_name in self.task_class_dict, \
            f"AssertionError: task name {eval_task_name} not in task_class_dict"
        eval_task_class = self.task_class_dict[eval_task_name]

        eval_task_obj = eval_task_class(
            verbose=self.verbose,
            logger=self.logger,
            cache_dir=self.cache_dir,
            project_dir=self.project_dir,
            seed_few_shot=self.seed_few_shot,
            max_num_few_shot=self.max_num_few_shot,
        )

        self.logger.info(f">>> Evaluation Task [few-shot examples only]: {eval_task_name}")

        backup_fp = eval_task_obj.few_shot_fp.replace(".json", "_backup.json")
        assert os.path.isfile(eval_task_obj.few_shot_fp), \
            f"AssertionError: file does not exist: {eval_task_obj.few_shot_fp}"
        os.system(f"cp {eval_task_obj.few_shot_fp} {backup_fp}")  # Make a copy for backup
        output_fp = eval_task_obj.few_shot_fp

        # Deal with each few-shot example
        few_shot_dict = eval_task_obj.few_shot
        all_results = copy.deepcopy(few_shot_dict)
        assert isinstance(few_shot_dict, dict)
        for ds_key, few_shot_list in few_shot_dict.items():
            cur_results = []
            assert isinstance(ds_key, str) and isinstance(few_shot_list, list)
            ds_name = eval_task_name
            if ds_name == ds_key:
                subset = None
                ds_id = ds_name
            else:
                subset = ds_key
                ds_id = f"{ds_name}---{subset}"

            len_fewshot = len(few_shot_list)
            if self.verbose:
                self.logger.info(f">>> [Dataset: {ds_id}] [Eval: few-shot] # = {len_fewshot}")

            # Run generation with batch_size = 1
            for idx, data_item in enumerate(few_shot_list):
                prompt_dict = eval_task_obj.set_prompt(
                    ds_name=ds_name,
                    subset=subset,
                    data_item=data_item,
                    num_few_shot=self.num_few_shot,
                    seed_few_shot=self.seed_few_shot,
                    use_cot=self.use_cot,
                    use_arr=self.use_arr,
                    arr_ablation=self.arr_ablation,
                )
                if prompt_dict is None:
                    continue

                try:
                    # Run generation - OpenAI request
                    response = self.client.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {"role": "developer", "content": "You are a helpful assistant."},  # system prompt
                            {"role": "user", "content": prompt_dict["prompt"]},
                        ],
                        temperature=self.temperature,
                    )
                except APIConnectionError as e:
                    self.logger.info(f">>> !!! >>> Failed to connect to OpenAI API: {e}")
                    sys.exit(1)
                except APIError as e:
                    self.logger.info(f">>> !!! >>> OpenAI API Error: {e}")
                    sys.exit(1)
                except RateLimitError as e:
                    self.logger.info(f">>> !!! >>> OpenAI API request exceeded rate limit: {e}")
                    sys.exit(1)
                except Exception as e:
                    self.logger.info(f">>> !!! >>> OpenAI Exception: {e}")
                    sys.exit(1)

                res_text = str(response.choices[0].message.content).strip()
                cur_gen_output = data_item
                if self.use_cot:
                    cur_gen_output["cot"] = res_text
                elif self.use_arr:
                    cur_gen_output["arr"] = res_text
                else:
                    cur_gen_output["gpt"] = res_text

                cur_results.append(cur_gen_output)

            all_results[ds_key] = cur_results

            if os.path.exists(output_fp):
                self.logger.info(f"Results will be overwritten: {output_fp}")
            else:
                self.logger.info(f"Results will be saved at: {output_fp}")
            dumped = json.dumps(
                all_results,
                indent=2,  # indent=None,
                default=self._handle_non_serializable,
                ensure_ascii=True,
            )
            with open(output_fp, "w", encoding="utf-8") as fp_out:
                fp_out.write(dumped)

        self.logger.info(
            f">>> DONE ALL. openai_model = {self.openai_model}; "
            f"num_few_shot: {self.num_few_shot}, use_cot: {self.use_cot}, use_arr: {self.use_arr}."
        )


def main(
    task: int = 0,
    eval_task_name: Optional[str] = None,
    openai_model: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    api_call_sleep: float = 3.0,
    temperature: float = 1.0,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    num_few_shot: int = 0,
    seed_few_shot: int = 42,
    max_num_few_shot: int = 10,
    max_eval_num: int = -1,
    use_cot: bool = False,
    use_arr: bool = False,
    arr_ablation: str = "111",
    **kwargs
) -> None:
    """
    - Stage 1: Reasoning Generation. Let GPT freely generate reasoning for later evaluation.
    - GPT API Generation for constructing the CoT/ARR reasoning for few-shot examples.

    :param task: 1. GPT API generation.
    :param eval_task_name: The name(s) of the evaluation task. (e.g., "boolq", "bbh", and "boolq,bbh")
    :param openai_model: e.g., "gpt-4o", "gpt-4o-2024-08-06"
    :param openai_api_key: your valid OpenAI API Key. https://platform.openai.com/
    :param api_call_sleep: The sleep time between API calls.
    :param temperature: The temperature used for generation. Default: 1.0
    :param cache_dir: The root directory of the cache.
    :param project_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param num_few_shot: The number of few-shot examples to provide. Default: 0
    :param seed_few_shot: Random seed for sampling few-shot examples.
    :param max_num_few_shot: The maximum number of few-shot examples.
    :param max_eval_num: The maximum number of evaluation instances (per subtask).
    :param use_cot: Use chain-of-thought prompting (providing CoT reasoning/rationale in the few-shot examples) or not.
    :param use_arr: Use our ARR method (providing ARR reasoning/rationale in the few-shot examples) or not.
        ARR: Analyzer, Retriever, and Reasoner
    :param arr_ablation: The ablation study of ARR prompting: 000 --> no A, no R, no R; 101 --> use A, no R, use R
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("GPT_Gen")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}\n")

    lm_gen = LMGenGPT(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_dir=project_dir,
        openai_model=openai_model,
        openai_api_key=openai_api_key,
        api_call_sleep=max(float(api_call_sleep), 0.1),
        temperature=max(float(temperature), 0.0),
        debug=debug,
        output_dir=output_dir,
        num_few_shot=int(num_few_shot),
        seed_few_shot=int(seed_few_shot),
        max_num_few_shot=int(max_num_few_shot),
        max_eval_num=int(max_eval_num),
        use_cot=use_cot,
        use_arr=use_arr,
        arr_ablation=str(arr_ablation).zfill(3),
    )

    task = int(task)
    match task:
        case 1:
            # After generation, manually Check the CoT/ARR reasoning/rationale generated by GPT
            if isinstance(eval_task_name, tuple) or isinstance(eval_task_name, list):
                for cur_task_name in eval_task_name:
                    cur_task_name = str(cur_task_name).strip()
                    lm_gen.gpt_generate_fewshot(eval_task_name=cur_task_name)
            elif isinstance(eval_task_name, str):
                eval_task_name = str(eval_task_name).strip()
                lm_gen.gpt_generate_fewshot(eval_task_name=eval_task_name)
            else:
                raise ValueError(f"--eval_task_name should be a tuple/list/str: {eval_task_name}")
        case 2:
            if isinstance(eval_task_name, tuple) or isinstance(eval_task_name, list):
                for cur_task_name in eval_task_name:
                    cur_task_name = str(cur_task_name).strip()
                    lm_gen.gpt_generate(eval_task_name=cur_task_name)
            elif isinstance(eval_task_name, str):
                eval_task_name = str(eval_task_name).strip()
                lm_gen.gpt_generate(eval_task_name=eval_task_name)
            else:
                raise ValueError(f"--eval_task_name should be a tuple/list/str: {eval_task_name}")
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
