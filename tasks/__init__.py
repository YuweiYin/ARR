#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import time
# import random
from typing import Optional, Dict, Any

import fire
from datasets import load_dataset
from datasets import Dataset
from transformers import AutoTokenizer

from utils.init_functions import logger_setup, random_setup


class EvalTaskManager:

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            seed_few_shot: int = 42,
            max_num_few_shot: int = 10,
    ):
        self.verbose = verbose
        self.logger = logger
        self.seed_few_shot = seed_few_shot
        self.max_num_few_shot = max_num_few_shot

        if isinstance(project_dir, str) and os.path.isdir(project_dir):
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd()
        assert os.path.isdir(project_dir)

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

        self.task_name = None
        self.task_info = None
        self.few_shot = None

        self.all_tasks = {
            "boolq", "logiqa", "commonsense_qa", "social_iqa", "sciq",
            "openbookqa", "ai2_arc", "bbh",  "mmlu", "mmlu_pro",
        }

    def load_task(
            self,
    ) -> Dict[str, Any]:
        assert isinstance(self.task_name, str) and self.task_name in self.all_tasks
        assert isinstance(self.task_info, dict) and isinstance(self.few_shot, dict)
        hf_ds_list = self.task_info["hf_dataset"]
        assert isinstance(hf_ds_list, list) and len(hf_ds_list) > 0

        self.logger.info(f">>> [task_name: {self.task_name}]")
        dataset = {
            "task_name": self.task_name,
            "data": [],
        }
        for hf_ds in hf_ds_list:
            assert isinstance(hf_ds, list) and len(hf_ds) == 3
            # self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}]")
            try:  # Load the subset
                cur_ds = load_dataset(
                    hf_ds[0],
                    hf_ds[1],
                    cache_dir=os.path.join(self.cache_dir, "datasets"),
                    trust_remote_code=True,
                )
                eval_split = hf_ds[2]
                assert eval_split in cur_ds
                ds_dict = {
                    "hf_dataset": hf_ds[0],
                    "hf_subset": hf_ds[1],
                    "eval_split": eval_split,
                    "eval_dataset": cur_ds[eval_split],
                }

                if "train" in cur_ds:
                    len_train = len(cur_ds["train"])
                    # random.seed(self.seed_few_shot)  # Pick few-shot examples from the training set
                    # few_shot_idx = random.sample(range(len_train), self.max_num_few_shot)
                    # few_shot_train = cur_ds["train"][few_shot_idx]
                else:
                    len_train = 0

                if "validation" in cur_ds:
                    len_valid = len(cur_ds["validation"])
                    # random.seed(self.seed_few_shot)  # Pick few-shot examples from the validation set
                    # few_shot_idx = random.sample(range(len_valid), self.max_num_few_shot)
                    # few_shot_valid = cur_ds["validation"][few_shot_idx]
                else:
                    len_valid = 0

                if "test" in cur_ds:
                    len_test = len(cur_ds["test"])
                    # random.seed(self.seed_few_shot)  # Pick few-shot examples from the test set
                    # few_shot_idx = random.sample(range(len_test), self.max_num_few_shot)
                    # few_shot_test = cur_ds["test"][few_shot_idx]
                else:
                    len_test = 0

                self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}] [eval_split = {eval_split}] "
                                 f"Train = {len_train}, Validation = {len_valid}, Test = {len_test}")

                dataset["data"].append(ds_dict)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f">>> Exception: {e}")

        self.logger.info(f">>> [task_name: {self.task_name}] len(dataset) = {len(dataset['data'])}\n\n")
        return dataset

    def set_prompt(
            self,
            ds_name: str,
            subset: str,
            data_item,
            num_few_shot: int = 0,
            seed_few_shot: int = 42,
            use_cot: bool = False,
            use_arr: bool = False,
            arr_ablation: str = "111",
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def text_cleaning(
            text_input: str,
    ) -> str:
        text_output = text_input.strip()
        if len(text_output) == 0:
            return ""

        # text_output = text_output.replace("**", "")
        text_output = text_output.replace("\n", " ")
        text_output = " ".join(text_output.split())  # replace multiple whitespaces
        text_output = text_output.strip()
        if len(text_output) == 0:
            return ""

        if not text_output.endswith("."):
            text_output += "."

        return text_output

    def token_stat(
            self,
            eval_task_obj,
            cache_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
    ) -> None:
        # Cache directory
        home_dir = os.path.expanduser("~")
        if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
            cache_dir = cache_dir
        else:
            cache_dir = os.path.join(home_dir, ".cache/huggingface")
            if not os.path.isdir(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        os.environ["HF_HOME"] = cache_dir
        hf_name = "--".join(hf_id.split("/"))
        model_path = os.path.join(
            cache_dir, "models--" + hf_name, "snapshots/model")
        assert os.path.isdir(model_path), f"AssertionError: assert os.path.isdir({model_path})"

        # Tokenizer and LLM model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            truncation_side="left",  # "right" for training, "left" for generating
            cache_dir=cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        max_len = tokenizer.max_len_single_sentence
        if self.verbose:
            self.logger.info(
                f">>> len(tokenizer.vocab) = {len(tokenizer.vocab)}; "
                f"tokenizer.max_len_single_sentence = {max_len}")  # LLaMA-3: 131071

        # Load the dataset and do statistics
        task_info = eval_task_obj.load_task()
        dataset_list = task_info["data"]

        # Deal with each task (and sub-tasks)
        all_stat = {}
        all_len_token = []
        for dataset_dict in dataset_list:
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

            if "options" in dataset_dict:
                ds_options = list(dataset_dict["options"])
            else:
                ds_options = []

            # Do statistics for each data item (batch_size = 1)
            cur_stat = []
            for idx, data_item in enumerate(eval_dataset):
                assert isinstance(data_item, dict)
                data_item["__ds_options"] = ds_options
                prompt_dict = eval_task_obj.set_prompt(
                    ds_name=ds_name,
                    subset=subset,
                    data_item=data_item,
                    num_few_shot=0,
                    seed_few_shot=self.seed_few_shot,
                    use_cot=False,
                    use_arr=False,
                )

                cur_prompt = prompt_dict["prompt"]
                input_ids = tokenizer(
                    cur_prompt,
                    padding=True,  # truncation=True, max_length=1024
                    return_tensors="pt",
                )
                len_input = input_ids.data["input_ids"].size(-1)

                cur_stat.append({
                    "prompt": cur_prompt,
                    "len_char": len(cur_prompt),
                    "len_token": len_input,
                })
                all_len_token.append(len_input)

            all_stat[ds_id] = cur_stat

        # Show logs
        assert len(all_len_token) > 0
        avg_len_token = sum(all_len_token) / len(all_len_token)
        self.logger.info(
            f">>> DONE ALL. hf_id = {hf_id}; model_path = {model_path}\n"
            f">>> !!! >>> #Sub-Tasks = {len(all_stat)}; #Total Ins. = {len(all_len_token)}; "
            f"avg_len_token: {avg_len_token}\n\n"
        )


def main(
        cache_dir: Optional[str] = None,
        project_dir: Optional[str] = None,
        seed: int = 42,
        seed_few_shot: int = 42,
        max_num_few_shot: int = 10,
        verbose: bool = False,
        hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        **kwargs
) -> None:
    """
    Evaluation Tasks and Datasets.

    :param cache_dir: The root directory of the cache.
    :param project_dir: The directory of the project root.
    :param seed: Random seed of all modules.
    :param seed_few_shot: Random seed for sampling few-shot examples.
    :param max_num_few_shot: The maximum number of few-shot examples.
    :param verbose: Verbose mode: show logs.
    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setups
    logger = logger_setup("Eval_Tasks")
    random_setup(seed=seed, has_cuda=False)

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}\n")

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

    # Dataset loading
    # eval_tasks = EvalTaskCommonsenseqa(
    #     verbose=verbose,
    #     logger=logger,
    #     cache_dir=cache_dir,
    #     project_dir=project_dir,
    #     seed_few_shot=seed_few_shot,
    #     max_num_few_shot=max_num_few_shot,
    # )
    # cur_dataset = eval_tasks.load_task()
    # eval_tasks.token_stat(eval_task_obj=eval_tasks, cache_dir=cache_dir, hf_id=hf_id)

    # Token length statistics
    all_qa = [EvalTaskBoolq, EvalTaskLogiqa, EvalTaskCommonsenseqa, EvalTaskSocialiqa, EvalTaskSciq,
              EvalTaskOpenbookqa, EvalTaskAi2Arc, EvalTaskBbh, EvalTaskMmlu, EvalTaskMmluPro]
    for eval_class in all_qa:
        eval_tasks = eval_class(
            verbose=verbose,
            logger=logger,
            cache_dir=cache_dir,
            project_dir=project_dir,
            seed_few_shot=seed_few_shot,
            max_num_few_shot=max_num_few_shot,
        )
        eval_tasks.token_stat(eval_task_obj=eval_tasks, cache_dir=cache_dir, hf_id=hf_id)

    timer_end = time.perf_counter()
    logger.info("Total Running Time: %.1f sec (%.1f min)" % (timer_end - timer_start, (timer_end - timer_start) / 60))


if __name__ == "__main__":
    fire.Fire(main)
