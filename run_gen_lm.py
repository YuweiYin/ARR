#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import time
import json
from typing import Optional

import fire
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

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


class LMGen:

    def __init__(
            self,
            verbose: bool,
            logger,
            cuda_dict: dict,
            seed: int = 42,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            hf_id: str = "meta-llama/Llama-3.1-8B-Instruct",
            bsz: int = 1,
            show_generation: bool = False,
            debug: bool = False,
            output_dir: Optional[str] = None,
            max_gen_len: int = 512,
            gen_temperature: float = 0.0,
            num_few_shot: int = 0,
            seed_few_shot: int = 42,
            max_num_few_shot: int = 10,
            use_cot: bool = False,
            use_arr: bool = False,
            arr_ablation: str = "111",
    ):
        self.verbose = verbose
        self.logger = logger
        self.cuda_dict = cuda_dict
        self.seed = seed
        self.hf_id = hf_id
        self.hf_name = "--".join(hf_id.split("/"))
        self.show_generation = show_generation  # If True, show outputs during generation
        self.debug = debug

        if isinstance(project_dir, str) and os.path.isdir(project_dir):
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd()
        assert os.path.isdir(project_dir)

        self.output_dir = output_dir
        self.bsz = bsz
        self.max_gen_len = max_gen_len
        self.gen_temperature = gen_temperature
        self.num_few_shot = num_few_shot
        self.seed_few_shot = seed_few_shot
        self.max_num_few_shot = max_num_few_shot
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

        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir
        self.model_path = os.path.join(
            self.cache_dir, "models--" + self.hf_name, "snapshots/model")
        assert os.path.isdir(self.model_path), f"AssertionError: assert os.path.isdir({self.model_path})"

        # Tokenizer and LLM model
        self.tokenizer = self.load_tokenizer(model_path=self.model_path, padding_side="left", truncation_side="left")
        self.terminators_gen = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
        ]
        self.model = None

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def load_tokenizer(
            self,
            model_path,
            padding_side="left",
            truncation_side="left",
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side=padding_side,
            truncation_side=truncation_side,  # "right" for training, "left" for generating
            cache_dir=self.cache_dir,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        max_len = tokenizer.max_len_single_sentence
        if self.verbose:
            self.logger.info(
                f">>> len(tokenizer.vocab) = {len(tokenizer.vocab)}; "
                f"tokenizer.max_len_single_sentence = {max_len}")  # LLaMA-3: 131071

        return tokenizer

    def load_model(
            self,
            model_path,
            tokenizer,
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # torch.bfloat16
            # torch_dtype=torch.float8_e5m2,  # torch.float8
            device_map="auto",  # !pip install accelerate
            # device_map=self.cuda_dict["device"] if self.debug else "auto",
            # device_map=self.device_mps if self.debug else "auto",
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            # local_files_only=True,
        )
        # model = model.to(device=self.cuda_dict["device"])
        # list(model.state_dict().keys())
        model.generation_config.pad_token_id = tokenizer.pad_token_id  # eos_token_id
        # model.resize_token_embeddings(len(self.tokenizer_train))  # if added new special tokens
        # model.train()
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.verbose:
            self.logger.info(f">>> Base Model loaded: {model_path}")
            self.logger.info(f">>> [Base Model] Number of total parameters: {total_params}")
            self.logger.info(f">>> [Base Model] Number of trainable parameters: {train_params}")

        return model

    def run_generation(
            self,
            prompts,
            model,
            tokenizer,
            need_tokenize: bool = True,
    ) -> dict:
        if need_tokenize:
            input_ids = self.tokenizer(
                prompts,
                padding=True,  # truncation=True, max_length=512
                return_tensors="pt",
            ).to(model.device)  # batch_size=1
        else:
            input_ids = prompts
            input_ids = input_ids.to(model.device)
        len_input = input_ids.data["input_ids"].size(-1)

        with torch.no_grad():
            # https://huggingface.co/docs/transformers/en/main_classes/text_generation
            assert self.max_gen_len > 0
            outputs = model.generate(
                **input_ids,
                max_new_tokens=self.max_gen_len,
                eos_token_id=self.terminators_gen,
                do_sample=self.gen_temperature > 0.0,
                # do_sample=True,  # False: greedy decoding (the most deterministic)
                temperature=self.gen_temperature if self.gen_temperature > 0.0 else None,  # defaults to 1.0
                # top_p=0.9,  # defaults to 1.0
                # output_attentions=False,
                # output_hidden_states=False,
                # output_scores=True,
                output_logits=True,
                return_dict_in_generate=True,
            )
        output_ids = outputs["sequences"]
        output_text = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        input_text = tokenizer.batch_decode(
            input_ids["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=True)

        assert len(input_text) == len(prompts) == len(output_text)
        output_text_pure = []
        for _input, _prompt, _output in zip(input_text, prompts, output_text):
            output_pure = _output[len(_input):]
            output_text_pure.append(output_pure)
            if self.verbose and self.show_generation:
                # self.logger.info("================================== >>> input (raw) <<<")
                # self.logger.info(_input)
                # self.logger.info("================================== >>> prompt <<<")
                # self.logger.info(_prompt)
                self.logger.info("================================== >>> output <<<")
                self.logger.info(output_pure)

        return {
            "prompts": prompts,
            "len_input": len_input,
            # "input_ids": input_ids,
            "input_text": input_text,
            "outputs": outputs,
            # "output_ids": output_ids,
            # "output_text": output_text,
            "output_text": output_text_pure,
        }

    def lm_generate(
            self,
            eval_task_name: str,
    ):
        # Stage 1: Reasoning Generation:
        #   Load QA datasets, load the model, set input prompts, freely generation,
        #   and save results to JSON files (task/dataset information, input, and output)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

        # Load the model
        if self.model is None:
            model = self.load_model(model_path=self.model_path, tokenizer=self.tokenizer)
            self.model = model
        else:
            model = self.model

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

            if "options" in dataset_dict:
                ds_options = list(dataset_dict["options"])
            else:
                ds_options = []

            # Run generation with batch_size = 1
            for idx, data_item in enumerate(eval_dataset):
                assert isinstance(data_item, dict)
                data_item["__ds_options"] = ds_options
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

                # Run generation (batch_size = 1)
                prompts = [prompt_dict["prompt"]]
                gen_dict = self.run_generation(
                    prompts=prompts, model=model, tokenizer=self.tokenizer, need_tokenize=True)
                cur_gen_output = {
                    "index": idx,
                    "prompt": prompt_dict["prompt"],  # The input prompt
                    "len_input": int(gen_dict["len_input"]),  # Number of tokens of the model input/prompt
                    "output_text": str(gen_dict["output_text"][0]).strip(),  # The LLM output (excluding the input)
                    "answer_str": prompt_dict["answer_str"],  # The golden answer (full string)
                    "answer_label": prompt_dict["answer_label"],  # The golden answer label (like 0 or "A")
                    "label_options": prompt_dict["label_options"],  # The options of all answer labels
                    "answer_options": prompt_dict["answer_options"],  # The options of all answer strings
                }

                cur_results.append(cur_gen_output)
                if self.verbose and len(cur_results) % show_cnt == 0:
                    self.logger.info(f">>> Progress: [{ds_id}] [{len(cur_results)} / {len_dataset}]")

            all_results[ds_id] = cur_results

        # Save the generation outputs and show logs
        output_dir = os.path.join(self.output_dir, eval_task_name, self.hf_name)
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
            f">>> DONE ALL. hf_id = {self.hf_id}; model_path = {self.model_path}\n"
            f"num_few_shot: {self.num_few_shot}, "
            f"use_cot: {self.use_cot}, use_arr: {self.use_arr}, "
            f"gen_temperature: {self.gen_temperature}, batch_size: {self.bsz}"
        )


def main(
    task: int = 0,
    eval_task_name: Optional[str] = None,
    hf_id: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    cuda: Optional[str] = None,
    bsz: int = 1,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    max_gen_len: int = 512,
    gen_temperature: float = 0.0,
    num_few_shot: int = 0,
    seed_few_shot: int = 42,
    max_num_few_shot: int = 10,
    use_cot: bool = False,
    use_arr: bool = False,
    arr_ablation: str = "111",
    **kwargs
) -> None:
    """
    Stage 1: Reasoning Generation. Let LLMs freely generate reasoning for later evaluation.

    :param task: 1. LM generation.
    :param eval_task_name: The name(s) of the evaluation task. (e.g., "boolq", "bbh", and "boolq,bbh")
    :param hf_id: ORGANIZATION_NAME/MODEL_NAME, e.g., "meta-llama/Llama-3.1-8B-Instruct"
    :param cache_dir: The root directory of the cache.
    :param project_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param cuda: To specify CUDA GPU devices, e.g., "0" OR "0,1". Default: None -- Use CPU or all available GPUs.
    :param bsz: The batch size.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param max_gen_len: The maximum number of newly generated tokens.
    :param gen_temperature: The temperature used in LLM generation. Default: 0.0
    :param num_few_shot: The number of few-shot examples to provide. Default: 0
    :param seed_few_shot: Random seed for sampling few-shot examples.
    :param max_num_few_shot: The maximum number of few-shot examples.
    :param use_cot: Use chain-of-thought prompting (providing CoT reasoning/rationale in the few-shot examples) or not.
    :param use_arr: Use our ARR method (providing ARR reasoning/rationale in the few-shot examples) or not.
        ARR: Analyzer, Retriever, and Reasoner
    :param arr_ablation: The ablation study of ARR prompting: 000 --> no A, no R, no R; 101 --> use A, no R, use R
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_Gen")
    cuda_dict = cuda_setup(cuda=cuda, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}\n")

    if isinstance(cache_dir, str) and os.path.isdir(cache_dir):
        os.environ["HF_HOME"] = cache_dir
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "datasets")
        # os.environ["HF_HOME"] = os.path.join(cache_dir, "hub")
    else:
        cache_dir = None

    lm_gen = LMGen(
        verbose=verbose,
        logger=logger,
        cuda_dict=cuda_dict,
        seed=seed,
        cache_dir=cache_dir,
        project_dir=project_dir,
        hf_id=hf_id,
        bsz=max(int(bsz), 1),
        debug=debug,
        output_dir=output_dir,
        max_gen_len=int(max_gen_len),
        gen_temperature=float(gen_temperature),
        num_few_shot=int(num_few_shot),
        seed_few_shot=int(seed_few_shot),
        max_num_few_shot=int(max_num_few_shot),
        use_cot=use_cot,
        use_arr=use_arr,
        arr_ablation=str(arr_ablation).zfill(3),
    )

    task = int(task)
    match task:
        case 1:
            if isinstance(eval_task_name, tuple) or isinstance(eval_task_name, list):
                for cur_task_name in eval_task_name:
                    cur_task_name = str(cur_task_name).strip()
                    logger.info(f">>> <START> {cur_task_name}\n")
                    lm_gen.lm_generate(eval_task_name=cur_task_name)
                    logger.info(f">>> <END> {cur_task_name}\n\n\n")
            elif isinstance(eval_task_name, str):
                eval_task_name = str(eval_task_name).strip()
                logger.info(f">>> <START> {eval_task_name}\n")
                lm_gen.lm_generate(eval_task_name=eval_task_name)
                logger.info(f">>> <END> {eval_task_name}\n\n\n")
            else:
                raise ValueError(f"--eval_task_name should be a tuple/list/str: {eval_task_name}")
        case _:
            raise ValueError(f"ValueError: task = {task}")

    timer_end = time.perf_counter()
    total_sec = timer_end - timer_start
    logger.info(f"Total Running Time: {total_sec:.1f} sec ({total_sec / 60:.1f} min; {total_sec / 3600:.2f} h)")


if __name__ == "__main__":
    fire.Fire(main)
