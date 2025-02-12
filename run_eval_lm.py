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


class LMEval:

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
            use_gen_output: bool = False,
            overwrite: bool = False,
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
        self.use_gen_output = use_gen_output
        self.overwrite = overwrite

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
            # local_files_only=True,
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
        # model.resize_token_embeddings(len(self.tokenizer_train))  # if added new special tokens (Option 1)
        # model.train()
        model.eval()
        total_params = sum(p.numel() for p in model.parameters())
        train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if self.verbose:
            self.logger.info(f">>> Base Model loaded: {model_path}")
            self.logger.info(f">>> [Base Model] Number of total parameters: {total_params}")
            self.logger.info(f">>> [Base Model] Number of trainable parameters: {train_params}")

        return model

    def run_language_modeling(
            self,
            prompts,
            model,
            tokenizer,
            need_tokenize: bool = True,
    ) -> dict:
        if need_tokenize:
            input_ids = self.tokenizer(
                prompts,
                padding=True,  # truncation=True, max_length=1024,
                return_tensors="pt",
            ).to(model.device)  # batch_size=1
        else:
            input_ids = prompts
            input_ids = input_ids.to(model.device)
        len_input = input_ids.data["input_ids"].size(-1)
        target_ids = input_ids["input_ids"].to(model.device)
        input_ids.data["labels"] = target_ids

        with torch.no_grad():
            outputs = model(**input_ids)
        output_ids = outputs["logits"].argmax(-1)
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

    def lm_evaluate(
            self,
            eval_task_name: str,
    ):
        # Stage 2: Option Section (Multiple-choice QA Evaluation):
        #   Load the result JSON file in the first stage (reasoning generation),
        #   concatenate context, question, options, reasoning, and each option,
        #   compute the language model losses, and select the option with the lowest LM loss

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load the generation outputs
        assert isinstance(self.output_dir, str) and os.path.isdir(self.output_dir), "Please specify --output_dir"
        output_dir = os.path.join(self.output_dir, eval_task_name, self.hf_name)
        output_fp = os.path.join(output_dir, "results_gen.json")
        if not os.path.isfile(output_fp):
            self.logger.info(
                f">>> hf_id = {self.hf_id}; model_path = {self.model_path}\n"
                f"output_dir: {output_dir}, use_gen_output: {self.use_gen_output}.\n"
                f">>> !!! >>> [SKIP; No --output_fp] output_fp does not exist: {output_fp}"
            )
            return
        with open(output_fp, "r", encoding="utf-8") as fp_in:
            gen_results = json.load(fp_in)

        # Set the saving filepath
        if self.use_gen_output:
            output_eval_fp = os.path.join(output_dir, "results_eval-use_gen.json")
        else:
            output_eval_fp = os.path.join(output_dir, "results_eval.json")
        if os.path.exists(output_eval_fp):
            if self.overwrite:
                self.logger.info(f"Results will be overwritten: {output_eval_fp}")
            else:
                self.logger.info(
                    f">>> hf_id = {self.hf_id}; model_path = {self.model_path}\n"
                    f"output_dir: {output_dir}, use_gen_output: {self.use_gen_output}.\n"
                    f">>> !!! >>> [SKIP; No --overwrite] File already exists: {output_eval_fp}"
                )
                return
        else:
            self.logger.info(f"Results will be saved at: {output_eval_fp}")

        assert eval_task_name in self.task_class_dict, \
            f"AssertionError: task name {eval_task_name} not in task_class_dict"
        eval_task_class = self.task_class_dict[eval_task_name]

        eval_task_obj = eval_task_class(
            verbose=self.verbose,
            logger=self.logger,
            cache_dir=self.cache_dir,
            project_dir=self.project_dir,
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
        all_scores = {}
        correct_num, wrong_num = 0, 0
        show_cnt = 100
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

            assert ds_id in gen_results
            cur_results = gen_results[ds_id]
            assert isinstance(cur_results, list) and len(cur_results) == len_dataset > 0

            # Run generation with batch_size = 1
            cur_ds_scores = []
            correct_idx, wrong_idx = [], []
            for idx, cur_res_dict in enumerate(cur_results):
                gen_prompt = str(cur_res_dict["prompt"]).strip()
                gen_output = str(cur_res_dict["output_text"]).strip()
                answer_str = cur_res_dict["answer_str"]
                answer_options = cur_res_dict["answer_options"]
                answer_label = cur_res_dict["answer_label"]
                label_options = cur_res_dict["label_options"]

                if isinstance(answer_options, list) and len(answer_options) > 0:
                    if answer_str not in answer_options:
                        wrong_idx.append(idx)  # exception case, call it error
                        cur_res_dict["eval_losses"] = []
                        cur_res_dict["eval_metric"] = "Acc"
                        cur_res_dict["eval_score"] = 0.0
                        cur_ds_scores.append(cur_res_dict)
                        self.logger.info(f">>> !!! >>> answer_str ({answer_str}) "
                                         f"not in answer_options: {answer_options}")
                        continue
                    assert answer_str in answer_options
                    concat_options = answer_options
                    correct_answer = answer_str
                    correct_index = int(answer_options.index(correct_answer))
                elif isinstance(label_options, list) and len(label_options) > 0:
                    if answer_label not in label_options:
                        wrong_idx.append(idx)  # exception case, call it error
                        cur_res_dict["eval_losses"] = []
                        cur_res_dict["eval_metric"] = "Acc"
                        cur_res_dict["eval_score"] = 0.0
                        cur_ds_scores.append(cur_res_dict)
                        self.logger.info(f">>> !!! >>> answer_label ({answer_label}) "
                                         f"not in answer_options: {answer_options}")
                        continue
                    assert answer_label in label_options
                    concat_options = label_options
                    correct_answer = answer_label
                    correct_index = int(label_options.index(correct_answer))
                else:
                    raise ValueError(f"ValueError: data item format. answer_str: {answer_str}")

                if isinstance(concat_options, list):
                    # Accuracy score: select the option with the lowest LLM perplexity / avg nll_loss
                    assert isinstance(correct_index, int)
                    if self.use_gen_output:
                        concat_prompts = [f"{gen_prompt} {gen_output}\nTherefore, the answer is: {_op}"
                                          for _op in concat_options]
                    else:
                        concat_prompts = [f"{gen_prompt}\nThe answer is: {_op}" for _op in concat_options]

                    # Run language modeling (batch_size = 1) --> obtain nll loss / perplexity
                    eval_losses = []
                    for concat_prompt in concat_prompts:
                        eval_dict = self.run_language_modeling(
                            prompts=[concat_prompt], model=model, tokenizer=self.tokenizer, need_tokenize=True)
                        eval_losses.append(eval_dict["outputs"]["loss"].cpu().detach().numpy().item())
                    eval_choice = int(np.argmin(eval_losses).item())
                    if eval_choice == correct_index:
                        cur_eval_score = 1.0
                        correct_idx.append(idx)
                    else:
                        cur_eval_score = 0.0
                        wrong_idx.append(idx)
                    cur_res_dict["eval_losses"] = eval_losses
                    cur_res_dict["eval_metric"] = "Acc"
                    cur_res_dict["eval_score"] = cur_eval_score
                else:
                    raise ValueError(f"ValueError: `concat_options` is not a list: {concat_options}")

                cur_ds_scores.append(cur_res_dict)
                if self.verbose and len(cur_ds_scores) % show_cnt == 0:
                    self.logger.info(f">>> Progress: [{ds_id}] [{len(cur_ds_scores)} / {len_dataset}]")

            cur_acc = float(len(correct_idx) / len_dataset)
            all_scores[ds_id] = {
                "acc": cur_acc,
                "correct_idx": correct_idx,
                "wrong_idx": wrong_idx,
                "ds_scores": cur_ds_scores,
            }
            correct_num += len(correct_idx)
            wrong_num += len(wrong_idx)
            if self.verbose:
                self.logger.info(f">>> [{ds_id}] [Acc = {cur_acc}] "
                                 f"# correct = {len(correct_idx)}; # wrong = {len(wrong_idx)}\n")

        # collect all acc
        acc_all = [float(v["acc"]) for k, v in all_scores.items()]
        assert len(acc_all) > 0
        acc_overall = sum(acc_all) / len(acc_all)
        all_scores["__acc_overall__"] = acc_overall
        self.logger.info(f">>> DONE ALL. [{eval_task_name}] [Acc_overall = {acc_overall}] "
                         f"# total correct = {correct_num}; # total wrong = {wrong_num}")

        # Save the generation outputs and show logs
        dumped = json.dumps(
            all_scores,
            indent=2,  # indent=None,
            default=self._handle_non_serializable,
            ensure_ascii=True,
        )
        with open(output_eval_fp, "w", encoding="utf-8") as fp_out:
            fp_out.write(dumped)
        self.logger.info(
            f">>> hf_id = {self.hf_id}; model_path = {self.model_path}\n"
            f"output_dir: {output_dir}, use_gen_output: {self.use_gen_output}."
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
    use_gen_output: bool = False,
    overwrite: bool = False,
    **kwargs
) -> None:
    """
    Stage 2: Option Selection. Evaluate LLMs on multiple-choice QA tasks.

    :param task: 1. LM evaluation on multiple-choice QA tasks.
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
    :param use_gen_output: Use the generated outputs or not.
    :param overwrite: Overwrite existing output files.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("LM_Eval")
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

    lm_eval = LMEval(
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
        use_gen_output=use_gen_output,
        overwrite=overwrite,
    )

    task = int(task)
    match task:
        case 1:
            if isinstance(eval_task_name, tuple) or isinstance(eval_task_name, list):
                for cur_task_name in eval_task_name:
                    cur_task_name = str(cur_task_name).strip()
                    logger.info(f">>> <START> {cur_task_name}\n")
                    lm_eval.lm_evaluate(eval_task_name=cur_task_name)
                    logger.info(f">>> <END> {cur_task_name}\n\n\n")
            elif isinstance(eval_task_name, str):
                eval_task_name = str(eval_task_name).strip()
                logger.info(f">>> <START> {eval_task_name}\n")
                lm_eval.lm_evaluate(eval_task_name=eval_task_name)
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
