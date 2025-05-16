#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import re
import time
import json
import string
from typing import Optional, List

import fire
import numpy as np

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
        self.show_generation = show_generation  # If True, show outputs during generation
        self.debug = debug

        if isinstance(project_dir, str) and os.path.isdir(project_dir):
            self.project_dir = project_dir
        else:
            self.project_dir = os.getcwd()
        assert os.path.isdir(project_dir)

        self.output_dir = output_dir
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
            # self.cache_dir = os.path.join(self.project_dir, ".cache/huggingface/")
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)
        if self.verbose:
            self.logger.info(f">>> cache_dir: {self.cache_dir}")

        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir
        self.punc_remover = str.maketrans("", "", string.punctuation)  # r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        self.space_remover = str.maketrans("", "", string.whitespace)  # " \t\n\r\v\f"

    @staticmethod
    def _handle_non_serializable(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        elif isinstance(o, set):
            return list(o)
        else:
            return str(o)

    def compute_exact_match(
            self,
            prediction: str,
            references: List[str],
    ) -> float:
        prediction = str(prediction).strip()
        references = [str(_ref).strip() for _ref in references]

        # Matching anyone in the references will have an EM score of 1; otherwise 0.
        for ref in references:
            if prediction == ref:
                return float(1.0)

        # Normalize strings and then match.  # " ".join(prediction.split())  # Clear adjacent whitespaces
        prediction_new, references_new = prediction, references
        prediction_new = prediction_new.translate(self.punc_remover).strip()  # Remove all punctuations
        prediction_new = prediction_new.translate(self.space_remover).strip()  # Remove all whitespaces
        # prediction_new = prediction_new.replace(" ", "").strip()
        references_new = [_ref.translate(self.punc_remover).strip() for _ref in references_new]
        references_new = [_ref.translate(self.space_remover).strip() for _ref in references_new]
        # references_new = [_ref.replace(" ", "").strip() for _ref in references_new]
        for ref in references_new:
            if prediction_new == ref:
                return float(1.0)

        return float(0.0)

    def lm_evaluate(
            self,
            eval_task_name: str,
    ):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Load the generation outputs
        assert isinstance(self.output_dir, str) and os.path.isdir(self.output_dir), "Please specify --output_dir"
        output_dir = os.path.join(self.output_dir, eval_task_name, "gpt-4o")
        output_fp = os.path.join(output_dir, "results_gen.json")
        # assert os.path.isfile(output_fp), f"Assertion Error: output_fp does not exist: {output_fp}"
        if not os.path.isfile(output_fp):
            self.logger.info(
                f">>> output_dir: {output_dir}, use_gen_output: {self.use_gen_output}.\n"
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
                    f">>> output_dir: {output_dir}, use_gen_output: {self.use_gen_output}.\n"
                    f">>> !!! >>> [SKIP; No --overwrite] File already exists: {output_eval_fp}"
                )
                return
        else:
            self.logger.info(f"Results will be saved at: {output_eval_fp}")

        # re_option = re.compile(r"([A-Z])|(\([A-Z]\))|([A-Z]\))")  # Match options: A, B, C, D, etc.
        re_option_0 = re.compile(r"^[A-Z]$")
        re_option_1 = re.compile(r"(\([A-Z]\))|([A-Z]\))")  # Match options: A, B, C, D, etc.
        re_option_2 = re.compile(r"(Yes)|(yes)|(No)|(no)|(valid)|(invalid)|(Valid)|(Invalid)|(True)|(False)|(true)|(false)")

        # Deal with each task (and sub-tasks)
        all_scores = {}
        correct_num, wrong_num = 0, 0
        miss_final_cnt = 0
        show_cnt = 100
        for ds_id, cur_results in gen_results.items():
            len_dataset = len(cur_results)
            if self.verbose:
                self.logger.info(f">>> [Dataset: {ds_id}] [Eval: # = {len_dataset}")
            assert isinstance(cur_results, list) and len(cur_results) == len_dataset > 0

            # Run generation with batch_size = 1
            cur_ds_scores = []
            correct_idx, wrong_idx = [], []
            for idx, cur_res_dict in enumerate(cur_results):
                # gen_prompt = str(cur_res_dict["prompt"]).strip()
                gen_output = str(cur_res_dict["output_text"]).strip()
                answer_str = str(cur_res_dict["answer_str"]).strip()
                # answer_options = cur_res_dict["answer_options"]
                answer_label = str(cur_res_dict["answer_label"]).strip()
                # label_options = cur_res_dict["label_options"]

                answers = [answer_str, answer_label, f"({answer_label})", f"{answer_label})", f"{answer_label}."]
                if "Final Answer:" in gen_output:
                    pred_final = gen_output.split("Final Answer:")[-1].strip()
                    pred_options_1 = re.findall(re_option_1, pred_final)
                    pred_options_2 = re.findall(re_option_2, pred_final)
                    pred_options = []
                    for po in pred_options_1 + pred_options_2:
                        if isinstance(po, str) and len(po) > 0:
                            pred_options.append(po.strip())
                        elif isinstance(po, tuple):
                            pred_options.extend(list(po))
                        else:
                            pass
                    if re.match(re_option_0, pred_final):  # The final answer is "A", "B", or "C", etc.
                        pred_options += re.findall(re_option_0, pred_final)
                    pred_options = [str(po).strip() for po in pred_options]
                    pred_options = [po for po in pred_options if len(po) > 0]
                    pred_options = list(set(pred_options))
                    if len(pred_options) == 0:
                        cur_eval_score = 0.0
                        miss_final_cnt += 1
                    else:
                        cur_eval_score = max([self.compute_exact_match(po, answers) for po in pred_options])
                else:
                    cur_eval_score = 0.0
                    miss_final_cnt += 1

                if cur_eval_score == 1.0:
                    correct_idx.append(idx)
                elif cur_eval_score == 0.0:
                    wrong_idx.append(idx)
                else:
                    raise ValueError(f"ValueError: cur_eval_score = {cur_eval_score}")

                cur_res_dict["eval_metric"] = "Acc"
                cur_res_dict["eval_score"] = cur_eval_score

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
                         f"# total correct = {correct_num}; # total wrong = {wrong_num}; "
                         f"# miss_final_cnt = {miss_final_cnt}")

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
            f">>> output_dir: {output_dir}, use_gen_output: {self.use_gen_output}."
        )


def main(
    task: int = 0,
    eval_task_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    project_dir: Optional[str] = None,
    seed: int = 42,
    verbose: bool = False,
    debug: bool = False,
    output_dir: Optional[str] = None,
    use_gen_output: bool = False,
    overwrite: bool = False,
    **kwargs
) -> None:
    """
    Stage 2: Option Selection. Evaluate GPT outputs on multiple-choice QA tasks.

    :param task: 1. language model evaluation.
    :param eval_task_name: The name(s) of the evaluation task. (e.g., "boolq", "bbh", and "boolq,bbh")
    :param cache_dir: The root directory of the cache.
    :param project_dir: The root directory of the current project/repo.
    :param seed: Random seed of all modules.
    :param verbose: Verbose mode: show logs.
    :param debug: Debugging / developing mode.
    :param output_dir: The path to the output file where the result metrics will be saved.
    :param use_gen_output: Use the generated outputs or not.
    :param overwrite: Overwrite existing output files.
    :return: None.
    """

    timer_start = time.perf_counter()

    # Setup of the logger, CUDA gpus, and random seed
    logger = logger_setup("GPT_Eval")
    cuda_dict = cuda_setup(cuda=None, logger=logger, verbose=verbose)
    random_setup(seed=seed, has_cuda=cuda_dict["has_cuda"])

    if isinstance(kwargs, dict):
        logger.info(f">>> Unused parameters in kwargs: {kwargs}")

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
