# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import re
import json
from typing import Optional, Dict, Any

from datasets import load_dataset, Dataset

from tasks import EvalTaskManager


class EvalTaskBbh(EvalTaskManager):

    def __init__(
            self,
            verbose: bool,
            logger,
            cache_dir: Optional[str] = None,
            project_dir: Optional[str] = None,
            seed_few_shot: int = 42,
            max_num_few_shot: int = 10,
    ):
        super().__init__(verbose, logger, cache_dir, project_dir, seed_few_shot, max_num_few_shot)

        # Train = 0, Valid = 0, Test = 6511 (use 5511 [in which 230 for few-shot]; ignore 1000)
        # Features: ["input", "target"]
        # Eval: test[10:] set; Few-shot: test[:10]
        # Subtasks: 27 (use 23; ignore 4 since they are not multiple-choice QA tasks)

        self.task_name = "bbh"
        self.task_info = {
            "has_passage": False,
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["lukaemon/bbh", "word_sorting", "test"],  # Test = 250 (Ignore)
                ["lukaemon/bbh", "web_of_lies", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_three_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_seven_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "tracking_shuffled_objects_five_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "temporal_sequences", "test"],  # Test = 250
                ["lukaemon/bbh", "sports_understanding", "test"],  # Test = 250
                ["lukaemon/bbh", "snarks", "test"],  # Test = 178
                ["lukaemon/bbh", "salient_translation_error_detection", "test"],  # Test = 250
                ["lukaemon/bbh", "ruin_names", "test"],  # Test = 250
                ["lukaemon/bbh", "reasoning_about_colored_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "penguins_in_a_table", "test"],  # Test = 146
                ["lukaemon/bbh", "object_counting", "test"],  # Test = 250 (Ignore)
                ["lukaemon/bbh", "navigate", "test"],  # Test = 250
                ["lukaemon/bbh", "multistep_arithmetic_two", "test"],  # Test = 250 (Ignore)
                ["lukaemon/bbh", "movie_recommendation", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_three_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_seven_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "logical_deduction_five_objects", "test"],  # Test = 250
                ["lukaemon/bbh", "hyperbaton", "test"],  # Test = 250
                ["lukaemon/bbh", "geometric_shapes", "test"],  # Test = 250
                ["lukaemon/bbh", "formal_fallacies", "test"],  # Test = 250
                ["lukaemon/bbh", "dyck_languages", "test"],  # Test = 250 (Ignore)
                ["lukaemon/bbh", "disambiguation_qa", "test"],  # Test = 250
                ["lukaemon/bbh", "date_understanding", "test"],  # Test = 250
                ["lukaemon/bbh", "causal_judgement", "test"],  # Test = 187
                ["lukaemon/bbh", "boolean_expressions", "test"],  # Test = 250
            ],
        }
        self.ignore_subtasks = {"word_sorting", "object_counting", "dyck_languages", "multistep_arithmetic_two"}

        self.few_shot_fp = os.path.join(project_dir, "tasks", self.task_name, "few_shot.json")
        assert os.path.isfile(self.few_shot_fp)
        with open(self.few_shot_fp, "r", encoding="utf-8") as fp_in:
            self.few_shot = json.load(fp_in)

        self.options_fp = os.path.join(project_dir, "tasks", self.task_name, "options.json")
        assert os.path.isfile(self.options_fp)
        with open(self.options_fp, "r", encoding="utf-8") as fp_in:
            self.options = json.load(fp_in)

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

        option_pattern = re.compile(r"^\([A-Z]\)$")
        # ds_options = {}
        # few_shot_json = {}
        for hf_ds in hf_ds_list:
            assert isinstance(hf_ds, list) and len(hf_ds) == 3
            # self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}]")
            if hf_ds[1] in self.ignore_subtasks:  # ignore subtasks that are not multiple-choice QA
                self.logger.info(f">>> Ignore subtask: [dataset: {hf_ds[0]} --- {hf_ds[1]}]")
                continue

            try:  # Load the subset
                cur_ds = load_dataset(
                    hf_ds[0],
                    hf_ds[1],
                    cache_dir=os.path.join(self.cache_dir, "datasets"),
                    trust_remote_code=True,
                )

                eval_split = hf_ds[2]
                assert eval_split in cur_ds

                assert "test" in cur_ds
                len_train, len_valid, len_test = 0, 0, len(cur_ds["test"]) - self.max_num_few_shot
                assert len_test > 0
                few_shot_test = cur_ds["test"][:self.max_num_few_shot]
                eval_data_dict = cur_ds["test"][self.max_num_few_shot:]
                eval_dataset = Dataset.from_dict(eval_data_dict)

                options = list(set(list(eval_dataset["target"])))
                options = [str(_op).strip() for _op in options]
                # ensure the format of option labels (other: Yes/No, True/False, valid/invalid)
                if "(A)" in options:
                    options = [_op for _op in options if re.match(option_pattern, _op) is not None]
                options.sort()

                ds_dict = {
                    "hf_dataset": hf_ds[0],
                    "hf_subset": hf_ds[1],
                    "eval_split": eval_split,
                    "eval_dataset": eval_dataset,
                    "options": options,
                }

                # ds_options[hf_ds[1]] = options

                # cur_few_shot_list = []
                # input_list = few_shot_test["input"]
                # target_list = few_shot_test["target"]
                # assert len(input_list) == len(target_list)
                # for idx_fs in range(len(input_list)):
                #     cur_few_shot_list.append({
                #         "input": input_list[idx_fs],
                #         "target": target_list[idx_fs],
                #         "cot": "",
                #         "arr": "",
                #     })
                # few_shot_json[hf_ds[1]] = cur_few_shot_list

                self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}] [eval_split = {eval_split}] "
                                 f"Train = {len_train}, Validation = {len_valid}, Test = {len_test}")

                dataset["data"].append(ds_dict)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f">>> Exception: {e}")

        # few_shot_json_fp = os.path.join(self.project_dir, "tasks", self.task_name, "few_shot.json")
        # with open(few_shot_json_fp, "w", encoding="utf-8") as fp_out:
        #     json.dump(few_shot_json, fp_out, ensure_ascii=False, indent=2)

        # options_json_fp = os.path.join(self.project_dir, "tasks", self.task_name, "options.json")
        # with open(options_json_fp, "w", encoding="utf-8") as fp_out:
        #     json.dump(ds_options, fp_out, ensure_ascii=False, indent=2)

        self.logger.info(f">>> [task_name: {self.task_name}] len(dataset) = {len(dataset['data'])}\n\n")  # 27
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
    ) -> Optional[Dict[str, Any]]:
        assert isinstance(self.task_name, str) and self.task_name in self.all_tasks

        if subset in self.ignore_subtasks:  # ignore subtasks that are not multiple-choice QA
            self.logger.info(f">>> Ignore subtask: [dataset: {ds_name} --- {subset}]")
            return None

        # Load data
        hf_ds_list = self.task_info["hf_dataset"]
        has_passage = self.task_info["has_passage"]
        assert isinstance(hf_ds_list, list) and len(hf_ds_list) > 0 and isinstance(has_passage, bool)
        assert isinstance(self.few_shot, dict)
        if subset in self.few_shot:
            few_shot_list = self.few_shot[subset]
        else:
            assert ds_name in self.few_shot
            few_shot_list = self.few_shot[ds_name]
        assert isinstance(few_shot_list, list) and len(few_shot_list) > 0
        few_shot_list = few_shot_list[:num_few_shot]

        # Process data
        question = str(data_item["input"]).strip()
        answer = str(data_item["target"]).strip()

        # Get all options for the "target"
        assert subset in self.options
        options = list(self.options[subset])
        options.sort()

        answer_str = answer  # Metric: exact match (Yes/No; True/False; valid/invalid; (A)/(B)/(C)/...; etc.)
        answer_label = answer
        label_options = options
        answer_options = options
        _len_question = len("Question:")
        if question.startswith("Question:") or question.startswith("question:"):
            question = question[_len_question:].strip()

        # Set the main prompt (zero-shot)
        prompt_main = f"""
Question: {question}\nAnswer:
        """.strip()

        # Set few-shot examples (with reasoning/rationale)
        prompt_few_shot = ""
        if num_few_shot > 0:
            for fs_example in few_shot_list:
                question_fs = str(fs_example["input"]).strip()
                answer_fs = str(fs_example["target"]).strip()

                answer_str_fs = answer_fs
                if question_fs.startswith("Question:") or question_fs.startswith("question:"):
                    question_fs = question[_len_question:].strip()

                cur_prompt_fs = f"""
Question: {question_fs}\nAnswer:
                """.strip()
                if use_cot:
                    cot_reasoning = self.text_cleaning(str(fs_example["cot"]))
                    cur_prompt_fs += " Let's think step by step. " + cot_reasoning
                    cur_prompt_fs = (cur_prompt_fs.strip() +
                                     f" Therefore, the answer is: {answer_str_fs}")
                elif use_arr:
                    arr_reasoning = self.text_cleaning(str(fs_example["arr"]))
                    match arr_ablation:
                        # ARR components: using ARR = 000, 100, 010, 001, or 111
                        case "000":  # DA: "Answer:"
                            cur_prompt_fs = cur_prompt_fs.strip() + f" {answer_str_fs}"
                        case "001":  # only Reasoning
                            cur_prompt_fs += " Let's answer the question with step-by-step reasoning. " + arr_reasoning
                        case "010":  # only Retrieving
                            cur_prompt_fs += (" Let's find relevant information, "
                                              "and answer the question. ") + arr_reasoning
                        case "100":  # only Analyzing
                            cur_prompt_fs += (" Let's analyze the intent of the question, "
                                              "and answer the question. ") + arr_reasoning
                        case "111":  # ARR: Analyzing + Retrieving + Reasoning
                            cur_prompt_fs += (" Let's analyze the intent of the question, find relevant information, "
                                              "and answer the question with step-by-step reasoning. ") + arr_reasoning
                        # ARR prompt variants: using ARR = var1, var2, var3, var4, or var5
                        case "var1":  # ARR - prompt variant (paraphrase) 1
                            cur_prompt_fs += (" Let's identify the question's intent, "
                                              "gather the necessary information, and then "
                                              "work through a logical, step-by-step solution. ") + arr_reasoning
                        case "var2":  # ARR - prompt variant (paraphrase) 2
                            cur_prompt_fs += (" We'll begin by examining what the question is asking, "
                                              "then retrieve any relevant details, and finally provide "
                                              "a well-reasoned answer step by step. ") + arr_reasoning
                        case "var3":  # ARR - prompt variant (paraphrase) 3
                            cur_prompt_fs += (" First, we'll interpret the purpose behind the question, "
                                              "collect supporting information, and "
                                              "proceed to solve it methodically. ") + arr_reasoning
                        case "var4":  # ARR - prompt variant (paraphrase) 4
                            cur_prompt_fs += (" Let's break this down by understanding the goal of the question, "
                                              "pulling in the required data, and then "
                                              "reasoning through the answer in a clear sequence. ") + arr_reasoning
                        case "var5":  # ARR - prompt variant (paraphrase) 5
                            cur_prompt_fs += (" To approach this, we'll clarify the question's intent, "
                                              "locate pertinent information, and then build our answer using "
                                              "structured, logical reasoning. ") + arr_reasoning
                        case _:
                            raise ValueError(f"ValueError: Unexpected value for arr_ablation: {arr_ablation}")
                    if arr_ablation != "000":
                        cur_prompt_fs = cur_prompt_fs.strip() + f" Therefore, the answer is: {answer_str_fs}"
                else:
                    cur_prompt_fs = cur_prompt_fs.strip() + f" {answer_str_fs}"
                prompt_few_shot += f"{cur_prompt_fs}\n\n"

        if use_cot:
            prompt_main += f" Let's think step by step."
        elif use_arr:
            match arr_ablation:
                # ARR components: using ARR = 000, 100, 010, 001, or 111
                case "000":  # DA: "Answer:"
                    pass
                case "001":  # only Reasoning
                    prompt_main += f" Let's answer the question with step-by-step reasoning."
                case "010":  # only Retrieving
                    prompt_main += f" Let's find relevant information, and answer the question."
                case "100":  # only Analyzing
                    prompt_main += f" Let's analyze the intent of the question, and answer the question."
                case "111":  # ARR: Analyzing + Retrieving + Reasoning
                    prompt_main += (f" Let's analyze the intent of the question, "
                                    f"find relevant information, and answer the question with step-by-step reasoning.")
                # ARR prompt variants: using ARR = var1, var2, var3, var4, or var5
                case "var1":  # ARR - prompt variant (paraphrase) 1
                    prompt_main += (f" Let's identify the question's intent, gather the necessary information, "
                                    f"and then work through a logical, step-by-step solution.")
                case "var2":  # ARR - prompt variant (paraphrase) 2
                    prompt_main += (f" We'll begin by examining what the question is asking, then retrieve any "
                                    f"relevant details, and finally provide a well-reasoned answer step by step.")
                case "var3":  # ARR - prompt variant (paraphrase) 3
                    prompt_main += (f" First, we'll interpret the purpose behind the question, collect "
                                    f"supporting information, and proceed to solve it methodically.")
                case "var4":  # ARR - prompt variant (paraphrase) 4
                    prompt_main += (f" Let's break this down by understanding the goal of the question, pulling in "
                                    f"the required data, and then reasoning through the answer in a clear sequence.")
                case "var5":  # ARR - prompt variant (paraphrase) 5
                    prompt_main += (f" To approach this, we'll clarify the question's intent, locate pertinent "
                                    f"information, and then build our answer using structured, logical reasoning.")
                case _:
                    raise ValueError(f"ValueError: Unexpected value for arr_ablation: {arr_ablation}")
        else:
            pass

        prompt = prompt_few_shot + prompt_main

        # Set the result dict
        result_dict = {
            "prompt": prompt,
            "answer_str": answer_str,
            "answer_label": answer_label,
            "label_options": label_options,
            "answer_options": answer_options,
        }
        return result_dict
