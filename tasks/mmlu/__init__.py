# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import json
# import random
from typing import Optional, Dict, Any

from datasets import load_dataset

from tasks import EvalTaskManager


class EvalTaskMmlu(EvalTaskManager):

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

        # Train = 0, Validation = 1514, Test = 13842
        # Features: ["question", "subject", "choices", "answer"]
        # Eval: test set; Few-shot: valid set

        self.task_name = "mmlu"
        self.task_info = {
            "has_passage": False,
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["hails/mmlu_no_train", "formal_logic", "test"],  # Validation = 14, Test = 126
                ["hails/mmlu_no_train", "philosophy", "test"],  # Validation = 34, Test = 311
                ["hails/mmlu_no_train", "high_school_world_history", "test"],  # Validation = 26, Test = 237
                ["hails/mmlu_no_train", "international_law", "test"],  # Validation = 13, Test = 121
                ["hails/mmlu_no_train", "jurisprudence", "test"],  # Validation = 11, Test = 108
                ["hails/mmlu_no_train", "world_religions", "test"],  # Validation = 19, Test = 171
                ["hails/mmlu_no_train", "moral_disputes", "test"],  # Validation = 38, Test = 346
                ["hails/mmlu_no_train", "high_school_european_history", "test"],  # Validation = 18, Test = 165
                ["hails/mmlu_no_train", "logical_fallacies", "test"],  # Validation = 18, Test = 163
                ["hails/mmlu_no_train", "high_school_us_history", "test"],  # Validation = 22, Test = 204
                ["hails/mmlu_no_train", "moral_scenarios", "test"],  # Validation = 100, Test = 895
                ["hails/mmlu_no_train", "professional_law", "test"],  # Validation = 170, Test = 1534
                ["hails/mmlu_no_train", "prehistory", "test"],  # Validation = 35, Test = 324
                ["hails/mmlu_no_train", "us_foreign_policy", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "security_studies", "test"],  # Validation = 27, Test = 245
                ["hails/mmlu_no_train", "econometrics", "test"],  # Validation = 12, Test = 114
                ["hails/mmlu_no_train", "high_school_microeconomics", "test"],  # Validation = 26, Test = 238
                ["hails/mmlu_no_train", "sociology", "test"],  # Validation = 22, Test = 201
                ["hails/mmlu_no_train", "high_school_geography", "test"],  # Validation = 22, Test = 198
                ["hails/mmlu_no_train", "high_school_psychology", "test"],  # Validation = 60, Test = 545
                ["hails/mmlu_no_train", "professional_psychology", "test"],  # Validation = 69, Test = 612
                ["hails/mmlu_no_train", "high_school_macroeconomics", "test"],  # Validation = 43, Test = 390
                ["hails/mmlu_no_train", "high_school_government_and_politics", "test"],  # Validation = 21, Test = 193
                ["hails/mmlu_no_train", "public_relations", "test"],  # Validation = 12, Test = 110
                ["hails/mmlu_no_train", "human_sexuality", "test"],  # Validation = 12, Test = 131
                ["hails/mmlu_no_train", "miscellaneous", "test"],  # Validation = 86, Test = 783
                ["hails/mmlu_no_train", "medical_genetics", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "management", "test"],  # Validation = 11, Test = 103
                ["hails/mmlu_no_train", "virology", "test"],  # Validation = 18, Test = 166
                ["hails/mmlu_no_train", "nutrition", "test"],  # Validation = 33, Test = 306
                ["hails/mmlu_no_train", "global_facts", "test"],  # Validation = 10, Test = 100
                ["hails/mmlu_no_train", "marketing", "test"],  # Validation = 25, Test = 234
                ["hails/mmlu_no_train", "college_medicine", "test"],  # Validation = 22, Test = 173
                ["hails/mmlu_no_train", "clinical_knowledge", "test"],  # Validation = 29, Test = 265
                ["hails/mmlu_no_train", "professional_accounting", "test"],  # Validation = 31, Test = 282
                ["hails/mmlu_no_train", "professional_medicine", "test"],  # Validation = 31, Test = 272
                ["hails/mmlu_no_train", "human_aging", "test"],  # Validation = 23, Test = 223
                ["hails/mmlu_no_train", "business_ethics", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "college_physics", "test"],  # Validation = 11, Test = 102
                ["hails/mmlu_no_train", "elementary_mathematics", "test"],  # Validation = 41, Test = 378
                ["hails/mmlu_no_train", "machine_learning", "test"],  # Validation = 11, Test = 112
                ["hails/mmlu_no_train", "high_school_statistics", "test"],  # Validation = 23, Test = 216
                ["hails/mmlu_no_train", "electrical_engineering", "test"],  # Validation = 16, Test = 145
                ["hails/mmlu_no_train", "college_computer_science", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "anatomy", "test"],  # Validation = 14, Test = 135
                ["hails/mmlu_no_train", "high_school_physics", "test"],  # Validation = 17, Test = 151
                # ["hails/mmlu_no_train", "high_school_computer_science", "test"],  # Valid = 9 (ignore), Test = 100
                ["hails/mmlu_no_train", "computer_security", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "conceptual_physics", "test"],  # Validation = 26, Test = 235
                ["hails/mmlu_no_train", "college_mathematics", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "astronomy", "test"],  # Validation = 16, Test = 152
                ["hails/mmlu_no_train", "high_school_mathematics", "test"],  # Validation = 29, Test = 270
                # ["hails/mmlu_no_train", "college_chemistry", "test"],  # Validation = 8 (ignore), Test = 100
                ["hails/mmlu_no_train", "abstract_algebra", "test"],  # Validation = 11, Test = 100
                ["hails/mmlu_no_train", "high_school_chemistry", "test"],  # Validation = 22, Test = 203
                ["hails/mmlu_no_train", "college_biology", "test"],  # Validation = 16, Test = 144
                ["hails/mmlu_no_train", "high_school_biology", "test"],  # Validation = 32, Test = 310
            ],
        }
        self.few_shot_fp = os.path.join(project_dir, "tasks", self.task_name, "few_shot.json")
        assert os.path.isfile(self.few_shot_fp)
        with open(self.few_shot_fp, "r", encoding="utf-8") as fp_in:
            self.few_shot = json.load(fp_in)

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
        # few_shot_json = {}
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
                assert "validation" in cur_ds and "test" in cur_ds
                len_train, len_valid, len_test = 0, len(cur_ds["validation"]), len(cur_ds["test"])
                assert len_valid > 0 and len_test > 0

                ds_dict = {
                    "hf_dataset": hf_ds[0],
                    "hf_subset": hf_ds[1],
                    "eval_split": eval_split,
                    "eval_dataset": cur_ds[eval_split],
                }

                # random.seed(self.seed_few_shot)
                # few_shot_idx = random.sample(range(len_valid), self.max_num_few_shot)
                # few_shot_valid = cur_ds["validation"][few_shot_idx]

                self.logger.info(f">>> [dataset: {hf_ds[0]} --- {hf_ds[1]}] [eval_split = {eval_split}] "
                                 f"Train = {len_train}, Validation = {len_valid}, Test = {len_test}")

                dataset["data"].append(ds_dict)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f">>> Exception: {e}")

        # few_shot_json_fp = os.path.join(self.project_dir, "tasks", self.task_name, "few_shot.json")
        # with open(few_shot_json_fp, "w", encoding="utf-8") as fp_out:
        #     json.dump(few_shot_json, fp_out, ensure_ascii=False, indent=2)

        self.logger.info(f">>> [task_name: {self.task_name}] len(dataset) = {len(dataset['data'])}\n\n")  # 55
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
        assert isinstance(self.task_name, str) and self.task_name in self.all_tasks

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
        question = str(data_item["question"]).strip()
        choices = data_item["choices"]
        answer = int(data_item["answer"])

        assert isinstance(choices, list) and len(choices) == 4
        index2label = {0: "A", 1: "B", 2: "C", 3: "D"}
        assert answer in index2label

        answer_str = choices[answer]
        answer_label = index2label[answer]
        label_options = ["A", "B", "C", "D"]
        answer_options = choices

        # Set the main prompt (zero-shot)
        ord_A = ord("A")
        prompt_main = f"Question: {question}\n" + "\n".join(
            [f"({chr(_idx + ord_A)}) {_op}" for _idx, _op in enumerate(choices)]) + "\nAnswer:"

        # Set few-shot examples (with reasoning/rationale)
        prompt_few_shot = ""
        if num_few_shot > 0:
            for fs_example in few_shot_list:
                question_fs = str(fs_example["question"]).strip()
                choices_fs = fs_example["choices"]
                answer_fs = int(fs_example["answer"])

                assert isinstance(choices_fs, list) and len(choices_fs) == 4
                assert answer_fs in index2label

                answer_str_fs = choices_fs[answer_fs]
                answer_label_fs = index2label[answer_fs]

                cur_prompt_fs = f"Question: {question_fs}\n" + "\n".join(
                    [f"({chr(_idx + ord_A)}) {_op}" for _idx, _op in enumerate(choices_fs)]) + "\nAnswer:"
                if use_cot:
                    cot_reasoning = self.text_cleaning(str(fs_example["cot"]))
                    cur_prompt_fs += " Let's think step by step. " + cot_reasoning
                    cur_prompt_fs = (cur_prompt_fs.strip() +
                                     f" Therefore, the answer is: ({answer_label_fs}) {answer_str_fs}")
                elif use_arr:  # ARR Ablation Study: using ARR = 000, 001, 010, 100, 111
                    arr_reasoning = self.text_cleaning(str(fs_example["arr"]))
                    match arr_ablation:
                        case "000":  # Baseline: "Answer:"
                            cur_prompt_fs = cur_prompt_fs.strip() + f" ({answer_label_fs}) {answer_str_fs}"
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
                        case _:
                            raise ValueError(f"ValueError: Unexpected value for arr_ablation: {arr_ablation}")
                    if arr_ablation != "000":
                        cur_prompt_fs = (cur_prompt_fs.strip() +
                                         f" Therefore, the answer is: ({answer_label_fs}) {answer_str_fs}")
                else:
                    cur_prompt_fs = cur_prompt_fs.strip() + f" ({answer_label_fs}) {answer_str_fs}"
                prompt_few_shot += f"{cur_prompt_fs}\n\n"

        if use_cot:
            prompt_main += f" Let's think step by step."
        elif use_arr:  # ARR Ablation Study: using ARR = 000, 001, 010, 100, 111
            match arr_ablation:
                case "000":  # Baseline: "Answer:"
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
