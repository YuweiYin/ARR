# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import json
import random
from typing import Optional, Dict, Any

from tasks import EvalTaskManager


class EvalTaskSciq(EvalTaskManager):

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

        # Train = 11679, Validation = 1000, Test = 1000
        # Features: ["question", "distractor3", "distractor1", "distractor2", "correct_answer", "support"]
        # Eval: test set; Few-shot: valid set

        self.task_name = "sciq"
        self.task_info = {
            "has_passage": True,
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["sciq", None, "test"],
            ],
        }
        self.few_shot_fp = os.path.join(project_dir, "tasks", self.task_name, "few_shot.json")
        assert os.path.isfile(self.few_shot_fp)
        with open(self.few_shot_fp, "r", encoding="utf-8") as fp_in:
            self.few_shot = json.load(fp_in)

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
        passage, question = str(data_item["support"]).strip(), str(data_item["question"]).strip()
        correct_answer = str(data_item["correct_answer"]).strip()
        distractor1 = str(data_item["distractor1"]).strip()
        distractor2 = str(data_item["distractor2"]).strip()
        distractor3 = str(data_item["distractor3"]).strip()

        shuffle_seed = int(hash(correct_answer)) % int(1e8)
        random.seed(shuffle_seed)  # To shuffle the options
        options = [correct_answer, distractor1, distractor2, distractor3]
        random.shuffle(options)

        index2label = {0: "A", 1: "B", 2: "C", 3: "D"}
        answer_label = "A"
        for o_idx, option in enumerate(options):
            if option == correct_answer:
                answer_label = index2label[o_idx]
                break
        answer_str = correct_answer
        label_options = ["A", "B", "C", "D"]
        answer_options = options

        # Set the main prompt (zero-shot)
        ord_A = ord("A")
        prompt_main = f"Passage: {passage}\nQuestion: {question}\n" + "\n".join(
            [f"({chr(_idx + ord_A)}) {_op}" for _idx, _op in enumerate(options)]) + "\nAnswer:"

        # Set few-shot examples (with reasoning/rationale)
        prompt_few_shot = ""
        if num_few_shot > 0:
            for fs_example in few_shot_list:
                passage_fs, question_fs = str(fs_example["support"]).strip(), str(fs_example["question"]).strip()
                correct_answer_fs = str(fs_example["correct_answer"]).strip()
                distractor1_fs = str(fs_example["distractor1"]).strip()
                distractor2_fs = str(fs_example["distractor2"]).strip()
                distractor3_fs = str(fs_example["distractor3"]).strip()

                shuffle_seed_fs = int(hash(correct_answer_fs)) % int(1e8)
                random.seed(shuffle_seed_fs)  # To shuffle the options
                options_fs = [correct_answer_fs, distractor1_fs, distractor2_fs, distractor3_fs]
                random.shuffle(options_fs)

                answer_label_fs = "A"
                for o_idx, option in enumerate(options_fs):
                    if option == correct_answer_fs:
                        answer_label_fs = index2label[o_idx]
                        break
                answer_str_fs = correct_answer_fs

                cur_prompt_fs = f"Passage: {passage_fs}\nQuestion: {question_fs}\n" + "\n".join(
                    [f"({chr(_idx + ord_A)}) {_op}" for _idx, _op in enumerate(options_fs)]) + "\nAnswer:"
                if use_cot:
                    cot_reasoning = self.text_cleaning(str(fs_example["cot"]))
                    cur_prompt_fs += " Let's think step by step. " + cot_reasoning
                    cur_prompt_fs = (cur_prompt_fs.strip() +
                                     f" Therefore, the answer is: ({answer_label_fs}) {answer_str_fs}")
                elif use_arr:  # ARR Ablation Study: using ARR = 000, 001, 010, 100, 111
                    arr_reasoning = self.text_cleaning(str(fs_example["arr"]))
                    match arr_ablation:
                        case "000":
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
