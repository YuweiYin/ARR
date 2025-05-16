# -*- coding: utf-8 -*-
"""
__author__ = "@YuweiYin"
"""

import os
import json
from typing import Optional, Dict, Any

from tasks import EvalTaskManager


class EvalTaskBoolq(EvalTaskManager):

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

        # Train = 9427, Validation = 3270, Test = 3245
        # Features: ["question", "passage", "idx", "label"]
        # Eval: valid set; Few-shot: train set

        self.task_name = "boolq"
        self.task_info = {
            "has_passage": True,
            "hf_dataset": [  # [hf_id, subset, eval_set]
                ["super_glue", "boolq", "validation"],
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
        passage, question = str(data_item["passage"]).strip(), str(data_item["question"]).strip()
        idx, label = int(data_item["idx"]), int(data_item["label"])

        assert label == 0 or label == 1
        answer_str = "No" if label == 0 else "Yes"
        answer_label = label
        label_options = [0, 1]
        answer_options = ["No", "Yes"]
        if not question.endswith("?"):
            question += "?"

        # Set the main prompt (zero-shot)
        prompt_main = f"""
Passage: {passage}\nQuestion: {question}\nAnswer:
        """.strip()

        # Set few-shot examples (with reasoning/rationale)
        prompt_few_shot = ""
        if num_few_shot > 0:
            for fs_example in few_shot_list:
                passage_fs, question_fs = str(fs_example["passage"]).strip(), str(fs_example["question"]).strip()
                idx_fs, label_fs = int(fs_example["idx"]), int(fs_example["label"])

                assert label_fs == 0 or label_fs == 1
                answer_str_fs = "No" if label_fs == 0 else "Yes"
                if not question_fs.endswith("?"):
                    question_fs += "?"

                cur_prompt_fs = f"""
Passage: {passage_fs}\nQuestion: {question_fs}\nAnswer:
                """.strip()
                if use_cot:
                    cot_reasoning = self.text_cleaning(str(fs_example["cot"]))
                    cur_prompt_fs += " Let's think step by step. " + cot_reasoning
                    cur_prompt_fs = cur_prompt_fs.strip() + f" Therefore, the answer is: {answer_str_fs}"
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
