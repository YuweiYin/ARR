# ARR: QA with LLMs via Analyzing, Retrieving, and Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2502.04689-b31b1b.svg)](https://arxiv.org/abs/2502.04689)

<details open><summary>Summary</summary>

<img src="https://yuweiyin.com/files/img/2025-02-15-ARR.jpg" alt="ARR" width="50%" />

* **ARR**: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning
* **Paper** (arXiv): https://arxiv.org/abs/2502.04689

```text
Large language models (LLMs) achieve remarkable performance on challenging benchmarks 
that are often structured as multiple-choice question-answering (QA) tasks. Zero-shot 
Chain-of-Thought (CoT) prompting enhances reasoning in LLMs but provides only vague and 
generic guidance ("think step by step"). This paper introduces ARR, an intuitive and 
effective zero-shot prompting method that explicitly incorporates three key steps in QA 
solving: analyzing the intent of the question, retrieving relevant information, and 
reasoning step by step. Comprehensive experiments across diverse and challenging QA tasks 
demonstrate that ARR consistently improves the Baseline (without ARR prompting) and 
outperforms CoT. Ablation and case studies further validate the positive contributions of 
each component: analyzing, retrieving, and reasoning. Notably, intent analysis plays 
a vital role in ARR. Additionally, extensive evaluations across various model sizes, 
LLM series, and generation settings solidify the effectiveness, robustness, and 
generalizability of ARR.
```

</details>

## Code coming soon...

## License

Please refer to the [LICENSE](./LICENSE) file for more details.

## Citation

* **Paper** (arXiv): https://arxiv.org/abs/2502.04689
* If you find our work helpful, please kindly cite us using this BibTeX:

```bibtex
@article{yin2025arr,
  title   = {ARR: Question Answering with Large Language Models via Analyzing, Retrieving, and Reasoning},
  author  = {Yin, Yuwei and Carenini, Giuseppe},
  journal = {arXiv preprint arXiv:2502.04689},
  year    = {2025},
  url     = {https://arxiv.org/abs/2502.04689},
}
```

---
