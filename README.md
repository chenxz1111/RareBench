# RareBench Can LLMs Serve as Rare Diseases Specialists?

<p align="center">
    🤗 <a href="https://huggingface.co/datasets/chenxz/RareBench" target="_blank">HF Repo</a> • 📃 <a href="https://arxiv.org/abs/2402.06341" target="_blank">Paper</a>
</p>

**RareBench** is a pioneering benchmark designed to systematically evaluate the capabilities of LLMs on 4 critical dimensions within the realm of rare diseases.
Meanwhile, we have compiled the largest open-source dataset on rare disease patients, establishing a benchmark for future studies in this domain. To facilitate differential diagnosis of rare diseases, we develop a dynamic few-shot prompt methodology, leveraging a comprehensive rare disease knowledge graph synthesized from multiple knowledge bases, significantly enhancing LLMs’ diagnos-
tic performance. Moreover, we present an exhaustive comparative study of GPT-4’s diagnostic capabilities against those of specialist physicians. Our experimental findings underscore the promising potential of integrating LLMs into the clinical diagnostic process for rare diseases. 

## ⚙️ How to evaluate on LongBench

#### Load Data

```python
from datasets import load_dataset

datasets = ["RAMEDIS", "MME", "HMS", "LIRICAL", "PUMCH_ADM"]

for dataset in datasets:
    data = load_dataset('chenxz/RareBench', dataset, split='test')
    print(data)
```

#### API-based LLMs

```
Put your own Openai key in the llm_utils/gpt_key.txt file.
Put your own Gemini key in the llm_utils/gemini_key.txt file.
Put your own Zhipuai key in the llm_utils/glm_key.txt file.
```

#### Local LLMs

```
Replace the content in the mapping/local_llm_path.json file with the path to the LLM on your local machine.
```

## 📄 Acknowledgement

- Some of the tasks of LongBench are based on the datasets proposed by previous researchers, including [HotpotQA](https://hotpotqa.github.io/),.

## 📝 Citation
```
@article{chen2024rarebench,
  title={RareBench: Can LLMs Serve as Rare Diseases Specialists?},
  author={Chen, Xuanzhong and Mao, Xiaohao and Guo, Qihan and Wang, Lun and Zhang, Shuyang and Chen, Ting},
  journal={arXiv preprint arXiv:2402.06341},
  year={2024}
}
```