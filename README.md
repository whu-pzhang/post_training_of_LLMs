# Post-Training of LLMs

本项目是 [DeepLearning.AI](https://www.deeplearning.ai/short-courses/post-training-of-llms/) 提供的《Post-Training of LLMs》课程的实践代码。

## 项目简介

本项目旨在探索如何对大语言模型（LLMs）进行后训练（Post-Training），包括微调（Fine-Tuning），监督训练（Supervised Fine-Tuning, SFT），直接偏好优化（Direct Preferance Optimization, DPO）和 在线强化学习（Online Reforce Learning, Online RL）。通过实践，您将学习如何加载模型、处理数据集、训练模型以及生成响应。

## 项目结构

```
├── helper.py                # 辅助函数，包括生成响应、测试模型等
├── sft.py                   # SFT训练脚本
├── models/                  # 存储训练后的模型及相关配置
│   └── SmolLM2-135M-SFT/    # 示例模型文件夹
│       ├── chat_template.jinja
│       ├── config.json
│       ├── generation_config.json
│       ├── merges.txt
│       ├── model.safetensors
│       ├── special_tokens_map.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── training_args.bin
│       └── vocab.json
├── pyproject.toml           # 项目依赖及配置
├── uv.lock                  # 依赖锁定文件
├── README.md                # 项目说明文件
```

## 环境配置

1. 使用 `uv` 安装项目依赖：
   ```bash
   uv sync
   ```
2. 如果需要更新依赖，请运行：
   ```bash
   uv add <package_name>
   ```

## 使用说明

### 1. 加载模型并生成响应

运行 `helper.py` 文件，加载模型并生成示例问题的响应：
```bash
python helper.py
```

### 2. 微调模型

运行 `sft.py` 文件，对模型进行监督微调：
```bash
python sft.py
```

### 3. 查看数据集

`helper.py` 提供了 `display_dataset` 函数，可以用于可视化数据集内容。

## 参考链接

- [DeepLearning.AI - Post-Training of LLMs](https://www.deelearning.ai/short-courses/post-training-of-llms/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [TRL库](https://github.com/huggingface/trl)

## 许可证

本项目遵循 MIT 许可证。



