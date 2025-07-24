
# OpenWebText 数据集准备

本目录包含用于准备 OpenWebText 数据集的脚本。

## 脚本说明

### `prepare.py` (原始版本)
- nanoGPT 项目的原始数据准备脚本
- 简单直接，无断点续传功能
- 适合一次性完整运行

### `prepare_robust.py` (推荐使用)
- 增强版本，支持断点续传和错误恢复
- 优化的缓存利用和进度跟踪
- 自动重试机制和优雅的信号处理
- 更好的用户体验和错误处理

### `cleanup_cache.py` (清理工具)
- 清理各种缓存和临时文件
- 交互式界面，可选择性清理
- 显示磁盘使用情况

## 使用建议

**首次使用或网络不稳定环境**：
```bash
python prepare_robust.py
```

**网络稳定且想要最简单的方式**：
```bash
python prepare.py
```

**清理缓存文件**：
```bash
python cleanup_cache.py
```

## 主要改进

`prepare_robust.py` 相比原始版本的改进：

1. **断点续传**：支持从任何步骤恢复
2. **缓存优化**：智能利用已有的tokenization缓存
3. **网络重试**：自动重试失败的网络操作
4. **进度跟踪**：详细的进度信息和文件大小显示
5. **错误处理**：更好的错误恢复和状态保存
6. **信号处理**：支持Ctrl+C优雅退出

## 数据集信息

处理完成后会生成：
- `train.bin`: 训练数据 (~17GB, ~9B tokens)
- `val.bin`: 验证数据 (~8.5MB, ~4M tokens)

这些文件与原始脚本生成的文件完全兼容。

## 原始信息

after running `prepare.py` (preprocess) we get:

- train.bin is ~17GB, val.bin ~8.5MB
- train has ~9B tokens (9,035,582,198)
- val has ~4M tokens (4,434,897)

this came from 8,013,769 documents in total.

references:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
