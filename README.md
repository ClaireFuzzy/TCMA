# DVTMD & TCMA

## Text-Conditioned Multigranularity Alignment for Drone Cross-Modal Text–Video Retrieval

**Zixu Zhao, Yang Zhan, Yunhao Li, and Yan Li**
*IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), 2026*

[Paper / 论文](https://ieeexplore.ieee.org/document/11724583/) · [DOI](https://doi.org/10.1109/JSTARS.2026.3741527) · [Preprint / 预印本](https://arxiv.org/abs/2510.10180)

This paper introduces both **DVTMD (Drone Video-Text Match Dataset)**, a dataset with fine-grained captions built on ERA videos, and **TCMA (Text-Conditioned Multigranularity Alignment)**, a method for bidirectional drone text–video retrieval. This repository provides the DVTMD annotations and TCMA code released so far.

本文同时提出 **DVTMD（Drone Video-Text Match Dataset，无人机视频文本匹配数据集）** 和 **TCMA（Text-Conditioned Multigranularity Alignment，文本条件多粒度对齐方法）**。DVTMD 基于 ERA 视频构建细粒度文本标注，TCMA 用于无人机视频与文本之间的双向检索。本仓库提供目前已整理发布的 DVTMD 标注与 TCMA 代码。

> **Status: under update. The environment setup and execution workflow are still being prepared. The current release is not yet runnable.**
>
> **当前状态：更新中。运行环境与执行流程仍在整理，当前版本暂不能直接运行。**

## Overview / 工作概述

**DVTMD** provides fine-grained, semantically diverse descriptions of drone videos for text–video retrieval. Users need both the DVTMD annotations provided here and the original videos downloaded separately from the ERA dataset.

**DVTMD** 为无人机视频提供细粒度、语义多样的描述，用于视频文本检索。使用时需要将本仓库提供的 DVTMD 标注与另行下载的 ERA 原始视频配合使用。

TCMA combines video–sentence matching, sentence-guided frame aggregation, and word-guided patch alignment. It selects informative words and patches and predicts text-dependent attention temperatures to focus on relevant visual content. See the [paper](https://ieeexplore.ieee.org/abstract/document/11724583/) for the method.

TCMA 在视频、帧和图像块三个粒度上进行文本引导的对齐，通过词语与图像块选择模块筛选有效信息，并利用文本自适应温度机制调整注意力分布。方法详情见[论文](https://ieeexplore.ieee.org/abstract/document/11724583/)。

| Component / 模块 | Implementation / 实现 |
| --- | --- |
| CLIP encoders / 文本与视觉编码器 | `modules/module_clip.py` |
| Multigranularity alignment / 多粒度对齐 | `modules/modeling_fineclip_patches_cdcl.py` |
| Word and patch selection / 词语与图像块选择 | `modules/differential_topk.py` |
| Dynamic temperature / 动态温度 | `modules/module_DTIB.py` |
| Training and retrieval evaluation / 训练与检索评估 | `main_eercf.py` |

## Repository Structure / 目录结构

```text
TCMA/
├── README.md
├── main_eercf.py                     # Training and evaluation entry point
├── train_mydata_32_32.sh             # Original experiment script
├── metrics.py                       # Retrieval metrics
├── util.py                          # Logging and utilities
├── modules/                         # Model, tokenizer, and optimizer
│   ├── modeling_fineclip_patches_cdcl.py
│   ├── differential_topk.py
│   ├── module_DTIB.py
│   ├── module_clip.py
│   ├── bpe_simple_vocab_16e6.txt.gz
│   └── cross-base/cross_config.json
├── dataloaders/                      # Dataset loaders and frame/video reading
├── data/tiankaibin/Text2VideoDataset/
│   └── pos.json                     # Tokenizer resource
└── DVTMD/DVTMD/
    ├── mydata_train.csv
    ├── mydata_data.json
    └── mydata_test_single.csv
```

## Environment / 环境配置

**Under update — not yet runnable.** The environment configuration, dependency versions, and execution workflow are still being organized. Installation and reproduction instructions will be added after verification. The current repository is available for browsing the code and preparing the dataset, but cannot yet be run directly for training or evaluation.

**更新中，当前暂不能运行。** 环境配置、依赖版本与执行流程仍在整理，验证完成后将补充安装和复现说明。目前可先查阅代码并准备数据，暂不能直接使用当前仓库进行训练或评估。

## Dataset / 数据集

### DVTMD and available annotations / DVTMD 与当前标注

The DVTMD Dataset describes 2,864 videos and 14,320 captions, with five captions per video. The files currently included here contain the following:

DVTMD 共包含 2,864 个视频和 14,320 条描述，每个视频对应 5 条描述。当前仓库实际包含的标注如下：

| File / 文件 | Videos / 视频数 | Contents / 内容 |
| --- | ---: | --- |
| `mydata_train.csv` | 1,473 | Training video IDs / 训练视频 ID |
| `mydata_data.json` | 1,473 | 7,365 training captions, five per video / 每个训练视频 5 条，共 7,365 条描述 |
| `mydata_test_single.csv` | 1,391 | One caption per test video / 每个测试视频 1 条描述 |

**Release scope:** raw videos, extracted frames, trained TCMA checkpoints, and the full five-caption-per-video test annotations are not included in this checkout. The single-caption test file is not equivalent to the complete test annotations described in the paper. CapERA annotations are also not bundled here.

**当前文件范围：** 本仓库未包含原始视频、抽帧图像、训练好的 TCMA 权重及每个测试视频 5 条描述的完整测试标注。现有单描述测试文件与论文中的完整测试标注不同；仓库也未附带 CapERA 标注。

### Download ERA videos / 下载 ERA 视频

**Users must download the original ERA dataset videos separately.** DVTMD adds text annotations to ERA videos; the video files are not redistributed in this repository. Visit the [official ERA dataset page](https://lcmou.github.io/ERA_Dataset/) and use **Download → Dataset → Download Link** to obtain the videos.

**用户需要自行下载 ERA 数据集的原始视频。** DVTMD 基于 ERA 视频构建文本标注，本仓库不重复提供视频文件。请访问 [ERA 数据集官方页面](https://lcmou.github.io/ERA_Dataset/)，通过 **Download → Dataset → Download Link** 下载视频。

After downloading and extracting the videos, match their IDs to the `video_id` fields in the DVTMD annotations. Keep the original video IDs. Preparing these data does not make the current code release runnable; environment and execution instructions are still being updated.

下载并解压后，将视频 ID 与 DVTMD 标注中的 `video_id` 字段对应，并保留原始视频 ID。当前可以先准备数据；运行环境与执行说明仍在更新，完成下载后也暂不能直接运行本版本。

### Annotation format / 标注格式

Training CSV / 训练集 CSV：

```csv
video_id
Tra_Baseball_001
Tra_Baseball_002
```

Training JSON, minimal format example / 训练 JSON 最小格式示例：

```json
{
  "videos": [
    {"video_id": "Tra_Baseball_001", "name": "Tra_Baseball_001", "split": "train", "id": 0}
  ],
  "sentences": [
    {"video_id": "Tra_Baseball_001", "caption": "Aerial view of a baseball stadium.", "sen_id": 0}
  ]
}
```

Evaluation CSV requires `video_id` and `sentence`. The bundled file also has `key` and `vid_key` columns, which this loader does not require.

评估 CSV 需要 `video_id` 和 `sentence` 两列。仓库中的文件还包含 `key`、`vid_key` 两列，但当前加载器不依赖这两列。

```csv
video_id,sentence
Test_Baseball_001,Aerial view of a baseball field.
```

The examples above illustrate the schema only. Keep the supplied video IDs unchanged when preparing visual inputs.

以上片段仅演示格式。准备视频或帧数据时，应保留标注文件中的原始视频 ID。

### Visual inputs / 视频与帧数据

Set `--features_path` to the directory containing either video files or per-video frame folders. Despite its name, this argument points to raw visual inputs, not precomputed feature tensors. The loader searches for `<video_id>.mp4`, then `<video_id>.avi`, then `<video_id>/`.

将 `--features_path` 设为视频文件或逐视频帧目录的上级目录。该参数虽然名为 features，实际读取的是视频或图像，而非预提取特征。加载器依次查找 `<video_id>.mp4`、`<video_id>.avi` 和 `<video_id>/`。

```text
/path/to/ERA_Frames/
├── Tra_Baseball_001/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
├── Tra_Baseball_002/
└── Test_Baseball_001/
```

Frame names must end in an underscore followed by an integer and `.jpg`, such as `frame_000001.jpg`, because `rawframe_util.py` sorts frames by this numeric suffix. Do not place unrelated files inside frame folders. `--slice_framepos 2` uniformly subsamples decoded frames when there are more than `--max_frames`.

帧文件名须采用 `frame_000001.jpg` 等“下划线 + 整数 + `.jpg`”结尾的形式，读取器据此排序。帧目录中不要混入其他文件。读取的帧数超过 `--max_frames` 时，`--slice_framepos 2` 会进行均匀采样。

## Training and Evaluation / 训练与评估

**Instructions are being updated.** Verified training and evaluation commands, pretrained-weight preparation, and checkpoint usage will be documented once the environment and execution workflow are ready. The existing `train_mydata_32_32.sh` is an experiment script with machine-specific paths, not a ready-to-run entry point for this release.

**使用说明更新中。** 环境与执行流程整理完成后，将补充经验证的训练、评估命令，以及预训练权重准备和模型权重使用说明。现有 `train_mydata_32_32.sh` 是包含特定机器路径的实验脚本，当前版本不能直接执行该脚本完成训练。

## Reported Results / 论文报告结果

The [published abstract](https://ieeexplore.ieee.org/abstract/document/11724583/) reports the following R@1 scores. These are paper results, not measurements from running this checkout.

[正式发表的论文摘要](https://ieeexplore.ieee.org/abstract/document/11724583/)报告了以下 R@1 指标。这些数值引用自论文，并非在当前仓库上重新运行得到的结果。

| Direction / 检索方向 | R@1 (%) |
| --- | ---: |
| Text-to-video / 文本检索视频 | 45.5 |
| Video-to-text / 视频检索文本 | 42.8 |

Refer to the paper for dataset-specific comparisons and backbone configurations. The bundled single-caption test file alone is insufficient to establish reproduction of the full benchmark.

各数据集和主干配置的完整对比请参阅论文。仅使用仓库当前的单描述测试文件，不能据此认定已复现完整基准结果。

## Citation / 引用

If you use TCMA or DVTMD in your research, please cite the paper. / 如果在研究中使用 TCMA 或 DVTMD，请引用本文。

```bibtex
@article{zhao2026tcma,
  author  = {Zhao, Zixu and Zhan, Yang},
  title   = {{TCMA}: Text-Conditioned Multigranularity Alignment for Drone Cross-Modal Text--Video Retrieval},
  journal = {IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year    = {2026},
  volume  = {19},
  pages   = {34849--34863},
  doi     = {10.1109/JSTARS.2026.3741527}
}
```

## Acknowledgements / 致谢

We thank [ERA](https://lcmou.github.io/ERA_Dataset/) for the source videos used to construct DVTMD. The TCMA implementation builds on the CLIP video-retrieval code lineage; we acknowledge [CLIP](https://github.com/openai/CLIP), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip), and [EERCF](https://arxiv.org/abs/2401.00701).

感谢 ERA 提供用于构建 DVTMD 的视频资源。TCMA 实现基于 CLIP 视频检索相关工作与代码，感谢 CLIP、CLIP4Clip 和 EERCF 的贡献。
