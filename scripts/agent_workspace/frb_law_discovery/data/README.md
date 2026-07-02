# 数据说明

本数据来自清华大学天文系李菂老师团队，是其团队工作 Frequency-dependent polarization of repeating fast radio bursts-implications for their origin (arXiv:2202.09601) 所涉及的可复现数据。值得注意的是，此数据并非完整原始数据，仅保留了与 31 个本地 burst 点直接相关的原始 HDF5 文件。

## 覆盖对象

共 31 个 burst：

- FRB20190303A / FAST：3 个
- FRB20190417A / FAST：5 个
- FRB20190520B / GBT：3 个
- FRB20201124A / FAST：11 个
- FRB20201124A / GBT：9 个

## 文件含义

- `raw/*.h5`：calibrated full-Stokes HDF5 数据，包含 Stokes `I,Q,U,V` 和 frequency array。数组维度约定为 `[frequency, time]`。
    * `I`：总强度。
    * `Q`：线偏振的一个分量，描述相互正交线偏振方向的强度差。
    * `U`：线偏振的另一个分量，与 `Q` 一起决定线偏振强度和偏振角。
    * `V`：圆偏振分量，正负号表示相反的圆偏振手性约定。
- `selection_windows.csv`：人工选择的分析窗口。每行对应一个 burst，记录 noise time window、burst time window 和 frequency window，是当前预处理的主要人工输入。
- `paper_table_s3_local_reference.csv`：上述 31 个本地 burst 在论文 Table S3 中的参考值，已映射到 HDF5 group name，仅供参考。
- `manifest_sha256.csv`：输入文件的大小和 SHA256，用于检查数据 provenance。

## 索引约定

- `selection_windows.csv` 中的窗口端点都是闭区间。读取时应使用 `I[freq_start_idx:freq_end_idx + 1, burst_start_idx:burst_end_idx + 1]`
- noise window 也使用闭区间
- frequency 是 axis 0，time 是 axis 1

## 原始计算思路

李菂老师团队推荐的处理流程是：

1. 从 `raw/*.h5` 读取 Stokes `I, Q, U, V` 和频率轴。
2. 在 noise window 上逐频率通道扣除 off-pulse mean。
3. 在 frequency window 上对 `I, Q, U, V` 做频率平均。
4. 计算 `L = sqrt(Q^2 + U^2)`，并按阈值 `2L / sigma_I > pi` 做 debias。
5. 在线偏振度中使用 `sum(L_debias) / sum(I)`。
6. 默认用 Stokes-I S/N 权重计算 center frequency，同时保留 I-weighted 和 frequency-midpoint 诊断量。

## 使用注意

- 本目录数据为只读输入；不要修改 `raw/`、`selection_windows.csv` 或参考表。
- 如需探索新的特征，请修改上层目录的 `feature_extraction.py`，从原始数据中生成新的 `saved/<exp_name>/output.csv`。
- 窗口选择本身是 provenance 的一部分。低 S/N 或窗口敏感 burst 可能导致与 Table S3 有差异，可以视情况改进窗口。
