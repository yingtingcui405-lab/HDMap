# 第一步：筛选有道路地图变化的场景

本目录提供给你运行的脚本、固定版本的官方标签、数据划分和说明。
目的：从现有 TbV + ArgoTweak 中建立变化场景索引，为之后的时空对齐、语义标注、多帧追踪和地图比较准备数据。
这一步是使用已有真值做数据集整理，不是训练或评估变化检测算法。

## 已核实的数据关系

| 本地数据 | 数量 | 意义 |
| --- | ---: | --- |
| `/Volumes/初棠/Argoverse 2/TbV_v1.0_shard0` 至 `shard20` | 1,043 个 log | 传感器、位姿、标定和地图 |
| `/Volumes/初棠/ArgoTweak/train+val` | 799 份 JSON | 对应全部 TbV train；官方再分为 697 train / 102 val |
| `/Volumes/初棠/ArgoTweak/test` | 111 份 JSON | 对应全部 TbV val；真实变化评测池 |
| TbV 原始 test | 133 个 log | 本地没有对应 ArgoTweak 标注，公开 TbV 标签也未覆盖，不能当作无变化 |

已比对 1,043 个目录名与官方划分，全部匹配。这只是目录与标签核查，不代表已经逐张检查图像、逐帧检查点云或验证下载完整性。

ArgoTweak train/val 的先验地图是人工构造的逼真变化；test 使用真实世界的地图不一致场景。两者应分别记录和评估。
TbV 原始 train 的地图与传感数据没有已标注变化，但同一段传感数据搭配 ArgoTweak 构造的先验地图后，会形成地图不一致样本。因此，标签属于“传感数据 + 所选地图”这一对，不能只绑定在传感序列上。

## 选择算法

1. 只遍历 shard 的下一层目录，建立 `log_id -> log_dir` 字典；不扫描百万级图像。
2. 使用官方划分，以完整 `log_id` 精确关联 TbV 和 ArgoTweak；不根据 shard 号、日期、城市或文件名猜测变化。
3. 读取 TbV 的 `tbv_val_split_annotations.json`：`events` 非空表示该 log 存在已标注变化事件。公开文件包含 105 个有事件场景、6 个空事件场景，共 213 个事件。
4. 读取 ArgoTweak test 的 `laneSegments`、`pedCrossings`、`drivableAreas`。元素的 `changes` 为 `[0]` 表示未标注变化，存在非零代码则纳入变化元素候选。保留所有代码，不只保留 baseline 使用的子集。
5. 对真实测试池取并集：`TbV events 非空 OR ArgoTweak 有非零 changes`。两者场景级判断不一致时仍保留，并额外列入 `review_logs.jsonl`。ArgoTweak v1 补充过原 TbV 漏标，不能强行要求交集。
6. 两种来源都标注无变化的真实测试场景进入对照集；人工构造数据进入独立训练池；133 个无公开标签场景进入 unknown 池。标签缺失/读取失败进入复查，不自动当负样本。
7. 保存清单和变化标签。原始数据保持整段 log，下一步再定位可见变化帧。

当前“变化”覆盖车道几何、标线、车道相关属性、人行横道等地图元素差异。不要只搜文件名中的 change/construction，也不要先用点云差分筛选：位姿误差、遮挡和车辆运动可能制造伪差异。

### 字段的两个易错点

- `new == null` 不能直接解释为现实中被删除；大量 `changes: [0]` 的未变元素也有 `new: null`。
- `old` 和 `new` 是数据结构字段，不能仅按英文名称解释为“过时地图”和“当前地图”。官方转换代码会根据变化代码组合这些字段，且为删除元素保留评估用的表示。本步骤完整保留原字段，不进行新增/删除方向转换。

使用最终的 `changes` 字段筛选；同时保留 `verified`、`modified`、`revised_change_history` 等原始信息作追溯，不把编辑历史或 modified 布尔值代替当前变化标签。
单靠这些标签也不能推出“变化会永久持续”。本步骤把 `permanence` 记录为 `not_inferred`。

## 你现在运行的命令

只需要 Python 3.9 或以上，无需 GPU、av2、pandas、PyTorch，也无需联网：

```bash
python3 "/Volumes/初棠/HDMap_Change_Prep/step01/select_logs.py"
```

默认输出到：

```text
/Volumes/初棠/HDMap_Change_Prep/step01/output_real_v1/
```

已有同名输出时，脚本会退出，避免覆盖；再次运行可指定新目录：

```bash
python3 "/Volumes/初棠/HDMap_Change_Prep/step01/select_logs.py" \
  --out "/Volumes/初棠/HDMap_Change_Prep/step01/output_real_v2"
```

默认读取 111 份真实测试标注，给其他场景建立索引。外接盘首次读取可能较慢，脚本每经过 50 个 log 输出进度。
如以后需要详细解析另外 799 份人工构造标注，再运行以下可选命令；这不是当前第一轮的必要步骤：

```bash
python3 "/Volumes/初棠/HDMap_Change_Prep/step01/select_logs.py" \
  --scan-constructed \
  --out "/Volumes/初棠/HDMap_Change_Prep/step01/output_all_v1"
```

## 输出怎么用

| 文件 | 用途 |
| --- | --- |
| `report.json` | 首先检查这个：场景数、缺失文件、复查数量、官方来源及版本 |
| `real_change_logs.jsonl` | 下一步处理的真实变化候选，每行一个完整 log |
| `real_no_change_controls.jsonl` | 真实测试池中的无变化对照，用于检查误报 |
| `constructed_pool.jsonl` | ArgoTweak 人工构造 train/val 的索引，保持官方划分 |
| `unlabeled_logs.jsonl` | 原始 TbV test，缺少公开变化真值，不参与首轮有监督处理 |
| `review_logs.jsonl` | 标签不一致、缺少路径、解析失败或未知划分；可能与上述文件重叠 |
| `tbv_events.jsonl` | 每行一个 TbV 原始变化事件，保留空间区域及变化类型 |
| `argotweak_changed_elements.jsonl` | 每行一个变化元素，保留完整 old/new/changes 等字段 |
| `all_logs.jsonl` | 所有本地 log 的完整索引 |

`constructed_pool.jsonl` 中的 `argotweak_scanned: false` 和计数 `null` 表示本次没读取该标注，绝不是零变化。
`layout_check: paths_only_not_sensor_integrity` 表示只检查地图/位姿/标定文件及传感器目录存在，未检查目录非空、传感器文件数量或内容损坏。

清单保留 `log_dir` 和 `argotweak_path`，直接通过路径读取原始数据即可。无需重新复制近 TB 级数据，也不必先把 21 个 shard 合并。
如未来要使用要求 `root/log_id` 结构的官方 loader，可以再单独建立软链接索引；当前脚本已支持你的 shard 布局。

## 第一步完成的判据

- `local_log_count = 1043`，`local_argotweak_file_count = 910`。
- `missing_official_log_ids` 和 `missing_argotweak_log_ids` 为空。
- 核查 `real_change`、`real_no_change_control`、`review_logs` 数量，不把有数据问题的场景直接交给队友。
- 对选中场景抽查图像与变化标签。当前的场景级筛选不等于全部帧都有变化，也不等于这些变化一定在相机视野内。

原 TbV 标签的计数基准：105 个变化 log，6 个无变化 log，213 个事件。其中 147 个事件属于 `lane_geometry_change`，另外 66 个是横道事件（20 insert_crosswalk、46 delete_crosswalk）。这是事件数，不是不同场景数，也不是变化帧数。
2026-10-23 对本地 ArgoTweak test 的只读核查：同样是 105 个有非零 changes 的场景、6 个无变化场景，场景级二元标签与 TbV 全部一致。其中 90 个场景涉及车道元素、38 个涉及横道，两者有重叠；共 2,602 个变化车道元素和 79 个变化横道元素。元素数与 TbV 的事件数不应直接比较，也不表示这些元素每一帧都可见。
若 ArgoTweak 版本变化，二者并集结果可能不同，以本次报告为准。

## 下一步的边界：空间定位后再选时间片段

完成清单后，先选少量场景跑通时空对齐。以车辆在 city 坐标系的位置和事件几何计算距离：

```text
候选帧(t) = 自车位置到至少一个变化区域/线段的最小距离 < R
```

可以先选 `R = 30 m` 作为工程起点，再结合所用相机视野、遮挡和标签范围调整；这是你们的处理参数，不是本脚本已完成的可见性判断。计算的是到区域/线段的距离，不是只到几何中心或顶点。
把连续候选帧组成时间片段，再按需求向前后扩展约 2–3 秒，供多帧追踪使用。扩展帧是上下文，不自动标成变化阳性；需要严格在线评估时，不能用未来帧辅助当前时刻决策。

后续训练与测试保持官方划分，不把同一 log 的帧随机拆到训练/测试。如果你们重新做研究划分，应按重复地点/路段分组，防止同一变化位置同时进入训练和测试。
变化真值用于筛选、监督和评测，不能作为实际变化检测/置信度模型的输入，否则会泄漏答案。
TbV 没有目标物体跟踪标注；以后做动态目标轨迹需要另行检测与追踪，不能把自车位姿当成其他车辆轨迹。

## 可复现性和验证

官方标签固定在 `metadata/sources.json` 记录的 Git commit，记录了 SHA-256。下载的 Python 参考文件不被执行；划分常量通过 AST 安全提取。

小型夹具测试命令（不处理原始数据）：

```bash
cd "/Volumes/初棠/HDMap_Change_Prep/step01"
python3 -B -m unittest -v test_select_logs
```

交付时上述 5 项测试已通过，且用真实的一个变化 JSON 和一个无变化 JSON 检查了解析函数。正式 `select_logs.py` 尚未在完整数据上运行，`output_real_v1` 留给你运行生成。

## 官方依据

- [TbV 数据格式与无物体标注说明](https://argoverse.github.io/user-guide/datasets/map_change_detection.html)
- [TbV 官方仓库 FAQ：划分和训练集无变化说明](https://github.com/johnwlambert/tbv#faq)
- [TbV 公开变化事件标签](https://github.com/johnwlambert/tbv/blob/main/labeled_data/tbv_val_split_annotations.json)
- [ArgoTweak 论文，第 4.2 节：构造训练集与真实测试集](https://arxiv.org/html/2509.08764v1#S4.SS2)
- [ArgoTweak v1 数据卡：补充漏标及修正说明](https://huggingface.co/datasets/lwild/ArgoTweak)
- [ArgoTweak 官方数据划分](https://github.com/KTH-RPL/ArgoTweak_baselines/blob/main/tools/splits.json)
- [ArgoTweak 官方字段解析代码](https://github.com/KTH-RPL/ArgoTweak_baselines/blob/main/tools/utils/annotate.py)

原始 TbV 和 ArgoTweak 数据沿用官方 CC BY-NC-SA 4.0 许可；本目录中的官方文件保留来源，不构成新的数据授权。
