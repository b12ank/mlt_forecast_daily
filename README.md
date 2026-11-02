# 六天实时电价日均预测开发框架

在北京时间 D 日中午前，预测 D+2～D+7 六天的实时电价日均价。默认单次命令仍使用最近已知价参考模型；现包含四阶段实验、价格日历特征、Ridge、CatBoost、LightGBM、共享/独立模型及用户天气宽表接口。自动化测试与预检查不代表已完成模型选择或精度优化；正式实验由用户运行。

**四阶段实验请先阅读 [BATCH_RUNS.md](BATCH_RUNS.md)**；天气接口规范见 [WEATHER_DATA.md](WEATHER_DATA.md)。

新增 TimesFM 2.5 纯历史电价基线：默认使用截至 D−1 的连续 512 天日价，预测 D～D+7 后输出 D+2～D+7。通过 `--config configs/experiments/timesfm.yaml` 接入现有回测和预测，独立安装与权重配置见 [TIMESFM.md](TIMESFM.md)。四阶段实验清单保持不变。

stage2 支持从 Ridge、CatBoost、LightGBM 中人工选择两个或三个不同模型。当前 `configs/selections.yaml` 已选择全部三个，每个分别比较 F0/F1/F2，共九个串行实验。

stage3 默认对 LightGBM、CatBoost 比较 F3（天气＋缺失标记，520列）、F4（天气，260列）、F5（F0＋天气且无天气缺失标记，302列），共六个实验，不重跑F2。原stage3参数搜索已迁移至stage4，可选择F0～F5与共享/独立结构。候选配置分别使用 `stage3_models` 和 `stage4_combinations`。

业务约束以根目录 `AGENTS.md` 与 `DATA_AVAILABILITY.md` 为准。当前数据快照的取数起点是 **2026-10-25**，不能由文件末行日期推断预测起点。

## 目录与职责

```text
develop/
├── README.md
├── environment.yml                 # Conda 环境：long_terem_epf
├── .gitignore
├── run.py                          # 唯一命令入口
├── configs/
│   ├── default.yaml                # 公共数据、任务、评价协议
│   └── experiments/reference.yaml  # 实验特征与模型选择
├── forecast/
│   ├── __init__.py
│   ├── data.py                     # 只读加载及标签一致性检查
│   ├── availability.py             # 按字段构建合法信息集
│   ├── pipeline.py                 # 配置、训练、回测、结果保存
│   ├── batch.py                    # 四阶段串行执行、续跑、汇总
│   ├── weather_data.py             # 用户历史/预报日度宽表与来源区分
│   ├── evaluation.py               # 固定评价与独立 HTML 报告
│   ├── features/
│   │   ├── __init__.py
│   │   ├── daily.py                # 最新价格、源日期、数据年龄
│   │   ├── hourly.py               # 未实现，显式启用时报错
│   │   ├── price_calendar.py       # 历史价格、滚动、日历与追溯
│   │   └── weather.py              # 旧 weather 组仍报错，使用新宽表接口
│   └── models/
│       ├── __init__.py
│       ├── reference.py
│       └── regression.py           # 简单均值、Ridge、树模型及独立结构
├── tests/
│   ├── test_availability.py
│   ├── test_pipeline.py
│   ├── test_evaluation.py
│   ├── test_commands.py
│   └── conftest.py
└── outputs/                        # 每次运行独立目录；不覆盖
```

市场数据从项目根目录电价目录读取；天气直接读取用户提供的 `develop/data/` 下两份日度CSV，不复制、不修改。动力煤不进入当前读取链路。运行结果和pytest缓存放在 `outputs/`；测试小型数据只在pytest临时目录构造，不作为项目天气样例交付。

## 环境安装

在项目根目录使用 PowerShell：

```powershell
conda env create -f develop/environment.yml
conda activate long_terem_epf
python -m pip install -r develop/requirements.txt
```

环境使用 Python 3.11，运行和测试依赖见上述安装命令。环境已经存在时直接激活，不重复创建。也可以用 `conda run -n long_term_epf python ...` 执行。

## 运行

以下均从项目根目录执行。命令无需指定具体取数时刻；电价及同表电网输入统一按分字段业务边界取数。业务任务仍是在 D 日中午前预测未来六天。

```powershell
# 运行配置 backtest.test 指定的开发调优区间
python develop/run.py backtest

# 六天未知未来预测；actuals.csv 全部为空
python develop/run.py predict --date 2026-10-25

# 重新评分或比较多个结果；命令会创建新的评价目录
python develop/run.py evaluate --runs <运行编号1> <运行编号2>

# 测试临时数据位于开发目录
python -m pytest develop/tests -o cache_dir=develop/outputs/pytest-cache -q
```

从 `develop/` 执行时使用 `python run.py ...`。相对配置路径始终相对 `develop/`，数据路径始终相对项目根目录，评价运行编号始终相对 `develop/outputs/`；不依赖 shell 当前目录。绝对配置和评价目录路径也支持。

每个命令打印输出目录。命令行参数错误退出码为 2，运行异常退出码为 1。找不到合法历史价格时保留失败预测行，在 `checks.json` 中说明原因，运行状态为 `partial`，不能只依据进程成功判断预测完整性。

## 数据边界与标签

| 字段类别 | 日度业务日上限 | 小时记录上限（含端点） |
|---|---|---|
| 实时电价、实际值 | D−1 | D 00:00 |
| 日前电价 | D | D+1 00:00 |
| 电网预测值 | D+1 | D+2 00:00 |

电价 CSV 编码为 GB18030，日期列实际名称为 **`时刻`**。小时业务日为 `date(timestamp - 1 hour)`；日表不再次移日。未知列拒绝进入输入，不能把所有含“预测”字样的列自动视为已知未来数据。

支持可选的 `<字段名>__available_at` 元数据列，非空值必须含时区。元数据保留读取但不进入数值特征，也不参与筛选或扩展业务边界。当前策略为 `business_boundaries_only_v1`，输出明确记录 `publication_time_check: not_run`。已有修订版本尚未独立核验，不能将业务边界检查通过解释为发布时间或历史版本审计通过。

回测和预测启动时自动执行数据完整性检查。标签始终来自日表实时价。核对小时业务日与日表一致、每个业务日有 24 个不同整点、有效标签对应 24 个非空小时价、均价绝对误差不超过 0.001。日价和小时价都未发布的尾部日期可以保留为空；部分小时缺失、重复记录或不一致均阻止正式运行，不会悄悄重算标签。

市场特征模块只能接收按起点截断后的信息集，小时表用于完整性检查，尚未增强预测。天气默认关闭；启用后训练始终读取目标日历史天气。按用户2026-10-29更新，回测推理由 `weather.weather_source: history/forecast` 控制，现有配置默认 history；批量运行继承 `configs/default.yaml`，单次回测使用 `--config` 指定配置。history 标为“理想信息条件下的开发实验”，forecast 标为“历史天气训练、预报天气推理”，两者分别运行、分别评估。forecast 要求预报表完整覆盖 `backtest.test` 闭区间（含内部每一天），不满足时在训练前报错或阻塞，不自动缩短区间。`predict --date D` 保持单日起点，固定使用预报天气推理。所有天气场景均不声称通过具体发布时间或历史版本审计，详见 `WEATHER_DATA.md`。

## 回测区间与训练

| 集合 | 目标日期 | 预测起点 D |
|---|---|---|
| 初始训练历史 | 自 2022-01-01 起，逐次受标签成熟限制 | 自动选取 |
| 开发调优 | 2026-07-01～2026-10-20 | 2026-06-29～2026-10-13 |

回测沿用配置键 `backtest.test`，只选择 D+2～D+7 全部落在该目标区间内的起点，共 107 个起点、642 个预测值。首条曲线为 2026-07-01～07-06，末条曲线为 2026-10-15～10-20。`validation` 配置已删除，旧实验配置也需移除该项，否则会报“未知配置项”；可通过 `--config` 覆盖 `backtest.test` 来调整回测目标区间。历史运行目录中的配置快照保留原样。

训练采用扩展窗口、每日新建模型重训。历史样本 d 必须满足 `d+7 <= D-1`，六天标签全部成熟且有值；其特征按 d 自身的边界构建。允许使用此前已经成熟的标签更新模型，也允许依据当前回测区间的成绩开发调优；该区间不再作为独立保留测试集。

参考模型 `requires_training=False`，调用空训练的 `fit`，不会浪费时间构造无用训练集。可训练模型设为 `True` 后，流程统一调用 `training_set`，其参数拟合也必须在 `fit` 内完成。测试模型已用于验证标签成熟、训练输入与拟合状态对未来数据扰动的不变性；这不等于未来新增模型已通过验收。

## 输出与评价

每次预测或回测保存：

- `config.yaml`：递归合并后的完整配置；列表整体替换，未知项或类型错误报错。
- `metadata.json`：状态、数据及代码哈希、环境版本、时区、数据边界策略、目标区间、预测起点范围、运行用途、模型和评价协议。源文件哈希在输入特征之外，不参与预测。
- `predictions.csv`：每个预测起点一行，固定九列。
- `actuals.csv`：相同日期键和六列真实值；未来预测为空。
- `metrics.csv`：`scope, metric, value, count`；无法定义的指标为空。
- `checks.json`：数据检查、各起点状态与失败原因；未运行的检查明确为 `not_run`。
- `features.json`：特征定义、输入文件与来源日期、原值、数据年龄和可用性依据。
- `report.html`：无需联网的报告，可选择预测起点查看六天曲线、数值、误差分布及指标。

预测表严格为：

```text
predict_time,D+2,D+7,pred_D+2,pred_D+3,pred_D+4,pred_D+5,pred_D+6,pred_D+7
```

真实值表将 `pred_` 替换为 `true_`。CSV 使用 UTF-8 BOM，日期为 `YYYY-MM-DD`，保留数值精度。不同起点预测同一目标日仍是不同业务预测，评分按记录等权。

`evaluation_v1` 定义：

1. 整体及 D+2～D+7 分步长 MAE、MSE；仅对预测和真实值均存在的记录评分。
2. 六天 15 对高低关系一致率，以及 5 个相邻日的涨跌/持平准确率。
3. 内部 4 个日期的峰谷：正转负为峰、负转正为谷，含平价不算拐点。按起点、目标日期和峰谷类型精确匹配，计算总体 precision、recall、F1。
4. 相等容差为 1e-8。零分母记为空；F1 直接按 `2TP/(预测事件数+真实事件数)` 计算。
5. 曲线指标只使用六天全部有效的起点，同时报告预测、可评分和完整曲线覆盖率。

主要比较指标为 MAE，其他指标并列展示，不合成为总分。跨实验比较须有相同市场输入版本、业务规则、评价协议和边界策略，天气宽表版本与实验口径也须一致。允许无天气基线与历史天气研究模型对照，并显示共同成功样本及覆盖率，但不能据此宣称预报天气的实际增益。旧档案口径与新宽表研究不能混合，旧文件不修改。评价只读原结果并创建新目录；曲线展示首个实验。

`evaluate` 只输出新指标、评价元数据和报告，不伪造未执行的训练或预测文件。完整训练/预测运行会生成全部八类文件。

## 后续扩展规则

- 新特征在 `features/` 实现，通过统一信息集读取数据，返回特征与来源记录，并增加对应边界和扰动测试。
- 新模型在 `models/` 实现 `fit(X, y)`、`predict(X)` 和 `requires_training`，在创建入口增加显式分支；预测必须按输入顺序返回有限数值。
- 特征表中的日期和步长是对齐信息；每种模型应明确使用的数值列，不得把所有列未经处理直接喂给模型。
- 每个新实验只增加必要的配置，复用回测与评价代码。不得复制训练脚本形成多个版本。
- 修改指标定义必须实现新协议版本；不能更改配置中的容差却仍称为 `evaluation_v1`。
- 天气接口是否实现与天气可用性是否审计通过分别记录；本轮宽表接口属于用户授权的研究例外，不代表历史预报验证通过。


## TimesFM XReg：timesfm1

在 `long_term_epf` 环境中进入 `develop` 后运行：

```powershell
python run.py batch --suite timesfm1 --check-only
python run.py batch --suite timesfm1
```

全特征基线与五个逐组消融，共六个实验；每个起点分别拟合六次 XReg，
使用512天电价上下文，输出D+2～D+7。单独backtest、predict、预报天气切换、
数据边界及续跑方法见 [TIMESFM.md](TIMESFM.md)。原stage1～stage4保持原有清单。
