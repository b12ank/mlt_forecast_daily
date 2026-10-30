# 四阶段实验运行说明

本次交付代码、配置和测试用例；可运行自动化测试和只读预检查，正式训练、回测和调参实验由用户运行。具体测试结果以实际执行记录为准。

环境与测试命令从项目根目录执行；后续使用 `python run.py ...` 的批次和评价命令从 `develop/` 执行。程序不自动选模、不自动启动下一批。

## 环境与测试（由你运行）

```powershell
conda activate long_terem_epf
python -m pip install -r develop/requirements.txt
python -m pytest develop/tests -q -o cache_dir=develop/outputs/pytest-cache
```

没有环境时先执行 `conda env create -f develop/environment.yml`。测试只使用合成小数据；模型测试减少训练轮数，不修改正式参数。未安装机器学习库时，对应模型测试会跳过，跳过不代表验收通过。依赖清单指定兼容范围，运行记录实际版本；升级依赖后需重跑验收并新建批次。

## 第一批：六个基线

```powershell
python run.py batch --suite stage1 --check-only
python run.py batch --suite stage1
```

预检查加载配置、导入依赖、核对日表/小时表完整性并显示实验清单，不训练、不创建运行目录。第二条串行运行最近价、近7天均价、近28天同星期均价、Ridge、CatBoost、LightGBM。

参数位于 `configs/suites/stage1.yaml`，默认种子42、树模型CPU四线程。修改 `threads` 可改变树模型线程数，不会并行启动实验。Ridge 底层线程由 NumPy/BLAS 环境管理。

固定特征组 `price_calendar` 包括：

- D−1、D−2、D−3、D−7、D−14 实时价，按准确日期读取。
- 截至 D−1 近3／7／14个日历日均值、总体标准差和有效天数。均值至少一个有效值，标准差至少两个。
- 最近涨跌、距周均值偏差、最近有效价及其年龄，各数值特征缺失标记。
- 目标日星期、周末、步长；同星期均值只用 D−1 及以前最近28天。

简单基线只使用各自对应价格特征，不拟合。可训练模型使用固定完整特征清单。原始日期只用于对齐追溯，不进入模型。训练输入 `6N×p`、标签长度 `6N`；预测输入 `6×p`、输出长度6；不递推预测价格。

## 第二批：人工选择两个或三个模型，比较天气

阅读第一批报告后在 `configs/selections.yaml` 中选择两个或三个不同模型，候选为 `ridge`、`catboost`、`lightgbm`。当前按用户选择配置为全部三个模型：

```yaml
stage2_models: [ridge, catboost, lightgbm]
stage3_models: [lightgbm, catboost]
stage4_combinations:
  - {model: lightgbm, feature_set: F2}
  - {model: catboost, feature_set: F2}
```

```powershell
python run.py batch --suite stage2 --check-only
python run.py batch --suite stage2
```

候选参数读取第一批配置；第一批运行后若改过配置，应恢复对应版本再比较。每个候选各跑三组：选择两个模型生成六个实验，选择三个模型生成九个实验。按配置中的模型顺序，每个模型依次串行运行 F0/F1/F2；不允许重复或未知模型：

| 标识 | 特征 |
|---|---|
| F0 | 历史电价＋日历 |
| F1 | F0＋等权聚合天气 |
| F2 | F0＋现有20类日度天气字段×13站点，共260列的信息及必要编码 |

F1 对风电站最大风速、光伏站短波辐射和日照时长、全部站点平均温度分别计算均值、总体标准差和有效站点数。等权统计不代表装机或负荷权重。

F2 采用用户提供文件的固定260列，不再根据缺失统计删列。日出日落按文件时钟编码为日内分钟，风向保留原值，天气代码作为类别。单元格缺失保留并生成标记。

天气接口见 `WEATHER_DATA.md`，历史与预报文件路径由默认配置 `weather.history`、`weather.forecast` 指定。训练始终使用历史天气；批量回测由 `configs/default.yaml` 的 `weather.weather_source` 选择 history（默认）或 forecast 推理，实际 `predict` 固定使用预报。两种配置各自运行、各自评估，不再要求预报档案或核查清单，也不创建样例文件。

history 模式保留原有行为：历史天气文件缺失、格式错误或完全没有可训练起点时，F1/F2标记 `blocked`，未匹配F0继续。天气可用时三组共同匹配，训练覆盖按历史表检查，推理覆盖按所选来源检查。forecast 模式在训练前要求预报表完整覆盖 `backtest.test` 闭区间，越界或内部缺日会阻塞包括匹配F0在内的相关实验，不缩短区间、不用历史补齐。计划起点数随test区间变化，默认107个。history 报告标为“理想信息条件下的开发实验”，forecast 报告标为“历史天气训练、预报天气推理”，均未执行具体发布时间及历史版本校验。

切换 forecast 时也要修改默认配置的 `backtest.test` 为预报数据覆盖的区间，例如 `['2026-09-25', '2026-10-20']`；程序不自动修改现有默认区间。切回 history 后可自行恢复原区间。

## 第三批：天气输入消融，不重跑 F2

`stage3_models` 支持从 ridge/catboost/lightgbm 中选一至三个不同模型，默认 `[lightgbm, catboost]`。每个模型固定依次运行 F3/F4/F5，共六个串行实验；使用第一批对应参数、共享结构、种子42和四线程。

| 特征组 | 输入 | 共享模型编码前列数 |
|---|---|---:|
| F3 | 全量天气260列＋天气缺失标记260列 | 520 |
| F4 | 全量天气260列，无天气缺失标记 | 260 |
| F5 | F0的42列＋全量天气260列，无天气缺失标记 | 302 |

F5保留F0自身20列缺失标记。F3/F4不使用价格、星期、周末或步长输入；日期和horizon只用于对齐与独立模型路由。无缺失标记不等于填零或删行，数值NaN及模型原有缺失处理保留。天气代码仍作为类别，Ridge独热编码后的维度另计。

```powershell
python run.py batch --suite stage3 --check-only
python run.py batch --suite stage3
```

三组统一使用与F2一致的训练资格、完整成熟标签及天气覆盖条件。纯天气组只检查边界内是否存在历史价格以保持共同样本资格，不构造价格特征。`metadata.json` 和 `features.json` 的 `input_schema` 记录模型实际输入列、编码前列数和天气缺失标记策略。

对照已有stage2的F2时，无需重跑F2。先核对输入数据版本、每个起点的训练样本、模型参数和预测起点一致，再比较共同样本：

```powershell
python run.py evaluate --runs <stage2批次编号/F2子实验目录> <stage3批次编号/F3子实验目录> <stage3批次编号/F4子实验目录> <stage3批次编号/F5子实验目录>
```

评价入口检查数据版本、规则、评价协议和真实值一致性；参数与训练样本仍需结合配置、checks.json和features.json审阅。回测继续属于理想信息条件下的开发实验。

## 第四批：参数与共享/独立结构比较

人工填写选择，例如：

```yaml
stage2_models: [ridge, catboost, lightgbm]
stage3_models: [lightgbm, catboost]
stage4_combinations:
  - {model: ridge, feature_set: F0}
  - {model: catboost, feature_set: F1}
```

```powershell
python run.py batch --suite stage4 --check-only
python run.py batch --suite stage4
```

每个组合可选F0～F5，四组参数×两种结构，共八个实验；两个组合为十六个。参数清单从原stage3完整迁移到 `configs/suites/stage4.yaml`。当前保留原选择的LightGBM F2和CatBoost F2，不自动加入消融获胜组合：

- `shared`：六个步长共享一个标量模型。
- `independent`：六个步长分别拟合模型和预处理，移除步长特征，按输入行原顺序返回预测。

独立结构训练量更大，但不放宽完整六天标签成熟规则。独立模型F3/F4仍为520/260列，F5移除horizon后为301列。第四批F0默认使用全部合法价格起点，F1～F5训练受历史天气覆盖限制、推理受所选天气来源覆盖限制；跨特征组合比较时要审阅训练数量及共同预测样本。如需第四批F0也采用天气共同样本，可设置 `weather.match_origins: true` 并启动新批次。报告按所选天气来源解释。

旧的外部选择文件须将 `stage3_combinations` 改为 `stage4_combinations`，并新增 `stage3_models`；程序遇到旧键会提示迁移。旧outputs不改名、不修改，代码/配置指纹变化后启动新批次，不可把旧stage3调参批次作为新stage3续跑。

## 输出、续跑与异常

每批在 `outputs/` 创建独立目录，终端显示实验和起点进度：

- `batch.json`：完整清单、人工选择、版本指纹、预检查、状态、每次尝试。
- `experiments.csv`：状态、耗时、训练起点数量范围、失败起点数、路径和错误。
- `metrics.csv`、`report.html`：各自和共同成功样本的整体/分步长MAE、MSE、排序、方向、拐点及覆盖率；无加权总分、无自动晋级。
- 各实验独立目录：原有八类文件，包括来源记录和可追溯特征。
- `comparison_*`：每次独立汇总目录，保留旧评价。

打开批次报告即可进入各实验六天曲线报告。

```powershell
python run.py batch --suite stage1 --resume <原批次编号或绝对路径>
```

只跳过产物完整、哈希一致且状态为 `passed` 的实验。中断、失败、部分完成实验从该实验第一个起点重新运行，使用新尝试目录，不覆盖旧尝试。配置、候选、代码、使用的数据、规则或依赖变化时拒绝续跑。补齐历史天气文件或安装缺失依赖属于版本变化；history 模式不读取预报文件，单独更新预报内容不影响其指纹；forecast 模式将两份天气文件哈希纳入指纹，预报文件变化或切换来源后须启动新批次。本次接口变更后请启动新批次。

公共数据完整性错误阻止整批启动；单个实验缺依赖、天气阻塞或运行失败会保留原因并继续其余实验。退出码：全部完整成功为0；阻塞、失败、部分完成、中断或汇总错误为1；命令参数错误为2。预检查部分阻塞也返回1，应阅读清单判断哪些实验可运行。

`batch.lock` 防止同一批次同时续跑；正常退出和Ctrl+C会移除。强制结束进程可能遗留锁，确认原进程停止后，仅手动删除该批次锁文件再续跑。

可用 `--selections <文件>` 指定其他人工选择配置。相对配置路径以 `develop/` 为基准，数据路径以项目根目录为基准，续跑编号以 `develop/outputs/` 为基准。

## 单次运行与边界

```powershell
python run.py backtest
python run.py predict --date 2026-10-27
python run.py evaluate --runs <运行目录1> <运行目录2>
```

默认单次命令仍是参考模型。可将某次实验保存的完整 `config.yaml` 作为 `--config` 输入，执行单次回测或未来预测，并生成新目录。评价批次子实验时使用绝对路径或 `批次编号/子实验编号`。

目标区间保持2026-07-01～2026-10-20，107个起点、642个预测位置，`evaluation_v1`不变。成绩均为开发调优结果，没有独立保留测试集。实时价到D−1；训练d满足 `d+7 <= D-1` 且六标签齐全；预处理只在当次训练集拟合。

市场数据保留 `publication_time_check: not_run`；天气的具体发布时间和历史版本校验也均为 `not_run`。实际 `predict` 使用历史天气训练、预报天气推理，读取预报表中D+2～D+7；缺少整天记录会在训练前报错，不从历史表补齐。报告和元数据标明这一输入来源差异，不伪造可用时点或审计通过结论。
