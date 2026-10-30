# 天气日度宽表接口

直接使用用户提供的CSV，不再要求逐次运行档案、发布时间元数据或核查清单，也不生成天气样例文件。用户于2026-10-27确认的研究例外记录在项目 `AGENTS.md` 和 `DATA_AVAILABILITY.md`。

## 文件与运行模式

默认配置（路径相对项目根目录）：

```yaml
weather:
  weather_source: history  # history 或 forecast，仅控制回测推理
  history: develop/data/weather_data_daily.csv
  forecast: develop/data/weather_data_daily_predict.csv
  match_origins: false
```

| 场景 | 训练天气 | 预测输入天气 | 结果含义 |
|---|---|---|---|
| 回测/批量：`weather_source: history` | 历史表d+2～d+7 | 历史表D+2～D+7 | 理想信息条件下的开发实验 |
| 回测/批量：`weather_source: forecast` | 历史表d+2～d+7 | 预报表D+2～D+7 | 历史天气训练、预报天气推理 |
| `predict --date D` | 历史表d+2～d+7 | 预报表D+2～D+7 | 历史天气训练、预报天气推理 |

history 回测不读取、不校验、不计算预报文件哈希，预报文件缺失或内容更新不会影响其结果与续跑指纹。forecast 回测和实际预测必须读取预报文件，记录其哈希，预报缺日不能用历史表补齐。`predict --date D` 不受配置开关影响，始终使用预报天气推理。

单次回测修改 `--config` 指定配置中的 `weather.weather_source`；批量实验修改 `configs/default.yaml`，各阶段继承它。现有配置默认 history，遗漏该项时继承默认值；只接受 history/forecast。两种模式分别运行、分别评估，无需混合汇总。

forecast 回测在训练前要求预报表覆盖 `backtest.test` 闭区间每一天，包括内部缺日检查。失败时显示目标区间、预报覆盖范围及缺失日期，不自动缩短区间；批量运行将相关实验标记为 blocked。配置示例（目标区间须根据所用文件设置）：

```yaml
weather:
  weather_source: forecast
backtest:
  test: ['2026-09-25', '2026-10-20']
```

当前默认 test 保持 `2026-07-01～2026-10-20`，切换 forecast 前应检查覆盖。日期覆盖通过不代表单元格完整。未使用天气且未启用 `match_origins` 的基线不依赖天气文件。

历史目标日天气回测不是历史预报回测精度，不能据此宣称预报天气的实际增益。实际预测以用户提供的文件作为D日起点预报快照；程序不会根据文件末行日期推断D，也不声称自动验证了该文件在D日的发布时间或历史修订版本。训练/推理天气来源分布存在差异，历史天气成绩不能直接作为运行精度。

## 结构与编码

- UTF-8（可带BOM），首列 `time`，格式 `YYYY-MM-DD 00:00:00`；日期唯一。文件允许存在日期缺口，只有覆盖六个完整目标日的起点才能使用。
- 其余260列为现有20类天气字段×13站点，名称采用 `字段_站点`。程序校验完整字段清单；读取预报文件时要求两份文件的列名与顺序一致，不根据全历史缺失率删列。
- 日期直接对应目标日，不套用电价小时业务日移日，不推断时区。
- 数值单元格可为空，保留NaN；是否生成天气缺失标记由特征组决定。Ridge中位数填补和标准化仅在每次训练集拟合，树模型保留数值缺失。
- 天气代码为整数类别；日出日落按 `%d/%m/%Y %H:%M:%S` 解析，日期应与本行天气日期一致，按文件时钟转换为日内分钟。风向保留原数值，不猜测单位后做正余弦转换。

F1：风电站最大风速、光伏站短波辐射与日照时长、全体站点平均温度，分别计算等权均值、总体标准差及有效站点数。标准差至少需要两个非空站点，等权统计不代表装机或负荷权重。

F2：使用全部260列的信息，加上缺失标记；日出日落转为分钟，天气代码保持类别，其他字段原值输入。

F3：仅 `weather_full`，天气260列＋天气缺失标记260列，共520列。F4：仅 `weather_full_no_missing`，保留天气260列，不生成天气缺失标记。F5：`price_calendar`＋`weather_full_no_missing`，F0的42列＋天气260列，共302列；F0自身缺失标记保留。

F3/F4没有价格、星期、周末或horizon模型输入，日期与horizon保留为对齐/路由信息；不伪造价格source_date。260列天气含247个数值及13个类别字段，维度为独热编码前。缺失天气代码仍使用模型的缺失类别表示，无标记组不额外添加二元缺失列。缓存同时区分history/forecast及weather_full/weather_full_no_missing。

## 覆盖、记录与比较

第二批F0/F1/F2和第三批F3/F4/F5统一按历史天气检查训练覆盖，按所选来源检查回测推理覆盖，并保持边界内存在历史有效价格的共同资格。forecast 模式共同匹配的F0也受预报覆盖约束，失败时同样阻塞。history 模式保留原有天气完全不可用时未匹配F0继续运行的行为。计划起点数由 test 区间决定，默认区间为107个。纯天气组的价格资格检查不产生价格输入。第四批沿用所选组合，F0是否同样限制天气覆盖由 `match_origins` 决定。完整六天电价标签成熟规则与实时电价D−1边界不变。

`features.json`记录天气来源类型、文件路径/哈希、原始CSV行号、目标日期、每列原始文本、编码定义和缺失处理。缓存键区分历史与预报来源，不因相同起点串用数据。

元数据包含：

- `weather_protocol: wide_daily_weather_research_v1`
- `weather_scenario`：history 回测为 `history_target_weather_research`，forecast 回测为 `history_train_forecast_backtest`，单日预测为 `history_train_forecast_inference`；无天气输入为 `none`
- `weather_training_source`、`weather_prediction_source`、`weather_matching_source`（共同样本限制的来源）
- `weather_publication_time_check: not_run`
- `weather_historical_revision_audit: not_run`

报告明确显示对应场景。无天气基线可以与历史天气研究实验比较，但不能混合旧档案口径、历史天气研究回测与预报推理成绩。天气接口或规则变更后启动新批次，旧运行结果不修改。

forecast 回测使用用户提供的预报宽表，不声称复原各历史起点的预报信息集；具体发布时间与历史版本校验仍未执行。`checks.json` 记录整个 test 区间的预报覆盖检查与来源文件。

测试用小数据仅在pytest临时目录构造，不在项目数据目录产生样例。自动化测试和预检查可用于交付验证，正式实验由用户运行。
