# TimesFM 与 timesfm1

使用 `conda activate long_term_epf`（Python 3.11），进入 `develop`。
本地权重：`E:/pycharmproject/epf/models/timesFM`，不自动下载。
依赖见 `requirements.txt`；XReg 使用 CPU JAX，TimesFM 经资源检查后使用 CUDA 或 CPU。

```powershell
python run.py batch --suite timesfm1 --check-only
python run.py batch --suite timesfm1
python run.py batch --suite timesfm1 --resume <批次编号>
python run.py backtest --config configs/experiments/timesfm_xreg.yaml --check-only
python run.py backtest --config configs/experiments/timesfm_xreg.yaml
python run.py predict --config configs/experiments/timesfm_xreg.yaml --date 2026-10-25
python run.py evaluate --runs <实验运行目录>
```

原纯电价接口继续使用 `configs/experiments/timesfm.yaml`。
`timesfm1` 为 full / no_actual / no_dayahead / no_grid_forecast / no_weather / no_calendar，
仅逐组消融，不搜索参数。每组保持上下文512天、ridge=0.01、seed=42、threads=4。
每个起点D分别对h=2..7调用XReg+TimesFM，内部预测D..D+h，取索引h。
协变量为 A[t-h-1]、DA[t-h]、F[t-h+1]、W[t]、C[t]；每列长512+h+1。
因此最终目标行最多使用A[D-1]、DA[D]、F[D+1]，不使用目标日电网记录。
六次调用共享冻结权重，分别进行回归拟合，不合并为一个批次的共享XReg。

全特征包含10个实际值、1个日前价、10个电网预测值、260个天气字段与日历。
天气代码和星期按类别编码；日历另含周末、年内周期sin/cos。
所有原始协变量保留固定缺失标记；数值缺失使用该任务历史512行中位数，
全空为0，类别缺失为独立类别。缩放/编码只拟合历史段；完整日期缺失失败。
所有消融共享完整特征的日期资格，包括no_weather，防止比较样本变化。

`weather.weather_source` 控制backtest/batch未来段天气，默认history；
predict固定使用forecast。历史段一直使用history，未来段需要D..D+7。
forecast回测预先检查整个test及起点桥接日期D、D+1，不截短区间、不跨来源补齐。
现有预报表不能覆盖默认2026-07-01..2026-10-20；本轮正式批次只运行history。

XReg直接拟合当前512个已知日价，元数据标记in_context_xreg，
不会调用树形模型的六天监督样本训练集；training_origins=0，
每个h的fit_label_count=512、fit_label_last=D-1另行记录。
历史六天监督样本的成熟规则没有修改。

输出沿用outputs/<时间>_batch_timesfm1_<标识>/及原有逐实验产物、比较报告、续跑指纹。
features.json记录原始来源哈希、字段、日期、CSV行号及每个h的对齐和填充参数。
history结果属于理想信息条件下的开发实验；电价、电网具体发布时间检查和天气
发布时间/历史版本核验均未执行，不能据此声称真实历史天气预报增益。

## timesfm2：no_actual 历史窗口对比

使用 `long_term_epf` 环境，在 develop 目录运行：

```text
python run.py batch --suite timesfm2 --check-only
python run.py batch --suite timesfm2
python run.py batch --suite timesfm2 --resume <批次目录>
python run.py backtest --config configs/experiments/timesfm_xreg_no_actual.yaml
python run.py predict --config configs/experiments/timesfm_xreg_no_actual.yaml --date 2026-10-25
```

固定七组 L64、L128、L256、L512、L768、L1024、L1536；仅 context_length 不同。
所有组保留实时价格主序列及日前价、电网预测值、天气、日历，不输入实际值。
每个 h 使用 DA[t-h]、F[t-h+1]、W[t]、C[t]，价格长 L，协变量长 L+h+1。
窗口同时改变 XReg 拟合标签数量和 TimesFM 残差上下文；不搜索其他参数。
数值填充、缩放与类别编码只拟合对应 L 个历史位置。七组均按全部107个起点
检查完整窗口资格，沿用 match_origins 的完整特征日期检查，不改变评价日期。

批次只加载一份冻结权重，切换窗口时更新推理配置；回归拟合仍逐起点、逐 h 独立。
L512 为 timesfm1/no_actual 的复现对照，已有 timesfm1 配置与结果不修改。
新增 window_summary.csv 汇总各窗口误差、曲线指标、耗时、相对 L512 变化、
最大绝对误差及最大五项平方误差占比；monthly_metrics.csv 按目标日期月份统计。
同一目标日的不同起点预测作为不同评价点，与原有总体误差口径一致。
结果为开发调优用途，不自动替换默认模型。
