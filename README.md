# D+2～D+7 实时电价日均预测

在起点 D 预测 D+2～D+7 六个业务日的实时电价日均价。日常上线使用 `load-data` 同步数据、`predict` 生成本地报告、`submit` 上传结果；`backtest`、`evaluate` 和 `batch` 接口继续保留，程序、依赖及结果格式不变。

## 部署目录与环境

本文以 **`develop/` 仓库目录为运行根目录**，所有运行、安装和测试命令均在该目录执行。目录布局：

```text
develop/
  configs/
  forecast/
  run.py
  requirements.txt
  .env.example
  .env                   # 本地创建，不提交
  data/                  # load-data 获取，不提交
  outputs/               # 运行时生成，不提交
```

首次下载时，将地址替换为实际仓库地址；已有检出直接进入 `develop/`：

```powershell
git clone <仓库地址> develop
Set-Location develop
```

进入仓库后准备环境：

```powershell
# 仅首次部署时创建，已有 .env 时不要覆盖
conda create -n env_name python=3.11
conda activate env_name
python -m pip install -r requirements.txt
```

`--config` 的相对路径以仓库目录为基准，支持绝对路径。配置内部的数据相对路径仍由程序按仓库上一层解析，例如 `develop/data/...` 指向本仓库的 `data/`；终端工作目录不会改变这一规则。

编辑本地 `.env`，设置 `DB_HOST`、`DB_PORT`、`DB`、`DB_USER`、`DB_PWD`；示例只有占位值，同名环境变量优先。真实凭据不随仓库提供。TimesFM XReg 还需自行准备本地权重，并将所用配置的 `model.params.checkpoint_dir` 设为实际目录；仓库保留原路径，不自动下载权重。

## 日常预测流程

```powershell
# 1. 读取源数据库，更新三份日度输入；不自动预测
python run.py load-data

# 2. 正式提交：默认固定 LightGBM、北京时间当天，写入结果表
python run.py submit
```

## 其他命令
```powershell
# 本地预测并结果保存在outputs目录；不写数据库(2026-11-13预测2026-11-15~2026-11-20)
python run.py predict --date 2026-11-13

# 支持本地一段范围的预测结果保存在outputs目录；不写数据库
python run.py submit --date 2026-10-01 2026-10-10

# 单日预测结果保存到数据库
python run.py submit --date 2026-11-13

# 支持回传一段范围的预测结果保存到数据库
python run.py submit --date 2026-10-01 2026-10-10

# 本地预测通过配置文件控制模型、日期范围
python run.py predict --config configs/predict/predict_example.yaml

# 提交也支持通过配置文件指定参数（指定模型、日期范围）
python run.py submit --config configs/submit/submit_example.yaml
```

`--load-data`、`--submit` 分别是对应子命令的别名。示例日期请按数据覆盖调整：例如起点 `2026-11-13` 需要到 `2026-11-20` 的天气预报，缺日时单日调用失败。`predict` 不是 `submit` 的前置步骤，两者都不会自动同步数据。

训练始终使用历史天气。单日提交严格使用预报；日期范围逐起点检查，缺预报时整个推理天气区间回退历史天气，历史也缺日则失败。树模型需 D+2～D+7，TimesFM XReg 需 D～D+7 和截至 D−1 的历史窗口。历史回退标注“理想信息条件下的历史补算”；具体发布时间与历史版本校验均未执行。

提交只读 `data/` 最新三份日度文件，不生成 `outputs`。`--config` 与 `--date` 互斥，`--dry-run` 可与两者分别组合。全部预测和校验成功后才在一个事务中写入 `epf_mlt_forecast_daily_result`，按 `(forecast_date, target_date, product_type, model_name)` 更新已有记录。
