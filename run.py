"""CLI; relative config paths are relative to develop, not the shell cwd."""
import argparse
import json
import sys

from forecast.pipeline import load_config, run_experiment, evaluate_runs, check_backtest


def main(argv=None):
    parser = argparse.ArgumentParser(description="D+2～D+7 实时电价日均预测")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("backtest", "predict"):
        command = sub.add_parser(name)
        command.add_argument("--config", default="configs/experiments/reference.yaml")
        if name == "backtest":
            command.add_argument("--check-only", action="store_true",
                                 help="只检查配置、数据、依赖和权重，不训练、不预测、不生成回测结果")
        if name == "predict":
            command.add_argument("--date", required=True, help="预测起点 YYYY-MM-DD")
    command = sub.add_parser("evaluate")
    command.add_argument("--runs", nargs="+", required=True, help="outputs 下的运行编号或绝对路径")
    command = sub.add_parser("batch", help="串行运行人工选择的实验批次")
    command.add_argument("--suite", choices=["stage1", "stage2", "stage3", "stage4", "timesfm1", "timesfm2"], required=True)
    command.add_argument("--check-only", action="store_true", help="只检查，不训练、不写运行结果")
    command.add_argument("--resume", help="原批次目录或 outputs 下的批次编号")
    command.add_argument("--selections", help="人工选择配置；相对 develop 或绝对路径")
    args = parser.parse_args(argv)
    try:
        if args.command == "batch":
            from forecast.batch import run_batch
            output, code = run_batch(args.suite, args.check_only, args.resume, args.selections)
            if output is not None:
                print(output)
            return code
        elif args.command == "evaluate":
            output = evaluate_runs(args.runs)
        elif args.command == "backtest" and args.check_only:
            inspection = check_backtest(load_config(args.config))
            print(json.dumps(inspection, ensure_ascii=False, indent=2, allow_nan=False))
            return 0 if inspection["status"] == "passed" else 1
        else:
            output = run_experiment(load_config(args.config), args.command,
                                    getattr(args, "date", None))
    except (ValueError, OSError, NotImplementedError, ImportError) as exc:
        print(f"失败: {exc}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
