"""默认并行回测全部客户、保留模型原始排名，比较历史全量及同模型提示词基线。"""

from __future__ import annotations

from evaluate_real_cases import main as evaluate
from report_comparison import DEFAULT_BASELINE_PATH

DEFAULT_WORKERS = 16


def main(argv: list[str] | None = None) -> int:
    return evaluate(
        argv, default_all=True, default_workers=DEFAULT_WORKERS,
        default_baseline=DEFAULT_BASELINE_PATH,
    )


if __name__ == "__main__":
    raise SystemExit(main())
