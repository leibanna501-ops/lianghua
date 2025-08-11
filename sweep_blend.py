# sweep_blend.py
from __future__ import annotations
import sys, itertools, json, time
from copy import deepcopy
from pathlib import Path

from acs import discover
from acs.config import load_cfg
from acs.pipeline import run
from acs.eval import evaluate
from acs.registry import REGISTRY


def _ts():
    return time.strftime("%Y%m%d_%H%M%S")


def main(cfg_path: str):
    discover()  # 确保 fetcher/feature/label/model/blend 都已注册

    base = load_cfg(cfg_path)

    # ========= 1) 融合参数 sweep =========
    print("Running blend sweep…")
    w_range = [round(i * 0.05, 2) for i in range(21)]
    grid = list(itertools.product([False, True], w_range))
    results = []
    for invert_sup, w_unsup in grid:
        c = deepcopy(base)
        c.blend.params.update({"invert_sup": invert_sup, "w_unsup": w_unsup})
        csv_path = run(c)
        metrics, _ = evaluate(cfg_path, csv_path=csv_path)
        rec = {"invert_sup": invert_sup, "w_unsup": w_unsup, **metrics}
        results.append(rec)
        print(
            f"trial invert={invert_sup} w={w_unsup} -> auc={metrics['auc']} brier={metrics['brier']}"
        )

    # 排序&Top5
    results.sort(
        key=lambda r: (
            -(r.get("auc") or 0),
            r.get("brier") if r.get("brier") is not None else 9,
        )
    )
    print("\nTOP trials:")
    for r in results[:5]:
        print(
            json.dumps(
                {
                    k: r[k]
                    for k in ["invert_sup", "w_unsup", "auc", "brier", "coverage"]
                },
                ensure_ascii=False,
            )
        )

    # 落盘
    sym = results[0].get("symbol") or base.data.symbol
    out_dir = Path(base.run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _ts()
    blend_csv = out_dir / f"sweep_blend_{sym}_{stamp}.csv"
    blend_json = out_dir / f"sweep_blend_{sym}_{stamp}.json"
    try:
        import pandas as pd

        pd.DataFrame(results).to_csv(blend_csv, index=False)
    except Exception:
        pass
    with open(blend_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nSaved blend sweep to:")
    print(f"  {blend_csv}")
    print(f"  {blend_json}")

    # 取最优融合参数
    best = results[0]
    best_invert = bool(best["invert_sup"])
    best_w = float(best["w_unsup"])

    # ========= 2) 校准器 sweep（两处补强：自动过滤缺失 & 容错落盘）=========
    print(
        f"\nRunning calibrator sweep at best blend (invert_sup={best_invert}, w_unsup={best_w})…"
    )

    # ① 自动识别可用校准器，只扫存在的
    desired = ["platt", "isotonic", "legacy_platt_isotonic"]
    avail = set(REGISTRY.names("calibrator"))
    calibs = [c for c in desired if c in avail]
    missing = [c for c in desired if c not in avail]
    if missing:
        print(f"Warning: missing calibrators skipped: {missing}")
    print(f"Available calibrators: {sorted(avail)}")
    if not calibs:
        print("No requested calibrators available. Skip calibrator sweep.")
        return

    calib_rows = []
    for calib in calibs:
        try:
            c = deepcopy(base)
            c.blend.params.update({"invert_sup": best_invert, "w_unsup": best_w})
            c.model.calibrator = calib
            csv_path = run(c)
            metrics, _ = evaluate(cfg_path, csv_path=csv_path)
            calib_rows.append({"calibrator": calib, **metrics})
            print(f"{calib} -> auc={metrics['auc']:.3f} brier={metrics['brier']:.6f}")
        except Exception as e:
            print(f"[skip] calibrator '{calib}' failed: {e}")

    # 落盘校准器结果
    if calib_rows:
        calib_csv = out_dir / f"sweep_calib_{sym}_{stamp}.csv"
        calib_json = out_dir / f"sweep_calib_{sym}_{stamp}.json"
        try:
            import pandas as pd

            pd.DataFrame(calib_rows).to_csv(calib_csv, index=False)
        except Exception:
            pass
        with open(calib_json, "w", encoding="utf-8") as f:
            json.dump(calib_rows, f, ensure_ascii=False, indent=2)
        print("\nSaved calibrator sweep to:")
        print(f"  {calib_csv}")
        print(f"  {calib_json}")


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python sweep_blend.py <path-to-yaml>"
    main(sys.argv[1])
