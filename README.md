# ACS Refactor — 项目说明（README）

> 这是一份“拿来就能看懂 & 跑起来”的说明文档：项目做什么、当前实现、如何扩展、如何运行、常见问题、后续路线图，全部放在一起。  
> 目标读者：第一次接触项目的开发者 / 未来的自己 / AI 代码助手。

---

## 目录

- [项目愿景（一句话）](#项目愿景一句话)
- [当前已经实现的功能](#当前已经实现的功能)
- [项目结构与职责划分](#项目结构与职责划分)
- [配置驱动（YAML）](#配置驱动yaml)
- [核心流水线](#核心流水线)
- [命令行用法（CLI）](#命令行用法cli)
- [自动扫参脚本：sweep_blend.py](#自动扫参脚本sweep_blendpy)
- [数据接口与兼容旧代码](#数据接口与兼容旧代码)
- [扩展指南（写你自己的插件）](#扩展指南写你自己的插件)
- [输出文件说明](#输出文件说明)
- [评估指标与校准概念](#评估指标与校准概念)
- [环境准备与安装](#环境准备与安装)
- [快速开始（Quickstart）](#快速开始quickstart)
- [常见问题（Troubleshooting）](#常见问题troubleshooting)
- [路线图（Roadmap）](#路线图roadmap)
- [贡献指南（Contribution）](#贡献指南contribution)
- [版本与变更记录（Changelog stub）](#版本与变更记录changelog-stub)
- [许可证](#许可证)

---

## 项目愿景（一句话）

把一只或多只股票的**历史行情数据** → 计算**特征** → 打**标签**（未来涨跌结果） → 产出**概率预测** → 做**融合与概率校准** → 输出**可评估、可重复**的结果文件与指标。  
这是一条从“数据”到“可信概率”的小流水线，强调**配置驱动**、**插件化扩展**和**可复现**。

---

## 当前已经实现的功能

- **命令行工具（CLI）**
  - `run-cfg <yaml>`：按配置跑一遍流水线，落盘预测 CSV。
  - `describe <csv>`：查看预测文件的统计与末尾几行（JSON 输出）。
  - `eval <yaml> [csv]`：按配置重算标签并对某个预测 CSV 打分（AUC/Brier/coverage），输出校准数据和图片。

- **自动化扫参脚本**：`sweep_blend.py`
  - 网格扫描融合参数：`invert_sup ∈ {False, True}`，`w_unsup ∈ {0.0..0.4}`（可扩展）。
  - 对最佳融合配置做校准器横评（已自动跳过未注册的校准器）。
  - 将结果以 `csv/json` 落盘。

- **可插拔架构（Registry + discover）**
  - 取数（fetcher）、特征（feature）、标签（label）、融合（blend）、校准（calibrator）均通过注册表管理。
  - 新增策略只需写模块 + `@REGISTRY.register(...)` 装饰器，无需改主流程。

- **数据兼容旧代码**
  - 优先调用你的 **legacy** 模块 `acs_panel_unified_3.py` 的 `fetch_daily(...)`，新框架只是适配层。
  - 提供 `call_legacy()` 自动补/过滤参数（如 `debug_log`），减少签名不匹配错误。

- **评估与可视化**
  - 指标：AUC（区分能力）、Brier（概率可信度）、coverage（覆盖率）。
  - 可靠性（Calibration）数据与图（`outputs/*_calibration.csv/.png`）。
  - `describe` 命令以 JSON 输出统计结果，便于脚本/前端继续处理。

- **缓存与日志**
  - Joblib 缓存与过程计时（见 `acs/utils/logging.py`），复跑更快，日志可追踪。

- **目前的最佳结果（示例）**
  - 在 `example.yaml` 下，最佳融合参数为 `invert_sup=True, w_unsup=0.4`
  - 典型指标：AUC ≈ 0.532，Brier ≈ 0.2756，coverage ≈ 0.992（随数据时间窗可能变化）

---

## 项目结构与职责划分

```
acs_refactor_skeleton/
├─ configs/
│  └─ example.yaml                # 配置示例（YAML）
├─ outputs/                       # 运行产物（CSV/PNG/JSON等）
├─ src/acs/
│  ├─ __init__.py                 # discover(): 统一注册插件入口
│  ├─ cli.py                      # Typer 命令行：run-cfg / describe / eval
│  ├─ registry.py                 # REGISTRY 实现与注册装饰器
│  ├─ config.py                   # Pydantic 配置模型与 load_cfg()
│  ├─ pipeline.py                 # 主流水线：fetch -> feature -> label -> blend -> calibrate -> save
│  ├─ eval.py                     # 评估：对预测与标签对齐，计算 AUC/Brier/coverage，生成校准数据/图
│  ├─ utils/
│  │  ├─ logging.py               # 日志与计时装饰器
│  │  └─ compat.py                # call_legacy(): 旧代码函数签名适配
│  ├─ io/
│  │  └─ fetchers.py              # fetch_daily(): 优先走 legacy；否则给占位空框架
│  ├─ features/
│  │  └─ ma_converge.py          # 示例特征：均线靠近度
│  ├─ labeling/
│  │  └─ weak_label_v1.py        # 示例标签：未来horizon天涨幅>threshold
│  ├─ blend/
│  │  └─ regime_gate.py          # 融合：regime_gate_oscillator（w_unsup, invert_sup）
│  └─ model/
│     └─ calibration.py          # 校准器：legacy_platt_isotonic（已注册）
├─ sweep_blend.py                 # 扫描 blend 参数与 calibrator 横评脚本
├─ acs_panel_unified_3.py         # 你的旧数据拉取代码（legacy，作为优先数据源）
└─ README.md (建议用本文档替换或作为 docs/README.md)
```

**各文件要点**
- `src/acs/__init__.py`  
  统一 `discover()`：import 各子模块以触发 `@REGISTRY.register()` 的副作用注册。
- `src/acs/registry.py`  
  简单的注册表：`REGISTRY.register(kind, name)` 装饰器；`REGISTRY.get(kind, name)` 获取实现。
- `src/acs/config.py`  
  Pydantic 数据类：`Cfg/DataCfg/FeatureCfg/LabelCfg/ModelCfg/BlendCfg/RunCfg`，以及 `load_cfg(path)`.
- `src/acs/pipeline.py`  
  核心流程：取数→排序→特征→标签→融合→校准→导出；并做 dtype/缺失处理、计时、缓存策略等。
- `src/acs/eval.py`  
  按配置重算标签，与预测对齐后计算指标；输出校准数据/图。
- `src/acs/io/fetchers.py`  
  对 `acs_panel_unified_3.py` 的 `fetch_daily` 做优先调用；并通过 `call_legacy()` 自动适配参数。
- `src/acs/model/calibration.py`  
  当前已注册 `legacy_platt_isotonic`；预留 `platt` 与 `isotonic` 插件位。
- `sweep_blend.py`  
  从配置复制出多个变体，循环调用 `run()` 与 `evaluate()`，将结果排行落盘。

---

## 配置驱动（YAML）

`configs/example.yaml`（示意）：

```yaml
data:
  symbol: "000001"          # 标的代码（示例）
  start: "2023-01-01"       # 开始日期（含）
  end: "2025-08-11"         # 结束日期（含）；可留空，但建议显式给出
  fetcher: "akshare"        # fetcher 名（注册名）；当前桥接到 legacy 的 akshare 拉取

feature:
  names: ["ma_converge"]    # 允许多个特征名
  params:
    ma_converge:
      ma_short: 20
      ma_long: 60

label:
  name: "weak_label_v1"
  params:
    horizon: 5              # 未来5天
    threshold: 0.03         # 涨幅>3% 记为正例

model:
  calibrator: "legacy_platt_isotonic"
  params: {}                # 如未来有模型/校准器参数可放这里

blend:
  name: "regime_gate_oscillator"
  params:
    w_unsup: 0.3            # 无监督分量权重（0~1）
    invert_sup: true        # 是否反转监督信号

run:
  cache_dir: ".cache"
  out_dir: "outputs"
  seed: 42
  fp32: true                # 强制 float32，减少内存差异
```

---

## 核心流水线

1. **Fetch（取数）**  
   - 来自 `acs_panel_unified_3.py` 的 `fetch_daily(symbol, start, end, debug_log, ...)`（优先）；  
   - 列名标准化为小写（`open, high, low, close, volume, amount, ...`）；  
   - 时间索引排序，上下游使用 `DatetimeIndex`。

2. **Feature（特征）**  
   - 示例：`ma_converge` 计算短长均线的靠近程度；  
   - 允许注册多个特征，结果合并为一个特征表。

3. **Label（标签）**  
   - 示例：`weak_label_v1(horizon, threshold)` → 未来 `horizon` 天的前向收益率是否超过阈值；  
   - 用于监督分量与评估对齐。

4. **Blend（融合）**  
   - `regime_gate_oscillator`：将监督概率与无监督状态做加权/门控；  
   - 支持 `w_unsup`（无监督权重）与 `invert_sup`（反转监督信号）。

5. **Calibration（校准）**  
   - 目前注册了 `legacy_platt_isotonic`；  
   - 目标是让模型输出的 **概率** “更像概率”（可靠性曲线更贴近对角线）。

6. **Export（导出）**  
   - 预测 CSV（命名含 symbol 哈希），至少包含：
     - `p_sup`：监督分量概率  
     - `p_final`：融合+校准后的最终概率  
   - `eval` 时会另写：
     - `*_calibration.csv` / `*_calibration.png`

---

## 命令行用法（CLI）

> **注意**：所有命令从项目根目录执行（与 `src/` 同级）。Windows PowerShell 中不要带 `>>>` 提示符。

```bash
# 1) 跑一遍流水线并落盘预测
python -m acs.cli run-cfg configs/example.yaml

# 2) 查看预测文件的统计与末尾5行（JSON）
python -m acs.cli describe outputs/000001_*.csv

# 3) 评估（按配置重算标签；如果不指定 csv，会尝试自动找最新）
python -m acs.cli eval configs/example.yaml
# 输出：
# {
#   "symbol": "000001",
#   "csv": "outputs\000001_....csv",
#   "coverage": 0.992076,
#   "auc": 0.532,
#   "brier": 0.27562,
#   "calibration_csv": "outputs\000001_calibration.csv",
#   "calibration_png": "outputs\000001_calibration.png"
# }
```

---

## 自动扫参脚本：`sweep_blend.py`

功能：
- 扫描 `invert_sup ∈ {False, True}` 与 `w_unsup ∈ {0.0,0.1,...,0.4}`；
- 输出每次试验的 AUC/Brier，并将 Top5 打印与落盘；
- 在最佳融合配置处，尝试不同校准器（若某校准器未注册，会**警告并跳过**）；
- 将结果分别保存为 `outputs/sweep_blend_*.csv/.json` 与 `outputs/sweep_calib_*.csv/.json`。

用法：

```bash
python sweep_blend.py configs/example.yaml
```

（可按需扩展：支持 `--w-grid 0.0,0.05,...`、`--symbols 000001,600000,...` 并行等）

---

## 数据接口与兼容旧代码

- **优先使用**你的旧模块 `acs_panel_unified_3.py` 的 `fetch_daily()`（akshare 数据源）。  
- 通过 `src/acs/utils/compat.py` 的 `call_legacy()` 自动补齐参数（如 `debug_log`）与过滤不需要的参数。  
- 如果 legacy 拉取失败或返回空，`fetchers.py` 有占位兜底（业务上建议尽快补全/排查）。

**旧函数签名（示意）**：
```python
def fetch_daily(
    symbol: str,
    start,
    end,
    debug_log: list,
    chunk_days: int = 120,
    max_retries: int = 4,
    base_sleep: float = 0.8,
) -> pd.DataFrame:
    ...
```
---

## 扩展指南（写你自己的插件）

### 1) 新增特征（feature）

```python
# src/acs/features/my_feature.py
import pandas as pd
from acs.registry import REGISTRY

@REGISTRY.register("feature", "my_feature")
def my_feature(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    # 举例：简单动量
    win = int(params.get("window", 10))
    out["mom"] = df["close"].pct_change(win)
    return out
```

- 在 `src/acs/__init__.py` 的 `discover()` 中**确保 import**：  
  ```python
  from .features import my_feature as _
  ```
- 在 YAML 中启用：
  ```yaml
  feature:
    names: ["ma_converge", "my_feature"]
    params:
      my_feature:
        window: 10
  ```

### 2) 新增标签（label）

```python
# src/acs/labeling/my_label.py
import pandas as pd
from acs.registry import REGISTRY

@REGISTRY.register("label", "my_label")
def my_label(df: pd.DataFrame, params: dict) -> pd.Series:
    horizon = int(params.get("horizon", 5))
    r = df["close"].pct_change(horizon).shift(-horizon)
    return (r > 0.02).astype("float32").rename("y")
```

- `discover()` 中 import；YAML 里 `label.name: "my_label"`。

### 3) 新增融合（blend）

```python
# src/acs/blend/my_blend.py
import pandas as pd
from acs.registry import REGISTRY

@REGISTRY.register("blend", "my_blend")
def my_blend(df: pd.DataFrame, sup: pd.Series, params: dict) -> pd.Series:
    # sup：监督概率；你也可以用 df 里的无监督状态
    alpha = float(params.get("alpha", 0.5))
    # 举例：简单滑动平滑
    sm = sup.ewm(alpha=alpha, adjust=False).mean().clip(0,1)
    return sm.rename("p_blend")
```

- `discover()` 中 import；YAML 里 `blend.name: "my_blend"` + `blend.params.alpha`。

### 4) 新增校准器（calibrator）

```python
# src/acs/model/platt.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from acs.registry import REGISTRY

@REGISTRY.register("calibrator", "platt")
def platt(p: pd.Series, y: pd.Series, params: dict) -> pd.Series:
    # Platt scaling: 用逻辑回归将分数映射为校准概率
    # 训练期：用历史（或配置指定窗口）拟合；这里简化为“在全样本重拟合”
    # 注意：真实生产建议滚动拟合，避免信息泄露
    m = LogisticRegression()
    x = p.values.reshape(-1,1)
    m.fit(x[~y.isna()], y.dropna().values)
    out = m.predict_proba(x)[:,1]
    return pd.Series(out, index=p.index, name="p_cal")
```

- `discover()` 中 import；YAML 里 `model.calibrator: "platt"`。  
- 也可写 `isotonic`（`sklearn.isotonic.IsotonicRegression`），注册名 `"isotonic"`。

---

## 输出文件说明

- **预测 CSV**：`outputs/<symbol>_<hash>.csv`  
  列示例：
  - `p_sup`：监督分量概率（0~1）
  - `p_final`：融合+校准后概率（0~1）

- **评估产物**（执行 `eval` 后）
  - `outputs/<symbol>_calibration.csv`：按概率分箱后的平均概率/真实命中率/样本数
  - `outputs/<symbol>_calibration.png`：可靠性曲线示意图
  - 评估 JSON（命令行打印）：`auc`, `brier`, `coverage`, 以及对应文件路径

- **扫参产物**（执行 `sweep_blend.py` 后）
  - `outputs/sweep_blend_<symbol>_<timestamp>.csv/.json`：各种 `(invert_sup, w_unsup)` 的指标排行榜
  - `outputs/sweep_calib_<symbol>_<timestamp>.csv/.json`：在最佳融合处的各校准器指标

---

## 评估指标与校准概念

- **AUC**（Area Under ROC Curve）：区分能力。0.5 = 随机，越高越好。  
- **Brier Score**：概率预测的均方误差，越低越好；反映“概率像不像概率”。  
- **Coverage**：有效样本占比（对齐预测和标签后仍保留的比例）。  
- **Calibration（校准）**：让“预测为 0.8 的那些天”实际命中率接近 80%。  
  - 常见方法：`platt`（逻辑回归）/ `isotonic`（单调回归）/ 组合或分段校准等。

> 生产建议：用**滚动/走前**的方式拟合校准器，避免信息泄露（见路线图）。

---

## 环境准备与安装

- **Python**：3.11+
- **依赖**：`pydantic`, `typer`, `rich`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`（画图可选）, `joblib`, 等  
- **开发安装（推荐）**：在项目根目录执行（含 `pyproject.toml` 或 `setup.cfg` 时）
  ```bash
  pip install -e .
  ```
  你将看到类似：
  ```
  Successfully installed acs-refactor-0.1.0 ...
  ```

- **Windows PowerShell 注意**  
  - 所有命令**不要**带 `>>>`；那是 Python 交互提示符。  
  - 在 PowerShell 直接敲 `import pandas as pd` 会报错；这类代码要在 `python` 交互里执行，或写入 `.py` 文件运行。

---

## 快速开始（Quickstart）

```bash
# 0) 安装
pip install -e .

# 1) 跑配置
python -m acs.cli run-cfg configs/example.yaml
# -> 产出 outputs/<symbol>_*.csv

# 2) 描述预测
python -m acs.cli describe outputs/<your_csv>.csv

# 3) 评估（重算标签）
python -m acs.cli eval configs/example.yaml

# 4) 扫描融合&校准
python sweep_blend.py configs/example.yaml
```

---

## 常见问题（Troubleshooting）

- **`No such command '.\configs\example.yaml'`**  
  - 说明你执行了 `python -m acs.cli .\configs\example.yaml`（缺少子命令）。  
  - 正确：`python -m acs.cli run-cfg .\configs\example.yaml`

- **`FileNotFoundError: .\configs\example.yaml`**  
  - 你不在项目根目录或路径写错。  
  - 进入项目根目录再执行，或用绝对路径。

- **`KeyError: 'akshare'`**  
  - 通常是没有调用 `discover()` 完成插件注册。  
  - CLI 已自动 `discover()`；自写脚本也要在跑之前调用一次：  
    ```python
    from acs import discover
    discover()
    ```

- **`TypeError: fetch_daily() missing 1 required positional argument: 'debug_log'`**  
  - legacy 函数签名要求 `debug_log`。  
  - 已通过 `call_legacy()` 适配，如仍报错，请确认 `acs_panel_unified_3.py` 可被导入且函数名一致。

- **`TypeError: '<=' not supported between Timestamp and NoneType`**  
  - 结束日期 `end` 为空且逻辑未兜底。  
  - 在 YAML 里显式填写 `data.end`，或完善分段逻辑对 None 的处理。

- **`RuntimeError: 区间 ... 返回空`（akshare）**  
  - 数据源返回空（停牌/节假日/网络）。  
  - 检查网络、日期区间与 symbol，或加入更稳健的重试/备用源。

- **`ValueError: Input X contains NaN`（LogisticRegression）**  
  - 模型/校准不接受缺失值。  
  - 在上游补齐（`fillna` / 插值 / 删除），或换能吃 NaN 的模型（如 HGB）。目前流水线已对关键列做了处理。

- **`TypeError: Object of type Timestamp is not JSON serializable`**  
  - 在 `describe` 时，我们已将索引转字符串。若你自写序列化逻辑，请统一转 `str`.

- **PowerShell 中看到 `>>>` 和 `SyntaxError`**  
  - 你把交互式 Python 提示符当成命令复制了。命令行里不要 `>>>`。

---

## 路线图（Roadmap）

**A. 近期（1 周）**
- [ ] 注册纯 `platt` 与 `isotonic` 校准器，横评与 `legacy_platt_isotonic`。
- [ ] 扩大 `w_unsup` 扫描范围至 0~1，步长 0.05（或引入贝叶斯优化）。
- [ ] 多标的批跑与并行，输出横截面稳健性报告。

**B. 近中期（1–2 周）**
- [ ] **滚动/走前评估（Walk-Forward）**：按窗口滚动拟合/调参/校准，避免信息泄露，输出 AUC/Brier 的时间序列。
- [ ] **回测闭环**：把 `p_final` 变为交易规则（阈值/分位/多空对冲），计算收益/回撤/换手/成本；生成日报/周报。
- [ ] **实验追踪**：每次运行的配置、哈希、数据窗、指标写入 metadata（json sidecar），保证 100% 可复现。

**C. 中期（1–2 个月）**
- [ ] 数据层加固：异常/缺失兜底、缓存/增量更新、交易日历与时区统一。
- [ ] 模型库扩展：HGB/XGBoost、序列模型、stacking；配套校准与漂移监控。
- [ ] 上生产：定时任务、产出 signals.csv、报警/监控面板。

---

## 贡献指南（Contribution）

- **代码风格**：PEP8 + 类型注解（mypy 友好），pandas 操作尽量 vectorized。  
- **模块职责**：单一职责、尽量纯函数；副作用（I/O/日志）集中在边界层。  
- **注册习惯**：所有新插件都使用 `@REGISTRY.register(kind, name)`；在 `discover()` 中显式 import。  
- **测试**：建议为每个插件提供最小单测（输入一小段 DataFrame，断言输出 shape/列名/非空比例）。  
- **提交信息**：`feat: ...` / `fix: ...` / `refactor: ...` / `docs: ...` / `chore: ...`

---

## 版本与变更记录（Changelog stub）

- **0.1.0**
  - 初版：CLI（三命令）、插件化架构、legacy 兼容、示例特征/标签/融合/校准、评估与扫参脚本、产物落盘。

---

## 许可证

> TODO：选择并填入（如 MIT / Apache-2.0）。在仓库根目录放置 `LICENSE` 文件。

---

## 附：典型开发小抄

- 打印所有已注册的插件名：
  ```python
  from acs import discover
  from acs.registry import REGISTRY
  discover()
  print(REGISTRY.names("fetcher"))
  print(REGISTRY.names("feature"))
  print(REGISTRY.names("label"))
  print(REGISTRY.names("blend"))
  print(REGISTRY.names("calibrator"))
  ```

- 在脚本中“就地跑一次”：
  ```python
  from acs import discover
  from acs.config import load_cfg
  from acs.pipeline import run
  from acs.eval import evaluate

  discover()
  cfg = load_cfg("configs/example.yaml")
  csv_path = run(cfg)
  metrics, _ = evaluate("configs/example.yaml", csv_path=csv_path)
  print(metrics)
  ```

---

**建议把本文保存为** `README.md`（Markdown 更适合 GitHub 渲染与链接跳转；txt 也可以但可读性逊色）。  
若需要更详尽开发手册，可在 `docs/` 下拆分子文档（数据层/特征库/评估与回测/生产部署等）。
