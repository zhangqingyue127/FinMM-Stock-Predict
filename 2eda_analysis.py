import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# 0. 路径与参数
# =========================================================
BASE_DIR = Path("./data_processed")
EDA_DIR = Path("./eda_outputs_purged")
FIG_DIR = EDA_DIR / "figures"
TABLE_DIR = EDA_DIR / "tables"
REPORT_DIR = EDA_DIR / "reports"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = BASE_DIR / "sample_master_purged.csv"
DAILY_FILE = BASE_DIR / "daily_clean.csv"
SUSP_FILE = BASE_DIR / "suspension_clean.csv"

LOOKBACK = 60
HORIZON = 7
TOP_N_STOCKS = 20
RANDOM_SEED = 42


# =========================================================
# 1. 通用函数
# =========================================================
def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "gbk", "gb18030"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except Exception:
            continue
    raise ValueError(f"无法读取文件: {path}")


def standardize_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.fillna("").str.zfill(6)
    return s


def build_suspension_ranges(susp_df: pd.DataFrame) -> dict:
    ranges = {}
    if susp_df.empty:
        return ranges

    tmp = susp_df.copy()
    tmp = tmp[tmp["Suspdate"].notna()].copy()
    far_future = pd.Timestamp("2099-12-31")
    tmp["Resmdate_filled"] = tmp["Resmdate"].fillna(far_future)

    for stk, g in tmp.groupby("Stkcd"):
        ranges[stk] = list(zip(g["Suspdate"].tolist(), g["Resmdate_filled"].tolist()))
    return ranges


def interval_overlap(start_date: pd.Timestamp, end_date: pd.Timestamp, intervals: list) -> bool:
    for s, e in intervals:
        if pd.isna(s) or pd.isna(e):
            continue
        if not (end_date < s or start_date > e):
            return True
    return False


def save_text(lines, path: Path):
    with open(path, "w", encoding="utf-8-sig") as f:
        if isinstance(lines, list):
            f.write("\n".join(lines))
        else:
            f.write(str(lines))


def add_value_labels(ax, fmt="{:.0f}", rotation=0, fontsize=9):
    for p in ax.patches:
        h = p.get_height()
        if pd.isna(h):
            continue
        ax.annotate(
            fmt.format(h),
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=fontsize,
            xytext=(0, 3),
            textcoords="offset points"
        )


def safe_close():
    plt.tight_layout()
    plt.close()


# =========================================================
# 2. 数据读取与预处理
# =========================================================
def load_data():
    sample_df = read_csv_auto(SAMPLE_FILE)
    daily_df = read_csv_auto(DAILY_FILE)
    susp_df = read_csv_auto(SUSP_FILE) if SUSP_FILE.exists() else pd.DataFrame()

    sample_df["stock_id"] = standardize_code(sample_df["stock_id"])
    sample_df["end_date"] = pd.to_datetime(sample_df["end_date"], errors="coerce")
    sample_df["label"] = pd.to_numeric(sample_df["label"], errors="coerce")
    sample_df["future_return_7"] = pd.to_numeric(sample_df["future_return_7"], errors="coerce")

    daily_df["Stkcd"] = standardize_code(daily_df["Stkcd"])
    daily_df["Trddt"] = pd.to_datetime(daily_df["Trddt"], errors="coerce")
    daily_df["Adjprcnd"] = pd.to_numeric(daily_df["Adjprcnd"], errors="coerce")
    daily_df = daily_df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)
    daily_df["row_idx"] = daily_df.groupby("Stkcd").cumcount()

    if not susp_df.empty:
        susp_df["Stkcd"] = standardize_code(susp_df["Stkcd"])
        susp_df["Suspdate"] = pd.to_datetime(susp_df["Suspdate"], errors="coerce")
        susp_df["Resmdate"] = pd.to_datetime(susp_df["Resmdate"], errors="coerce")
        susp_df["Annctime"] = pd.to_datetime(susp_df["Annctime"], errors="coerce")

    sample_df = sample_df.dropna(subset=["stock_id", "end_date"]).copy()
    sample_df = sample_df.sort_values(["stock_id", "end_date"]).reset_index(drop=True)

    mapper = daily_df[["Stkcd", "Trddt", "row_idx"]].rename(
        columns={"Stkcd": "stock_id", "Trddt": "end_date"}
    )
    sample_df = sample_df.merge(mapper, on=["stock_id", "end_date"], how="left")

    return sample_df, daily_df, susp_df


# =========================================================
# 3. 类别分布 EDA
# =========================================================
def eda_label_distribution(sample_df: pd.DataFrame):
    label_count = sample_df["label"].value_counts(dropna=False).sort_index()
    label_ratio = sample_df["label"].value_counts(normalize=True, dropna=False).sort_index()

    label_table = pd.DataFrame({
        "count": label_count,
        "ratio": label_ratio
    })
    label_table.to_csv(TABLE_DIR / "label_distribution.csv", encoding="utf-8-sig")

    split_label = sample_df.pivot_table(
        index="split",
        columns="label",
        values="sample_id",
        aggfunc="count",
        fill_value=0
    )
    split_label["total"] = split_label.sum(axis=1)
    for col in split_label.columns:
        if col != "total":
            split_label[f"ratio_label_{col}"] = split_label[col] / split_label["total"]
    split_label.to_csv(TABLE_DIR / "split_label_distribution.csv", encoding="utf-8-sig")

    plt.figure(figsize=(6, 4))
    ax = label_count.plot(kind="bar")
    ax.set_title("Label Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Sample Count")
    add_value_labels(ax)
    plt.savefig(FIG_DIR / "label_distribution.png", dpi=200, bbox_inches="tight")
    safe_close()

    report_lines = [
        "=== 类别分布 EDA ===",
        f"总样本数: {len(sample_df)}",
        f"标签0样本数: {int(label_count.get(0, 0))}",
        f"标签1样本数: {int(label_count.get(1, 0))}",
        f"标签0占比: {label_ratio.get(0, 0):.4f}",
        f"标签1占比: {label_ratio.get(1, 0):.4f}"
    ]
    save_text(report_lines, REPORT_DIR / "label_distribution_report.txt")


# =========================================================
# 4. 每只股票样本数分布 EDA
# =========================================================
def eda_stock_sample_distribution(sample_df: pd.DataFrame):
    stock_count = sample_df.groupby("stock_id").size().sort_values(ascending=False)
    stock_count.to_csv(TABLE_DIR / "stock_sample_count.csv", encoding="utf-8-sig")

    desc = stock_count.describe()
    concentration_top1 = stock_count.head(1).sum() / stock_count.sum()
    concentration_top5 = stock_count.head(5).sum() / stock_count.sum()
    concentration_top10 = stock_count.head(10).sum() / stock_count.sum()
    concentration_top20 = stock_count.head(20).sum() / stock_count.sum()

    plt.figure(figsize=(8, 5))
    plt.hist(stock_count.values, bins=30)
    plt.title("Distribution of Sample Count per Stock")
    plt.xlabel("Sample Count per Stock")
    plt.ylabel("Number of Stocks")
    plt.savefig(FIG_DIR / "stock_sample_count_hist.png", dpi=200, bbox_inches="tight")
    safe_close()

    plt.figure(figsize=(12, 5))
    ax = stock_count.head(TOP_N_STOCKS).plot(kind="bar")
    ax.set_title(f"Top {TOP_N_STOCKS} Stocks by Sample Count")
    ax.set_xlabel("Stock ID")
    ax.set_ylabel("Sample Count")
    add_value_labels(ax, fontsize=8)
    plt.savefig(FIG_DIR / "top_stock_sample_count.png", dpi=200, bbox_inches="tight")
    safe_close()

    report_lines = [
        "=== 股票样本数分布 EDA ===",
        f"股票数: {stock_count.shape[0]}",
        f"平均每只股票样本数: {desc['mean']:.2f}",
        f"中位数: {desc['50%']:.2f}",
        f"最小值: {desc['min']:.0f}",
        f"最大值: {desc['max']:.0f}",
        f"Top1 占全部样本比例: {concentration_top1:.4f}",
        f"Top5 占全部样本比例: {concentration_top5:.4f}",
        f"Top10 占全部样本比例: {concentration_top10:.4f}",
        f"Top20 占全部样本比例: {concentration_top20:.4f}"
    ]
    save_text(report_lines, REPORT_DIR / "stock_sample_distribution_report.txt")


# =========================================================
# 5. 时间分布 / 市场阶段偏差 EDA
# =========================================================
def eda_time_distribution(sample_df: pd.DataFrame):
    tmp = sample_df.copy()
    tmp["year"] = tmp["end_date"].dt.year
    tmp["month"] = tmp["end_date"].dt.to_period("M").astype(str)

    month_count = tmp.groupby("month").size()
    month_label_rate = tmp.groupby("month")["label"].mean()
    year_count = tmp.groupby("year").size()
    year_label_rate = tmp.groupby("year")["label"].mean()

    month_count.to_csv(TABLE_DIR / "month_sample_count.csv", encoding="utf-8-sig")
    month_label_rate.to_csv(TABLE_DIR / "month_positive_rate.csv", encoding="utf-8-sig")
    year_count.to_csv(TABLE_DIR / "year_sample_count.csv", encoding="utf-8-sig")
    year_label_rate.to_csv(TABLE_DIR / "year_positive_rate.csv", encoding="utf-8-sig")

    split_month = tmp.pivot_table(
        index="month",
        columns="split",
        values="sample_id",
        aggfunc="count",
        fill_value=0
    )
    split_month.to_csv(TABLE_DIR / "month_split_count.csv", encoding="utf-8-sig")

    plt.figure(figsize=(14, 5))
    ax = month_count.plot(kind="bar")
    ax.set_title("Monthly Sample Count")
    ax.set_xlabel("Month")
    ax.set_ylabel("Sample Count")
    plt.xticks(rotation=60)
    plt.savefig(FIG_DIR / "monthly_sample_count.png", dpi=200, bbox_inches="tight")
    safe_close()

    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(month_count.index, month_count.values, marker="o")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Sample Count")
    ax1.set_title("Monthly Sample Count and Positive Rate")
    ax1.tick_params(axis="x", rotation=60)

    ax2 = ax1.twinx()
    ax2.plot(month_label_rate.index, month_label_rate.values, marker="s")
    ax2.set_ylabel("Positive Rate")

    plt.savefig(FIG_DIR / "monthly_count_positive_rate.png", dpi=200, bbox_inches="tight")
    safe_close()

    plt.figure(figsize=(8, 4))
    ax = year_count.plot(kind="bar")
    ax.set_title("Yearly Sample Count")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sample Count")
    add_value_labels(ax)
    plt.savefig(FIG_DIR / "yearly_sample_count.png", dpi=200, bbox_inches="tight")
    safe_close()

    report_lines = [
        "=== 时间分布 / 市场阶段偏差 EDA ===",
        f"时间范围: {tmp['end_date'].min().date()} ~ {tmp['end_date'].max().date()}",
        "各年份样本数:",
        year_count.to_string(),
        "",
        "各年份标签1比例:",
        year_label_rate.to_string()
    ]
    save_text(report_lines, REPORT_DIR / "time_distribution_report.txt")


# =========================================================
# 6. 停牌影响 EDA
# =========================================================
def reconstruct_candidate_windows(daily_df: pd.DataFrame, susp_df: pd.DataFrame) -> pd.DataFrame:
    suspension_map = build_suspension_ranges(susp_df)
    rows = []

    for stk, g in daily_df.groupby("Stkcd", sort=True):
        g = g.sort_values("Trddt").reset_index(drop=True)
        n = len(g)
        if n < LOOKBACK + HORIZON:
            continue

        adj = g["Adjprcnd"].to_numpy()
        stk_intervals = suspension_map.get(stk, [])

        for i in range(LOOKBACK - 1, n - HORIZON):
            hist_start_date = g.loc[i - LOOKBACK + 1, "Trddt"]
            end_date = g.loc[i, "Trddt"]
            future_end_date = g.loc[i + HORIZON, "Trddt"]
            row_idx = int(g.loc[i, "row_idx"])

            suspension_hit = interval_overlap(hist_start_date, future_end_date, stk_intervals) if stk_intervals else False
            p_t = adj[i]
            p_t7 = adj[i + HORIZON]
            price_invalid = pd.isna(p_t) or pd.isna(p_t7) or (p_t <= 0)

            rows.append({
                "stock_id": stk,
                "end_date": end_date,
                "row_idx": row_idx,
                "hist_start_date": hist_start_date,
                "future_end_date": future_end_date,
                "suspension_hit": int(suspension_hit),
                "price_invalid": int(price_invalid)
            })

    candidate_df = pd.DataFrame(rows)
    return candidate_df


def eda_suspension_impact(sample_df: pd.DataFrame, daily_df: pd.DataFrame, susp_df: pd.DataFrame):
    if susp_df.empty:
        save_text("停牌文件不存在，跳过停牌影响分析。", REPORT_DIR / "suspension_impact_report.txt")
        return

    candidate_df = reconstruct_candidate_windows(daily_df, susp_df)
    candidate_df.to_csv(TABLE_DIR / "candidate_window_flags.csv", index=False, encoding="utf-8-sig")

    stock_susp_event_count = susp_df.groupby("Stkcd").size().sort_values(ascending=False)
    stock_susp_event_count.to_csv(TABLE_DIR / "stock_suspension_event_count.csv", encoding="utf-8-sig")

    tmp = susp_df.copy()
    tmp["approx_suspend_days"] = (tmp["Resmdate"].fillna(tmp["Suspdate"]) - tmp["Suspdate"]).dt.days.clip(lower=0)
    stock_susp_days = tmp.groupby("Stkcd")["approx_suspend_days"].sum().sort_values(ascending=False)
    stock_susp_days.to_csv(TABLE_DIR / "stock_approx_suspension_days.csv", encoding="utf-8-sig")

    dropped_by_susp = candidate_df.groupby("stock_id")["suspension_hit"].sum().sort_values(ascending=False)
    dropped_by_susp.to_csv(TABLE_DIR / "stock_dropped_by_suspension.csv", encoding="utf-8-sig")

    total_candidates = len(candidate_df)
    total_drop_susp = int(candidate_df["suspension_hit"].sum())
    total_drop_price = int(candidate_df["price_invalid"].sum())
    final_sample_count = len(sample_df)

    plt.figure(figsize=(12, 5))
    ax = stock_susp_event_count.head(TOP_N_STOCKS).plot(kind="bar")
    ax.set_title(f"Top {TOP_N_STOCKS} Stocks by Suspension Events")
    ax.set_xlabel("Stock ID")
    ax.set_ylabel("Suspension Event Count")
    add_value_labels(ax, fontsize=8)
    plt.savefig(FIG_DIR / "top_stock_suspension_events.png", dpi=200, bbox_inches="tight")
    safe_close()

    plt.figure(figsize=(12, 5))
    ax = dropped_by_susp.head(TOP_N_STOCKS).plot(kind="bar")
    ax.set_title(f"Top {TOP_N_STOCKS} Stocks by Candidate Windows Dropped due to Suspension")
    ax.set_xlabel("Stock ID")
    ax.set_ylabel("Dropped Window Count")
    add_value_labels(ax, fontsize=8)
    plt.savefig(FIG_DIR / "top_stock_dropped_by_suspension.png", dpi=200, bbox_inches="tight")
    safe_close()

    report_lines = [
        "=== 停牌影响 EDA ===",
        f"候选窗口总数: {total_candidates}",
        f"因停牌区间重叠被过滤窗口数: {total_drop_susp}",
        f"因价格异常被过滤窗口数: {total_drop_price}",
        f"最终进入 sample_master 的样本数: {final_sample_count}",
        f"停牌过滤比例(候选窗口口径): {total_drop_susp / total_candidates if total_candidates > 0 else 0:.4f}",
        f"价格异常过滤比例(候选窗口口径): {total_drop_price / total_candidates if total_candidates > 0 else 0:.4f}",
        f"有停牌记录的股票数: {stock_susp_event_count.shape[0]}",
        "",
        "停牌事件最多的前10只股票:",
        stock_susp_event_count.head(10).to_string(),
        "",
        "因停牌导致候选窗口过滤最多的前10只股票:",
        dropped_by_susp.head(10).to_string()
    ]
    save_text(report_lines, REPORT_DIR / "suspension_impact_report.txt")


# =========================================================
# 7. 窗口重叠程度 EDA
# =========================================================
def eda_window_overlap(sample_df: pd.DataFrame):
    tmp = sample_df.dropna(subset=["row_idx"]).copy()
    tmp = tmp.sort_values(["stock_id", "row_idx"]).reset_index(drop=True)

    overlap_rows = []

    for stk, g in tmp.groupby("stock_id"):
        g = g.sort_values("row_idx").reset_index(drop=True)
        if len(g) < 2:
            continue

        g["prev_row_idx"] = g["row_idx"].shift(1)
        g["prev_split"] = g["split"].shift(1)
        g["gap"] = g["row_idx"] - g["prev_row_idx"]
        g["overlap_days"] = (LOOKBACK - g["gap"]).clip(lower=0)
        g["overlap_ratio"] = g["overlap_days"] / LOOKBACK
        g["transition"] = g["prev_split"].astype(str) + "->" + g["split"].astype(str)

        overlap_rows.append(g)

    if len(overlap_rows) == 0:
        save_text("样本过少，无法计算窗口重叠。", REPORT_DIR / "window_overlap_report.txt")
        return

    overlap_df = pd.concat(overlap_rows, ignore_index=True)
    overlap_df = overlap_df[overlap_df["prev_row_idx"].notna()].copy()
    overlap_df.to_csv(TABLE_DIR / "sample_window_overlap_detail.csv", index=False, encoding="utf-8-sig")

    overall_mean_overlap = overlap_df["overlap_ratio"].mean()
    overall_median_overlap = overlap_df["overlap_ratio"].median()
    high_overlap_ratio = (overlap_df["overlap_ratio"] >= 0.8).mean()
    exact_adjacent_ratio = (overlap_df["gap"] == 1).mean()

    split_overlap = overlap_df.groupby("split")["overlap_ratio"].agg(["count", "mean", "median", "max", "min"])
    split_overlap.to_csv(TABLE_DIR / "split_window_overlap_summary.csv", encoding="utf-8-sig")

    plt.figure(figsize=(8, 5))
    plt.hist(overlap_df["overlap_ratio"].dropna().values, bins=30)
    plt.title("Distribution of Window Overlap Ratio")
    plt.xlabel("Overlap Ratio")
    plt.ylabel("Frequency")
    plt.savefig(FIG_DIR / "window_overlap_ratio_hist.png", dpi=200, bbox_inches="tight")
    safe_close()

    cross_split_df = overlap_df[overlap_df["prev_split"] != overlap_df["split"]].copy()
    cross_split_df.to_csv(TABLE_DIR / "cross_split_overlap_detail.csv", index=False, encoding="utf-8-sig")

    if not cross_split_df.empty:
        transition_count = cross_split_df["transition"].value_counts()
        transition_count.to_csv(TABLE_DIR / "cross_split_transition_count.csv", encoding="utf-8-sig")

        plt.figure(figsize=(8, 4))
        ax = transition_count.plot(kind="bar")
        ax.set_title("Cross-Split Transition Count")
        ax.set_xlabel("Transition")
        ax.set_ylabel("Count")
        add_value_labels(ax)
        plt.savefig(FIG_DIR / "cross_split_transition_count.png", dpi=200, bbox_inches="tight")
        safe_close()

        plt.figure(figsize=(8, 5))
        plt.hist(cross_split_df["overlap_ratio"].dropna().values, bins=20)
        plt.title("Cross-Split Overlap Ratio Distribution")
        plt.xlabel("Overlap Ratio")
        plt.ylabel("Frequency")
        plt.savefig(FIG_DIR / "cross_split_overlap_ratio_hist.png", dpi=200, bbox_inches="tight")
        safe_close()

        cross_mean = cross_split_df["overlap_ratio"].mean()
        cross_max = cross_split_df["overlap_ratio"].max()
        cross_high = (cross_split_df["overlap_ratio"] >= 0.8).mean()
    else:
        cross_mean = np.nan
        cross_max = np.nan
        cross_high = np.nan

    report_lines = [
        "=== 窗口重叠程度 EDA ===",
        f"总连续样本对数: {len(overlap_df)}",
        f"平均重叠比例: {overall_mean_overlap:.4f}",
        f"中位数重叠比例: {overall_median_overlap:.4f}",
        f"重叠比例 >= 0.8 的样本对占比: {high_overlap_ratio:.4f}",
        f"相邻样本 gap=1 的占比: {exact_adjacent_ratio:.4f}",
        "",
        "=== 跨 split 图像泄漏风险检查 ===",
        f"跨 split 连续样本对数: {len(cross_split_df)}",
        f"跨 split 平均重叠比例: {cross_mean if pd.notna(cross_mean) else 'NA'}",
        f"跨 split 最大重叠比例: {cross_max if pd.notna(cross_max) else 'NA'}",
        f"跨 split 中重叠比例 >= 0.8 的占比: {cross_high if pd.notna(cross_high) else 'NA'}",
        "",
        "说明：如果跨 split 的重叠比例很高，虽然不等于标签泄漏，但说明 train/val/test 图像可能非常相似，需要在论文中明确说明。"
    ]
    save_text(report_lines, REPORT_DIR / "window_overlap_report.txt")


# =========================================================
# 8. split 纯时间切分检查
# =========================================================
def eda_split_integrity(sample_df: pd.DataFrame):
    tmp = sample_df.dropna(subset=["row_idx"]).copy()

    same_key_multi_split = tmp.groupby(["stock_id", "end_date"])["split"].nunique()
    same_key_multi_split_bad = same_key_multi_split[same_key_multi_split > 1]

    order_issues = []
    for stk, g in tmp.groupby("stock_id"):
        g = g.sort_values("row_idx")
        split_order = g["split"].tolist()

        train_idx = g.loc[g["split"] == "train", "row_idx"]
        val_idx = g.loc[g["split"] == "val", "row_idx"]
        test_idx = g.loc[g["split"] == "test", "row_idx"]

        if len(train_idx) > 0 and len(val_idx) > 0:
            if train_idx.max() >= val_idx.min():
                order_issues.append((stk, "train_val"))
        if len(val_idx) > 0 and len(test_idx) > 0:
            if val_idx.max() >= test_idx.min():
                order_issues.append((stk, "val_test"))
        if len(train_idx) > 0 and len(test_idx) > 0 and len(val_idx) == 0:
            if train_idx.max() >= test_idx.min():
                order_issues.append((stk, "train_test"))

    split_count = tmp["split"].value_counts()
    split_count.to_csv(TABLE_DIR / "split_sample_count.csv", encoding="utf-8-sig")

    report_lines = [
        "=== split 纯时间切分检查 ===",
        f"sample_id 去重后样本数: {tmp['sample_id'].nunique() if 'sample_id' in tmp.columns else len(tmp)}",
        f"(stock_id, end_date) 对应多个 split 的异常个数: {len(same_key_multi_split_bad)}",
        f"按股票检查时间顺序异常个数: {len(order_issues)}",
        "",
        "各 split 样本数:",
        split_count.to_string()
    ]

    if len(order_issues) > 0:
        issue_df = pd.DataFrame(order_issues, columns=["stock_id", "issue_type"])
        issue_df.to_csv(TABLE_DIR / "split_order_issues.csv", index=False, encoding="utf-8-sig")
        report_lines.append("")
        report_lines.append("存在 split 顺序异常，详情见 split_order_issues.csv")
    else:
        report_lines.append("")
        report_lines.append("未发现 train/val/test 混在一起的问题。")

    save_text(report_lines, REPORT_DIR / "split_integrity_report.txt")


# =========================================================
# 9. 图像样例图 + 典型样本图
# =========================================================
def pick_typical_samples(sample_df: pd.DataFrame):
    tmp = sample_df.dropna(subset=["future_return_7", "row_idx"]).copy()
    if tmp.empty:
        return pd.DataFrame()

    pos = tmp[tmp["label"] == 1].sort_values("future_return_7")
    neg = tmp[tmp["label"] == 0].sort_values("future_return_7")

    selected = []

    if len(pos) > 0:
        selected.append(pos.iloc[-1])
        selected.append(pos.iloc[len(pos) // 2])

    if len(neg) > 0:
        selected.append(neg.iloc[0])
        selected.append(neg.iloc[len(neg) // 2])

    out = pd.DataFrame(selected).drop_duplicates(subset=["stock_id", "end_date"]).reset_index(drop=True)
    return out


def get_window_data(daily_df: pd.DataFrame, stock_id: str, end_row_idx: int):
    g = daily_df[daily_df["Stkcd"] == stock_id].sort_values("row_idx").reset_index(drop=True)
    hist_start = end_row_idx - LOOKBACK + 1
    future_end = end_row_idx + HORIZON

    if hist_start < 0 or future_end >= len(g):
        return None

    hist_df = g.iloc[hist_start:end_row_idx + 1].copy()
    future_df = g.iloc[end_row_idx + 1:future_end + 1].copy()
    return hist_df, future_df


def eda_sample_plots(sample_df: pd.DataFrame, daily_df: pd.DataFrame):
    selected = pick_typical_samples(sample_df)
    if selected.empty:
        save_text("无可用样本，无法生成图像样例图和典型样本图。", REPORT_DIR / "sample_plot_report.txt")
        return

    selected.to_csv(TABLE_DIR / "selected_typical_samples.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, selected.iterrows()):
        result = get_window_data(daily_df, row["stock_id"], int(row["row_idx"]))
        if result is None:
            ax.set_axis_off()
            continue

        hist_df, future_df = result

        hist_price = hist_df["Adjprcnd"].astype(float).values
        future_price = future_df["Adjprcnd"].astype(float).values

        base_price = hist_price[0]
        hist_norm = hist_price / base_price
        future_norm = future_price / base_price

        x_hist = np.arange(len(hist_norm))
        x_future = np.arange(len(hist_norm), len(hist_norm) + len(future_norm))

        ax.plot(x_hist, hist_norm, linewidth=1.5)
        ax.plot(x_future, future_norm, linestyle="--", linewidth=1.5)
        ax.axvline(len(hist_norm) - 1, linestyle=":")
        ax.set_title(
            f"{row['stock_id']} | {pd.to_datetime(row['end_date']).date()} | "
            f"label={int(row['label'])} | ret7={row['future_return_7']:.4f}"
        )
        ax.set_xlabel("Trading Day Index")
        ax.set_ylabel("Normalized Adj Price")

    for i in range(len(selected), 4):
        axes[i].set_axis_off()

    plt.suptitle("Typical Sample Windows")
    plt.savefig(FIG_DIR / "typical_sample_windows.png", dpi=200, bbox_inches="tight")
    safe_close()

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, selected.iterrows()):
        result = get_window_data(daily_df, row["stock_id"], int(row["row_idx"]))
        if result is None:
            ax.set_axis_off()
            continue

        hist_df, _ = result
        hist_price = hist_df["Adjprcnd"].astype(float).values
        hist_norm = hist_price / hist_price[0]

        ax.plot(np.arange(len(hist_norm)), hist_norm, linewidth=1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"{row['stock_id']} | {pd.to_datetime(row['end_date']).date()} | y={int(row['label'])}",
            fontsize=9
        )

    for i in range(len(selected), 4):
        axes[i].set_axis_off()

    plt.suptitle("Image-like Historical Window Examples")
    plt.savefig(FIG_DIR / "image_like_examples.png", dpi=200, bbox_inches="tight")
    safe_close()

    report_lines = [
        "=== 图像样例图 / 典型样本图 ===",
        "已输出:",
        "1. typical_sample_windows.png：历史60日窗口 + 未来7日走势",
        "2. image_like_examples.png：只保留历史窗口，模拟后续图像输入的可视化形式",
        "",
        "样本明细见 selected_typical_samples.csv"
    ]
    save_text(report_lines, REPORT_DIR / "sample_plot_report.txt")


# =========================================================
# 10. 汇总总报告
# =========================================================
def build_master_summary(sample_df: pd.DataFrame, daily_df: pd.DataFrame, susp_df: pd.DataFrame):
    lines = [
        "EDA 总览",
        "======================================",
        f"sample_master 样本数: {len(sample_df)}",
        f"涉及股票数: {sample_df['stock_id'].nunique()}",
        f"daily_clean 行数: {len(daily_df)}",
        f"daily_clean 股票数: {daily_df['Stkcd'].nunique()}",
        f"suspension_clean 行数: {len(susp_df) if not susp_df.empty else 0}",
        f"样本时间范围: {sample_df['end_date'].min().date()} ~ {sample_df['end_date'].max().date()}",
        "",
        "输出目录说明：",
        f"图: {FIG_DIR.resolve()}",
        f"表: {TABLE_DIR.resolve()}",
        f"报告: {REPORT_DIR.resolve()}",
        "",
        "重点图文件：",
        "- label_distribution.png",
        "- top_stock_sample_count.png",
        "- monthly_sample_count.png",
        "- monthly_count_positive_rate.png",
        "- top_stock_suspension_events.png",
        "- top_stock_dropped_by_suspension.png",
        "- window_overlap_ratio_hist.png",
        "- cross_split_overlap_ratio_hist.png",
        "- typical_sample_windows.png",
        "- image_like_examples.png"
    ]
    save_text(lines, REPORT_DIR / "EDA_master_summary.txt")


# =========================================================
# 11. 主程序
# =========================================================
if __name__ == "__main__":
    print("========== 开始 EDA ==========")

    sample_df, daily_df, susp_df = load_data()

    print("[1/7] 类别分布分析")
    eda_label_distribution(sample_df)

    print("[2/7] 股票样本数分布分析")
    eda_stock_sample_distribution(sample_df)

    print("[3/7] 时间分布 / 市场阶段偏差分析")
    eda_time_distribution(sample_df)

    print("[4/7] 停牌影响分析")
    eda_suspension_impact(sample_df, daily_df, susp_df)

    print("[5/7] 窗口重叠 / 图像泄漏风险分析")
    eda_window_overlap(sample_df)

    print("[6/7] split 完整性检查")
    eda_split_integrity(sample_df)

    print("[7/7] 图像样例图 / 典型样本图")
    eda_sample_plots(sample_df, daily_df)

    build_master_summary(sample_df, daily_df, susp_df)

    print("========== EDA 完成 ==========")
    print(f"输出目录: {EDA_DIR.resolve()}")