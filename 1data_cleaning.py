import pandas as pd
import numpy as np
from pathlib import Path

# =========================================================
# 0. 路径与参数配置
# =========================================================
BASE_DIR = Path("./data_raw")
OUT_DIR = Path("./data_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_FILE = BASE_DIR / "TRD_Dalyr.csv"
SUSP_FILE = BASE_DIR / "suspension.csv"
INDEX_FILE = BASE_DIR / "index_basic.csv"  # 可选，不参与样本构造

# 样本构造参数
LOOKBACK = 60
HORIZON = 7
PURGE_GAP = LOOKBACK + HORIZON  

# 时间切分
TRAIN_END = pd.Timestamp("2024-12-31")
VAL_END = pd.Timestamp("2025-09-30")  # test 自动为剩余部分

# A股市场类型：1上证A, 4深证A, 16创业板, 32科创板, 64北证A
A_SHARE_MARKET_TYPES = {1, 4, 16, 32, 64}

# 交易状态：1=正常交易
NORMAL_TRADING_STATUS = 1


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
    raise ValueError(f"无法读取文件，请检查编码: {path}")


def standardize_stock_code(series: pd.Series) -> pd.Series:
    """
    统一股票代码为 6 位数字字符串
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.fillna("").str.zfill(6)
    s = s.where(s.str.len() == 6, "")
    return s


def standardize_index_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.fillna("").str.zfill(6)
    return s


def save_missing_report(df: pd.DataFrame, filename: str):
    report = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_ratio": df.isna().mean()
    }).sort_values("missing_count", ascending=False)
    report.to_csv(OUT_DIR / filename, encoding="utf-8-sig")


# =========================================================
# 2. 基础清洗
# =========================================================
def clean_daily_data(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)

    required_cols = [
        "Stkcd", "Trddt", "Opnprc", "Hiprc", "Loprc", "Clsprc",
        "Dnshrtrd", "Dnvaltrd", "Dretnd", "Adjprcnd",
        "Markettype", "Trdsta", "ChangeRatio"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"日个股文件缺少字段: {missing_cols}")

    # 统一代码和日期
    df["Stkcd"] = standardize_stock_code(df["Stkcd"])
    df["Trddt"] = pd.to_datetime(df["Trddt"], errors="coerce")

    # 数值列
    num_cols = [
        "Opnprc", "Hiprc", "Loprc", "Clsprc",
        "Dnshrtrd", "Dnvaltrd", "Dretnd", "Adjprcnd",
        "Markettype", "Trdsta", "ChangeRatio"
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 去掉空代码/空日期
    df = df[(df["Stkcd"] != "") & (df["Trddt"].notna())].copy()

    print(f"[daily] 原始行数: {len(df)}")
    print(f"[daily] 原始股票数: {df['Stkcd'].nunique()}")

    # 去重
    before = len(df)
    df = df.drop_duplicates(subset=["Stkcd", "Trddt"], keep="first").copy()
    after = len(df)
    print(f"[daily] 去重删除 {before - after} 行")

    # 排序
    df = df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)

    # 缺失报告
    save_missing_report(df, "daily_missing_report.csv")

    # 异常价格检查
    abnormal_mask = (
        (df["Hiprc"] < df["Loprc"]) |
        (df["Opnprc"] < df["Loprc"]) |
        (df["Opnprc"] > df["Hiprc"]) |
        (df["Clsprc"] < df["Loprc"]) |
        (df["Clsprc"] > df["Hiprc"]) |
        (df["Opnprc"] <= 0) |
        (df["Hiprc"] <= 0) |
        (df["Loprc"] <= 0) |
        (df["Clsprc"] <= 0) |
        (df["Adjprcnd"] <= 0)
    )

    abnormal_rows = df.loc[abnormal_mask].copy()
    abnormal_rows.to_csv(
        OUT_DIR / "daily_abnormal_rows.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[daily] 异常价格行数: {len(abnormal_rows)}")

    # 剔除异常价格
    df = df.loc[~abnormal_mask].copy()

    # 保留A股
    before = df["Stkcd"].nunique()
    df = df[df["Markettype"].isin(A_SHARE_MARKET_TYPES)].copy()
    after = df["Stkcd"].nunique()
    print(f"[daily] 过滤非A股后股票数: {after}（变化 {before} -> {after}）")

    # 保留正常交易日（不整股删ST，只删对应非正常交易记录）
    before = df["Stkcd"].nunique()
    df = df[df["Trdsta"] == NORMAL_TRADING_STATUS].copy()
    after = df["Stkcd"].nunique()
    print(f"[daily] 过滤非正常交易日后股票数: {after}（变化 {before} -> {after}）")

    df = df.sort_values(["Stkcd", "Trddt"]).reset_index(drop=True)
    df.to_csv(OUT_DIR / "daily_clean.csv", index=False, encoding="utf-8-sig")

    print(f"[daily] 清洗后保存: {OUT_DIR / 'daily_clean.csv'}")
    print(f"[daily] 清洗后行数: {len(df)}")
    print(f"[daily] 清洗后股票数: {df['Stkcd'].nunique()}")

    return df


def clean_suspension_data(path: Path) -> pd.DataFrame:
    df = read_csv_auto(path)

    required_cols = ["Stkcd", "Stknme", "Annctime", "Type", "Suspdate", "Resmdate", "Timeperd"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"停复牌文件缺少字段: {missing_cols}")

    df["Stkcd"] = standardize_stock_code(df["Stkcd"])
    df["Annctime"] = pd.to_datetime(df["Annctime"], errors="coerce")
    df["Suspdate"] = pd.to_datetime(df["Suspdate"], errors="coerce")
    df["Resmdate"] = pd.to_datetime(df["Resmdate"], errors="coerce")
    df["Timeperd"] = pd.to_numeric(df["Timeperd"], errors="coerce")
    df["Type"] = pd.to_numeric(df["Type"], errors="coerce")

    df = df[df["Stkcd"] != ""].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["Stkcd", "Suspdate", "Resmdate", "Type"], keep="first").copy()
    after = len(df)
    print(f"[susp] 去重删除 {before - after} 行")

    abnormal_mask = (
        df["Suspdate"].notna() &
        df["Resmdate"].notna() &
        (df["Resmdate"] < df["Suspdate"])
    )
    abnormal_rows = df.loc[abnormal_mask].copy()
    abnormal_rows.to_csv(
        OUT_DIR / "suspension_abnormal_rows.csv",
        index=False,
        encoding="utf-8-sig"
    )
    print(f"[susp] 异常日期行数: {len(abnormal_rows)}")

    df = df.loc[~abnormal_mask].copy()
    df = df.sort_values(["Stkcd", "Suspdate", "Resmdate"]).reset_index(drop=True)

    save_missing_report(df, "suspension_missing_report.csv")
    df.to_csv(OUT_DIR / "suspension_clean.csv", index=False, encoding="utf-8-sig")
    print(f"[susp] 清洗后保存: {OUT_DIR / 'suspension_clean.csv'}")

    return df


def clean_index_basic(path: Path) -> pd.DataFrame:
    if not path.exists():
        print("[index] index_basic.csv 不存在，跳过")
        return pd.DataFrame()

    df = read_csv_auto(path)

    required_cols = ["Indexcd", "Idxinfo01", "Idxinfo11", "Idxinfo07", "Idxinfo08", "Idxinfo09"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"指数基本信息文件缺少字段: {missing_cols}")

    df["Indexcd"] = standardize_index_code(df["Indexcd"])
    df["Idxinfo11"] = pd.to_datetime(df["Idxinfo11"], errors="coerce")

    df = df.drop_duplicates(subset=["Indexcd"], keep="first").copy()
    df = df.sort_values(["Indexcd"]).reset_index(drop=True)

    save_missing_report(df, "index_basic_missing_report.csv")
    df.to_csv(OUT_DIR / "index_basic_clean.csv", index=False, encoding="utf-8-sig")
    print(f"[index] 清洗后保存: {OUT_DIR / 'index_basic_clean.csv'}")

    return df


# =========================================================
# 3. 停牌区间映射
# =========================================================
def build_suspension_ranges(susp_df: pd.DataFrame) -> dict:
    """
    构造每只股票的停牌区间列表
    """
    ranges = {}
    if susp_df.empty:
        return ranges

    tmp = susp_df.copy()
    tmp = tmp[tmp["Suspdate"].notna()].copy()

    # Resmdate为空表示长期停牌/退市，填一个远日期
    far_future = pd.Timestamp("2099-12-31")
    tmp["Resmdate_filled"] = tmp["Resmdate"].fillna(far_future)

    for stk, g in tmp.groupby("Stkcd"):
        ranges[stk] = list(zip(g["Suspdate"].tolist(), g["Resmdate_filled"].tolist()))

    return ranges


def interval_overlap(start_date: pd.Timestamp, end_date: pd.Timestamp, intervals: list) -> bool:
    """
    判断 [start_date, end_date] 是否与任一停牌区间重叠
    """
    for s, e in intervals:
        if pd.isna(s) or pd.isna(e):
            continue
        if not (end_date < s or start_date > e):
            return True
    return False

def locate_last_leq_idx(date_series: pd.Series, cutoff: pd.Timestamp):
    arr = date_series.to_numpy(dtype="datetime64[ns]")
    pos = np.where(arr <= np.datetime64(cutoff))[0]
    return int(pos[-1]) if len(pos) > 0 else None


def assign_split_purged(i: int, train_cut_idx, val_cut_idx, gap: int) -> str:
    """
    方式A：固定 val/test 起点，缩短前一个 split
    用“交易日索引”而不是自然日做 purge gap
    """
    if val_cut_idx is not None and i > val_cut_idx:
        return "test"

    if train_cut_idx is None:
        if val_cut_idx is None:
            return "test"
        if i <= val_cut_idx - gap:
            return "val"
        return "gap_val_test"

    if i <= train_cut_idx - gap:
        return "train"

    if i <= train_cut_idx:
        return "gap_train_val"

    if val_cut_idx is None:
        return "test"

    if i <= val_cut_idx - gap:
        return "val"

    if i <= val_cut_idx:
        return "gap_val_test"

    return "test"

# =========================================================
# 4. 样本构造 + 标签计算
# =========================================================
def build_sample_master(daily_df: pd.DataFrame, susp_df: pd.DataFrame) -> pd.DataFrame:
    suspension_map = build_suspension_ranges(susp_df)
    sample_rows = []

    total_candidates = 0
    drop_by_suspend = 0
    drop_by_price = 0
    drop_by_gap_train_val = 0
    drop_by_gap_val_test = 0

    for stk, g in daily_df.groupby("Stkcd", sort=True):
        g = g.sort_values("Trddt").reset_index(drop=True)
        n = len(g)

        if n < LOOKBACK + HORIZON:
            continue

        adj = g["Adjprcnd"].to_numpy()

        # end_date 对应索引 i
        start_i = LOOKBACK - 1
        end_i = n - HORIZON - 1

        stk_intervals = suspension_map.get(stk, [])

        # 关键：按“该股票自己的交易日索引”定位边界
        train_cut_idx = locate_last_leq_idx(g["Trddt"], TRAIN_END)
        val_cut_idx = locate_last_leq_idx(g["Trddt"], VAL_END)

        for i in range(start_i, end_i + 1):
            total_candidates += 1

            hist_start_date = g.loc[i - LOOKBACK + 1, "Trddt"]
            end_date = g.loc[i, "Trddt"]
            future_end_date = g.loc[i + HORIZON, "Trddt"]

            # 停牌影响窗口剔除
            if stk_intervals and interval_overlap(hist_start_date, future_end_date, stk_intervals):
                drop_by_suspend += 1
                continue

            p_t = adj[i]
            p_t7 = adj[i + HORIZON]

            if pd.isna(p_t) or pd.isna(p_t7) or p_t <= 0:
                drop_by_price += 1
                continue

            # 关键：方式A purged split
            split = assign_split_purged(i, train_cut_idx, val_cut_idx, PURGE_GAP)

            if split == "gap_train_val":
                drop_by_gap_train_val += 1
                continue

            if split == "gap_val_test":
                drop_by_gap_val_test += 1
                continue

            future_return_7 = (p_t7 - p_t) / p_t
            label = 1 if future_return_7 > 0 else 0

            sample_id = f"{stk}_{end_date.strftime('%Y-%m-%d')}"

            sample_rows.append({
                "sample_id": sample_id,
                "stock_id": stk,
                "end_date": end_date,
                "label": label,
                "future_return_7": future_return_7,
                "split": split
            })

    sample_df = pd.DataFrame(sample_rows)
    sample_df = sample_df.sort_values(["stock_id", "end_date"]).reset_index(drop=True)

    print(f"[sample] 候选窗口数: {total_candidates}")
    print(f"[sample] 因停牌区间重叠剔除: {drop_by_suspend}")
    print(f"[sample] 因价格/标签异常剔除: {drop_by_price}")
    print(f"[sample] 因 train-val purge gap 剔除: {drop_by_gap_train_val}")
    print(f"[sample] 因 val-test purge gap 剔除: {drop_by_gap_val_test}")
    print(f"[sample] 最终样本数: {len(sample_df)}")

    sample_df.to_csv(OUT_DIR / "sample_master_purged.csv", index=False, encoding="utf-8-sig")
    print(f"[sample] 样本主表已保存: {OUT_DIR / 'sample_master_purged.csv'}")

    if not sample_df.empty:
        sample_df["end_month"] = sample_df["end_date"].dt.to_period("M").astype(str)

        label_dist = sample_df["label"].value_counts(dropna=False).sort_index()
        label_dist.to_csv(OUT_DIR / "label_distribution_purged.csv", encoding="utf-8-sig")

        stock_count = sample_df.groupby("stock_id").size().sort_values(ascending=False)
        stock_count.to_csv(OUT_DIR / "stock_sample_count_purged.csv", encoding="utf-8-sig")

        month_count = sample_df.groupby("end_month").size()
        month_count.to_csv(OUT_DIR / "month_sample_count_purged.csv", encoding="utf-8-sig")

        split_count = sample_df.groupby("split").size()
        split_count.to_csv(OUT_DIR / "split_sample_count_purged.csv", encoding="utf-8-sig")

        purge_report = pd.DataFrame({
            "item": [
                "total_candidates",
                "drop_by_suspend",
                "drop_by_price",
                "drop_by_gap_train_val",
                "drop_by_gap_val_test",
                "final_samples"
            ],
            "count": [
                total_candidates,
                drop_by_suspend,
                drop_by_price,
                drop_by_gap_train_val,
                drop_by_gap_val_test,
                len(sample_df)
            ]
        })
        purge_report.to_csv(OUT_DIR / "purge_summary.csv", index=False, encoding="utf-8-sig")

        print("[sample] purged 标签分布、股票样本数、月度样本数、split统计已保存")

    return sample_df

# =========================================================
# 5. 主程序
# =========================================================
if __name__ == "__main__":
    print("========== 开始数据预处理与样本构造 ==========")

    daily_df = clean_daily_data(DAILY_FILE)
    susp_df = clean_suspension_data(SUSP_FILE)
    _ = clean_index_basic(INDEX_FILE)
    sample_df = build_sample_master(daily_df, susp_df)

    print("\n========== 处理完成 ==========")
    print(f"输出目录: {OUT_DIR.resolve()}")