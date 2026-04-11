import pandas as pd
from pathlib import Path

# =========================
# 1. 文件路径
# =========================
HS300_FILE = Path("./data_raw/TRD_Dalyr_沪深300.csv")
OUT_TXT = Path("./data_raw/hs300_codes_new.txt")

# =========================
# 2. 读取文件
# =========================
def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        for enc in ["utf-8", "utf-8-sig", "gbk", "gb18030"]:
            try:
                return pd.read_csv(path, encoding=enc, low_memory=False)
            except Exception:
                continue
        raise ValueError(f"无法读取 CSV，请检查编码: {path}")
    else:
        raise ValueError(f"暂不支持的文件类型: {path.suffix}")

# =========================
# 3. 自动识别股票代码列
# =========================
def find_code_column(df: pd.DataFrame):
    candidates = [
        "Stkcd", "证券代码", "股票代码", "代码",
        "成分券代码", "成份股代码", "样本股代码",
        "证券代码[Stkcd]"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        c_str = str(c)
        if ("代码" in c_str) or ("Stkcd" in c_str):
            return c
    return None

# =========================
# 4. 统一股票代码格式
# =========================
def standardize_stock_code(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)         # 去掉 Excel/CSV 读出来的 .0
    s = s.str.extract(r"(\d+)", expand=False)          # 提取任意长度数字
    s = s.fillna("").str.zfill(6)                      # 补足 6 位
    s = s.where(s.str.len() == 6, "")                  # 只保留补齐后正好 6 位的代码
    return s

# =========================
# 5. 主流程
# =========================
df = read_table(HS300_FILE)
print("文件列名：", list(df.columns))

code_col = find_code_column(df)
if code_col is None:
    raise ValueError("未找到股票代码列，请检查文件列名。")

print(f"识别到的股票代码列：{code_col}")

df["stock_code_std"] = standardize_stock_code(df[code_col])
codes = (
    df["stock_code_std"]
    .dropna()
    .loc[lambda x: x != ""]
    .drop_duplicates()
    .sort_values()
    .tolist()
)

# 保存为 txt，每行一个代码
OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_TXT, "w", encoding="utf-8") as f:
    for code in codes:
        f.write(code + "\n")

print(f"共提取到 {len(codes)} 个唯一股票代码")
print(f"已保存到: {OUT_TXT}")