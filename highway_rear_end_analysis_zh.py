# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "geopandas",
#     "numpy==2.4.1",
#     "scipy",
#     "matplotlib",
#     "seaborn",
#     "shapely",
#     "folium==0.20.0",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(
    width="full",
    app_title="省道測速照相效益分析 — marimo 互動式分析",
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from shapely.geometry import Point
    from scipy import stats
    from datetime import timedelta
    from pathlib import Path
    import os
    import folium

    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    return Point, Path, folium, gpd, mo, np, os, pd, plt, sns, stats


@app.cell
def _(mo):
    mo.md(r"""
    # 省道測速照相效益分析

    本互動式筆記本評估省道測速照相設備的事故降低成效，透過比較設備設置**前後**的事故率變化進行分析。
    本專案使用 **marimo** 建立互動式分析介面，讓 PM 或業主能夠直接操作參數、選擇站點、
    在地圖上查看空間分布，而非僅閱讀靜態報表。

    ## marimo 互動功能
    - **響應式參數** — 調整緩衝區距離或分析期間，結果自動重新計算
    - **互動式表格** — 點擊列即可查看個別站點詳情
    - **即時地圖** — 相機位置、事故點、緩衝區的空間疊合
    - **儀表板卡片** — 以 styled HTML 呈現的摘要統計

    ## 分析方法
    - **空間分析**：以相機為中心建立緩衝區，與事故紀錄進行空間配對（GeoPandas）
    - **率值比較**：設置前後月均事故率比較
    - **統計檢定**：Mann–Whitney U 檢定（p < 0.05）

    > **說明**：測速照相位置為屏東縣**公開資料**。
    > 事故紀錄為**模擬資料**，資料結構與政府真實資料相同，但數值為虛構。
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 參數設定
    """)
    return


@app.cell
def _(mo):
    buffer_dist_input = mo.ui.number(
        value=150,
        start=50,
        stop=500,
        step=10,
        label="緩衝區距離 (公尺)",
    )

    target_months_input = mo.ui.number(
        value=36,
        start=12,
        stop=60,
        step=6,
        label="分析期間 (月)",
    )

    min_post_months_input = mo.ui.number(
        value=3.0,
        start=1.0,
        stop=12.0,
        step=0.5,
        label="最低設置後期間 (月)",
    )

    mo.hstack([buffer_dist_input, target_months_input, min_post_months_input])
    return buffer_dist_input, min_post_months_input, target_months_input


@app.cell
def _(buffer_dist_input, min_post_months_input, target_months_input):
    BUFFER_DIST = buffer_dist_input.value
    TARGET_MONTHS = target_months_input.value
    MIN_POST_MONTHS = min_post_months_input.value
    return BUFFER_DIST, MIN_POST_MONTHS, TARGET_MONTHS


@app.cell
def _(Path, os, pd):
    # --- Data paths (relative to this file) ---
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = BASE_DIR / "data"

    CAMERA_PATH = DATA_DIR / "cameras.csv"
    ACCIDENT_PATH = DATA_DIR / "accidents_demo.csv"
    SEA_POINTS_PATH = DATA_DIR / "sea_points.csv"

    def convert_roc_date(roc_date_str):
        """Convert ROC (Republic of China) date format to pandas Timestamp.

        Handles formats: YYYMMDD (7 digits), YYMMDD (6 digits), YYYYMMDD (8 digits).
        ROC year = Western year - 1911.
        """
        try:
            if pd.isna(roc_date_str):
                return pd.NaT
            s = str(roc_date_str).replace(".", "").replace("/", "")
            if s.endswith("0") and len(s) > 8:
                s = s[:-1]
            if isinstance(roc_date_str, (int, float)):
                s = str(int(roc_date_str))
            else:
                s = s.split(".")[0]
            if len(s) == 8 and (s.startswith("19") or s.startswith("20")):
                return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
            if len(s) == 7:
                year = int(s[:3]) + 1911
                month = int(s[3:5])
                day = int(s[5:7])
            elif len(s) == 6:
                year = int(s[:2]) + 1911
                month = int(s[2:4])
                day = int(s[4:6])
            else:
                return pd.NaT
            return pd.Timestamp(year=year, month=month, day=day)
        except:
            return pd.NaT
    return ACCIDENT_PATH, CAMERA_PATH, SEA_POINTS_PATH, convert_roc_date


@app.cell
def _(CAMERA_PATH, SEA_POINTS_PATH, convert_roc_date, os, pd):
    """Load camera data and parse setup dates."""
    df_cam_raw = pd.read_csv(CAMERA_PATH)

    # Load sea-point exclusion list (cameras with incorrect coordinates)
    sea_ids = []
    if os.path.exists(SEA_POINTS_PATH):
        df_sea = pd.read_csv(SEA_POINTS_PATH)
        if "編號" in df_sea.columns:
            sea_ids = df_sea["編號"].unique().tolist()

    date_col = "設置日期" if "設置日期" in df_cam_raw.columns else df_cam_raw.columns[3]

    def parse_setup_date(val):
        """Parse camera setup date in ROC format (e.g. '109.05' → 2020-05-01)."""
        try:
            val_str = str(val)
            if "." in val_str:
                parts = val_str.split(".")
                return pd.Timestamp(
                    year=int(parts[0]) + 1911, month=int(parts[1]), day=1
                )
            else:
                return convert_roc_date(val)
        except:
            return pd.NaT

    df_cam_raw["setup_date"] = df_cam_raw[date_col].apply(parse_setup_date)

    lon_col = [c for c in df_cam_raw.columns if "經度" in c or "Longitude" in c][0]
    lat_col = [c for c in df_cam_raw.columns if "緯度" in c or "Latitude" in c][0]
    return df_cam_raw, lat_col, lon_col, sea_ids


@app.cell
def _(ACCIDENT_PATH, pd):
    """Load pre-filtered accident data (provincial highway, speed-related)."""
    df_acc = pd.read_csv(ACCIDENT_PATH, encoding="utf-8", low_memory=False)
    return (df_acc,)


@app.cell
def _(
    BUFFER_DIST,
    Point,
    convert_roc_date,
    df_acc,
    df_cam_raw,
    gpd,
    lat_col,
    lon_col,
    mo,
    pd,
):
    """Process accident data: deduplicate, geocode, and spatial join preparation."""
    # Deduplicate by unique accident ID
    if "總編號" in df_acc.columns:
        df_acc_unique = df_acc.drop_duplicates(subset=["總編號"], keep="first")
    else:
        df_acc_unique = df_acc.copy()

    # Identify coordinate columns
    acc_lon_col = (
        "GPS經度"
        if "GPS經度" in df_acc.columns
        else [c for c in df_acc.columns if "經度" in c][0]
    )
    acc_lat_col = (
        "GPS緯度"
        if "GPS緯度" in df_acc.columns
        else [c for c in df_acc.columns if "緯度" in c][0]
    )
    acc_date_col = (
        "發生日期"
        if "發生日期" in df_acc.columns
        else [c for c in df_acc.columns if "日期" in c][0]
    )

    df_acc_unique[acc_lon_col] = pd.to_numeric(df_acc_unique[acc_lon_col], errors="coerce")
    df_acc_unique[acc_lat_col] = pd.to_numeric(df_acc_unique[acc_lat_col], errors="coerce")
    df_acc_geo = df_acc_unique.dropna(subset=[acc_lon_col, acc_lat_col])

    # Date parsing (ROC → Western calendar)
    df_acc_geo = df_acc_geo.copy()
    df_acc_geo["acc_date"] = df_acc_geo[acc_date_col].apply(convert_roc_date)
    df_acc_geo = df_acc_geo.dropna(subset=["acc_date"])

    # Identify casualty columns
    collision_col = next((c for c in df_acc_geo.columns if "事故類型及型態" in c), None)
    death_col = next((c for c in df_acc_geo.columns if c == "死亡"), None)
    injured_col = next((c for c in df_acc_geo.columns if c == "受傷"), None)
    cause_col = next((c for c in df_acc_geo.columns if "主要肇事因素" in c), None)

    # Ensure numeric types
    if death_col:
        df_acc_geo[death_col] = pd.to_numeric(df_acc_geo[death_col], errors="coerce").fillna(0)
    if injured_col:
        df_acc_geo[injured_col] = pd.to_numeric(df_acc_geo[injured_col], errors="coerce").fillna(0)

    # Calculate total casualties (deaths + injuries)
    if death_col and injured_col:
        df_acc_geo["Casualty"] = df_acc_geo[death_col] + df_acc_geo[injured_col]
        casualty_col = "Casualty"
    else:
        casualty_col = None

    MAX_ACC_DATE = df_acc_geo["acc_date"].max()
    MIN_ACC_DATE = df_acc_geo["acc_date"].min()

    # Build camera buffer zones (project to TWD97 / EPSG:3826 for meter-based buffer)
    df_cam_valid_geo = df_cam_raw.dropna(subset=[lon_col, lat_col])
    gdf_cam = gpd.GeoDataFrame(
        df_cam_valid_geo,
        geometry=[
            Point(xy)
            for xy in zip(df_cam_valid_geo[lon_col], df_cam_valid_geo[lat_col])
        ],
        crs="EPSG:4326",
    ).to_crs(epsg=3826)
    gdf_cam["geometry"] = gdf_cam.geometry.buffer(BUFFER_DIST)

    # Build accident GeoDataFrame
    gdf_acc = gpd.GeoDataFrame(
        df_acc_geo,
        geometry=[
            Point(xy)
            for xy in zip(df_acc_geo[acc_lon_col], df_acc_geo[acc_lat_col])
        ],
        crs="EPSG:4326",
    ).to_crs(epsg=3826)

    mo.md(f"""
    ### 資料處理完成
    - **唯一事故數**: {len(df_acc_unique):,} 件
    - **有座標事故**: {len(gdf_acc):,} 件
    - **資料期間**: {MIN_ACC_DATE.date()} ~ {MAX_ACC_DATE.date()}
    - **緩衝區距離**: {BUFFER_DIST} 公尺
    """)
    return (
        MAX_ACC_DATE,
        MIN_ACC_DATE,
        casualty_col,
        cause_col,
        collision_col,
        death_col,
        df_acc_geo,
        gdf_acc,
        gdf_cam,
        injured_col,
    )


@app.cell
def _(BUFFER_DIST, gdf_acc, gdf_cam, gpd, mo):
    """Spatial join: find accidents within camera buffer zones."""
    joined = gpd.sjoin(gdf_acc, gdf_cam, how="inner", predicate="within")

    mo.md(f"""
    ### 空間連結結果
    - **緩衝區距離**: {BUFFER_DIST} 公尺
    - **匹配事故數**: {len(joined):,} 件
    """)
    return (joined,)


@app.cell
def _(
    MAX_ACC_DATE,
    MIN_ACC_DATE,
    MIN_POST_MONTHS,
    TARGET_MONTHS,
    casualty_col,
    cause_col,
    collision_col,
    death_col,
    df_acc_geo,
    df_cam_raw,
    injured_col,
    joined,
    lat_col,
    lon_col,
    np,
    pd,
    sea_ids,
    stats,
):
    """Core analysis: compute before/after metrics for each camera site."""

    def count_codes(df, col, prefixes):
        """Count rows where column values start with given prefixes."""
        if not col or df.empty:
            return [0] * len(prefixes)
        c = df[col].astype(str).str.strip()
        return [c.str.startswith(p).sum() for p in prefixes]

    def get_monthly_seq(df, s, e, count_source="count", sum_col=None):
        """Build a monthly time series of counts or sums for MWU test."""
        if df.empty:
            return []
        per_idx = pd.period_range(s, e, freq="M")
        if count_source == "sum" and sum_col:
            cnt = df.groupby(df["acc_date"].dt.to_period("M"))[sum_col].sum()
        else:
            cnt = df.groupby(df["acc_date"].dt.to_period("M")).size()
        cnt = cnt.reindex(per_idx, fill_value=0)
        return cnt.values

    results = []

    for idx, cam_row in df_cam_raw.iterrows():
        cam_id = cam_row.get("編號", idx)
        location = str(cam_row.get(df_cam_raw.columns[1], ""))
        setup_date = cam_row["setup_date"]
        lon = cam_row[lon_col]
        lat = cam_row[lat_col]

        status = "OK"
        if cam_id in sea_ids:
            status = "Sea Point"
        elif pd.isna(lon) or pd.isna(lat):
            status = "Missing Coordinates"
        elif pd.isna(setup_date):
            status = "Missing Setup Date"

        # Initialize metrics
        pre_total = post_total = 0
        pre_months_valid = post_months_valid = 0
        pre_rate = post_rate = 0
        raw_red_rate = did_net = 0
        mwu_p = np.nan
        mwu_sig = "No"

        # Casualty metrics
        pre_casualty = post_casualty = 0
        pre_casualty_rate = post_casualty_rate = 0
        casualty_red_rate = did_casualty_net = 0
        mwu_casualty_p = np.nan
        mwu_casualty_sig = "No"

        pre_dead = post_dead = pre_injured = post_injured = 0
        pre_types = [0, 0, 0]
        post_types = [0, 0, 0]
        pre_causes = [0, 0]
        post_causes = [0, 0]
        t_day = (
            setup_date + pd.DateOffset(months=1)
            if not pd.isna(setup_date)
            else pd.NaT
        )

        if status == "OK":
            pre_start_target = t_day - pd.DateOffset(months=TARGET_MONTHS)
            pre_start = max(pre_start_target, MIN_ACC_DATE)
            pre_end = t_day - pd.DateOffset(days=1)

            post_start = t_day + pd.DateOffset(days=1)
            post_end_target = t_day + pd.DateOffset(months=TARGET_MONTHS)
            post_end = min(post_end_target, MAX_ACC_DATE)

            pre_months_valid = (pre_end - pre_start).days / 30.44
            post_months_valid = (post_end - post_start).days / 30.44

            if post_months_valid < MIN_POST_MONTHS:
                status = "Insufficient Data (<3m)"
            if post_start > MAX_ACC_DATE:
                status = "Future Installation"
                post_months_valid = 0

            if idx in joined.index_right.values:
                accs = joined[joined["index_right"] == idx]

                pre_mask = (accs["acc_date"] >= pre_start) & (
                    accs["acc_date"] <= pre_end
                )
                post_mask = (accs["acc_date"] >= post_start) & (
                    accs["acc_date"] <= post_end
                )

                pre_accs = accs[pre_mask]
                post_accs = accs[post_mask]

                # --- Accident counts ---
                pre_total = len(pre_accs)
                post_total = len(post_accs)

                pre_rate = pre_total / pre_months_valid if pre_months_valid > 0 else 0
                post_rate = (
                    post_total / post_months_valid if post_months_valid > 0 else 0
                )

                if pre_rate > 0:
                    raw_red_rate = (pre_rate - post_rate) / pre_rate * 100
                else:
                    raw_red_rate = 0
                    if post_rate > 0:
                        raw_red_rate = -100

                # --- Casualty analysis ---
                if casualty_col:
                    pre_casualty = pre_accs[casualty_col].sum()
                    post_casualty = post_accs[casualty_col].sum()
                    pre_casualty_rate = (
                        pre_casualty / pre_months_valid if pre_months_valid > 0 else 0
                    )
                    post_casualty_rate = (
                        post_casualty / post_months_valid
                        if post_months_valid > 0
                        else 0
                    )
                    if pre_casualty_rate > 0:
                        casualty_red_rate = (
                            (pre_casualty_rate - post_casualty_rate)
                            / pre_casualty_rate
                            * 100
                        )
                    else:
                        casualty_red_rate = 0
                        if post_casualty_rate > 0:
                            casualty_red_rate = -100

                # --- Control group (province-wide filtered accidents) ---
                c_pre_mask = (df_acc_geo["acc_date"] >= pre_start) & (
                    df_acc_geo["acc_date"] <= pre_end
                )
                c_post_mask = (df_acc_geo["acc_date"] >= post_start) & (
                    df_acc_geo["acc_date"] <= post_end
                )

                c_pre_rate = (
                    len(df_acc_geo[c_pre_mask]) / pre_months_valid
                    if pre_months_valid > 0
                    else 0
                )
                c_post_rate = (
                    len(df_acc_geo[c_post_mask]) / post_months_valid
                    if post_months_valid > 0
                    else 0
                )
                ctrl_red_rate = (
                    (c_pre_rate - c_post_rate) / c_pre_rate * 100
                    if c_pre_rate > 0
                    else 0
                )
                did_net = raw_red_rate - ctrl_red_rate

                # Control casualty rate
                if casualty_col:
                    c_pre_casualty = df_acc_geo.loc[c_pre_mask, casualty_col].sum()
                    c_post_casualty = df_acc_geo.loc[c_post_mask, casualty_col].sum()
                    c_pre_cas_rate = (
                        c_pre_casualty / pre_months_valid if pre_months_valid > 0 else 0
                    )
                    c_post_cas_rate = (
                        c_post_casualty / post_months_valid
                        if post_months_valid > 0
                        else 0
                    )
                    ctrl_cas_red_rate = (
                        (c_pre_cas_rate - c_post_cas_rate) / c_pre_cas_rate * 100
                        if c_pre_cas_rate > 0
                        else 0
                    )
                    did_casualty_net = casualty_red_rate - ctrl_cas_red_rate

                # Collision types
                pre_types = count_codes(pre_accs, collision_col, ["13", "12", "18"])
                post_types = count_codes(post_accs, collision_col, ["13", "12", "18"])

                pre_dead = pre_accs[death_col].sum() if death_col else 0
                post_dead = post_accs[death_col].sum() if death_col else 0
                pre_injured = pre_accs[injured_col].sum() if injured_col else 0
                post_injured = post_accs[injured_col].sum() if injured_col else 0

                # --- Mann-Whitney U tests ---
                if status == "OK":
                    pre_seq = get_monthly_seq(pre_accs, pre_start, pre_end)
                    post_seq = get_monthly_seq(post_accs, post_start, post_end)
                    if len(pre_seq) > 0 and len(post_seq) > 0:
                        try:
                            stat, p = stats.mannwhitneyu(
                                pre_seq, post_seq, alternative="greater"
                            )
                            mwu_p = p
                            if p < 0.05:
                                mwu_sig = "Yes"
                        except:
                            pass

                    if casualty_col:
                        pre_cas_seq = get_monthly_seq(
                            pre_accs, pre_start, pre_end,
                            count_source="sum", sum_col=casualty_col,
                        )
                        post_cas_seq = get_monthly_seq(
                            post_accs, post_start, post_end,
                            count_source="sum", sum_col=casualty_col,
                        )
                        if len(pre_cas_seq) > 0 and len(post_cas_seq) > 0:
                            try:
                                stat, p = stats.mannwhitneyu(
                                    pre_cas_seq, post_cas_seq, alternative="greater"
                                )
                                mwu_casualty_p = p
                                if p < 0.05:
                                    mwu_casualty_sig = "Yes"
                            except:
                                pass

        results.append({
            "Camera_ID": cam_id,
            "Location": location,
            "Longitude": lon,
            "Latitude": lat,
            "Setup_Date": t_day,
            "Status": status,
            "Pre_Total": pre_total,
            "Post_Total": post_total,
            "Pre_Months_Valid": round(pre_months_valid, 2),
            "Post_Months_Valid": round(post_months_valid, 2),
            "Pre_Rate_Monthly": round(pre_rate, 2),
            "Post_Rate_Monthly": round(post_rate, 2),
            "Reduction_Rate_%": round(raw_red_rate, 2),
            "DiD_Net_Reduction_%": round(did_net, 2),
            "Test_Counts_MWU_P": round(mwu_p, 4) if not np.isnan(mwu_p) else "",
            "Test_Counts_Sig": mwu_sig,
            "Pre_Casualty": pre_casualty,
            "Post_Casualty": post_casualty,
            "Casualty_Reduction_%": round(casualty_red_rate, 2),
            "DiD_Casualty_%": round(did_casualty_net, 2),
            "Casualty_MWU_P": (
                round(mwu_casualty_p, 4) if not np.isnan(mwu_casualty_p) else ""
            ),
            "Casualty_Sig": mwu_casualty_sig,
            "Pre_Dead": pre_dead,
            "Post_Dead": post_dead,
            "Pre_Injured": pre_injured,
            "Post_Injured": post_injured,
        })

    df_results = pd.DataFrame(results)
    return (df_results,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 分析結果總覽
    """)
    return


@app.cell
def _(df_results, mo, pd):
    """Device status distribution summary."""
    status_counts = df_results["Status"].value_counts()
    status_df = pd.DataFrame(
        {"狀態": status_counts.index, "數量": status_counts.values}
    )

    mo.vstack([mo.md("### 設備狀態分布"), mo.ui.table(status_df, selection=None)])
    return


@app.cell
def _(df_results, mo):
    """Summary dashboard with styled HTML metric cards."""
    df_valid = df_results[df_results["Status"] == "OK"].copy()
    df_valid_with_acc = df_valid[
        (df_valid["Pre_Total"] > 0) | (df_valid["Post_Total"] > 0)
    ]

    n_valid = len(df_valid)
    n_with_acc = len(df_valid_with_acc)

    def get_mean(df, col):
        return round(df[col].mean(), 2) if len(df) > 0 else 0

    def color_val(val):
        if val > 0:
            return "#27ae60"  # Green = reduction (good)
        if val < 0:
            return "#c0392b"  # Red = increase (bad)
        return "#7f8c8d"

    if n_with_acc > 0:
        acc_red = get_mean(df_valid_with_acc, "Reduction_Rate_%")
        acc_imp_cnt = len(df_valid_with_acc[df_valid_with_acc["Reduction_Rate_%"] > 0])
        acc_imp_pct = round(acc_imp_cnt / n_with_acc * 100, 1)

        cas_red = get_mean(df_valid_with_acc, "Casualty_Reduction_%")
        cas_imp_cnt = len(
            df_valid_with_acc[df_valid_with_acc["Casualty_Reduction_%"] > 0]
        )
        cas_imp_pct = round(cas_imp_cnt / n_with_acc * 100, 1)
    else:
        acc_red = acc_imp_cnt = acc_imp_pct = 0
        cas_red = cas_imp_cnt = cas_imp_pct = 0

    def card(title, value, unit, subtext="", color="#2c3e50"):
        return f"""
        <div style="
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border: 1px solid #eee;
            flex: 1;
            min-width: 180px;
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        ">
            <div style="font-size: 0.9em; color: #7f8c8d; margin-bottom: 5px;">{title}</div>
            <div style="font-size: 1.8em; font-weight: bold; color: {color};">{value}<span style="font-size: 0.5em; color: #95a5a6; margin-left: 3px;">{unit}</span></div>
            <div style="font-size: 0.8em; color: #95a5a6; margin-top: 5px;">{subtext}</div>
        </div>
        """

    html_content = f"""
    <div style="font-family: 'Segoe UI', sans-serif; background: #f8f9fa; padding: 20px; border-radius: 10px;">
        <h3 style="margin-top: 0; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; display: inline-block;">事故分析儀表板</h3>

        <div style="margin-top: 15px; margin-bottom: 20px; color: #666;">
            <strong>分析樣本：</strong>有效設備 {n_valid} 台 / 有事故紀錄 {n_with_acc} 台
        </div>

        <h4 style="color: #34495e; margin-bottom: 10px;">事故件數</h4>
        <div style="display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 25px;">
            {card("平均下降率", acc_red, "%", "原始比較", color_val(acc_red))}
            {card("改善站點比例", acc_imp_pct, "%", f"{acc_imp_cnt} / {n_with_acc} 台", color_val(acc_imp_pct-50))}
        </div>

        <h4 style="color: #34495e; margin-bottom: 10px;">死傷人數</h4>
        <div style="display: flex; gap: 15px; flex-wrap: wrap;">
            {card("平均下降率", cas_red, "%", "原始比較", color_val(cas_red))}
            {card("改善站點比例", cas_imp_pct, "%", f"{cas_imp_cnt} / {n_with_acc} 台", color_val(cas_imp_pct-50))}
        </div>
    </div>
    """

    mo.md(html_content)
    return (df_valid_with_acc,)


@app.cell
def _(mo):
    mo.md(f"""
    ---
    ### 有事故資料的設備清單
    """)
    return


@app.cell
def _(df_valid_with_acc, plt, sns):
    """Distribution charts: reduction rates and improvement counts."""
    if len(df_valid_with_acc) > 0:
        fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))

        sns.histplot(
            df_valid_with_acc["Reduction_Rate_%"],
            bins=20, kde=True, ax=axes1[0, 0], color="steelblue",
        )
        axes1[0, 0].axvline(x=0, color="red", linestyle="--")
        axes1[0, 0].set_title("事故下降率分布")

        sns.histplot(
            df_valid_with_acc["Casualty_Reduction_%"],
            bins=20, kde=True, ax=axes1[0, 1], color="mediumpurple",
        )
        axes1[0, 1].axvline(x=0, color="red", linestyle="--")
        axes1[0, 1].set_xlabel("死傷下降率 (%)")
        axes1[0, 1].set_title("死傷人數下降率分布")

        acc_improved = len(df_valid_with_acc[df_valid_with_acc["Reduction_Rate_%"] > 0])
        acc_worsened = len(df_valid_with_acc[df_valid_with_acc["Reduction_Rate_%"] <= 0])
        cas_improved = len(df_valid_with_acc[df_valid_with_acc["Casualty_Reduction_%"] > 0])
        cas_worsened = len(df_valid_with_acc[df_valid_with_acc["Casualty_Reduction_%"] <= 0])

        sns.barplot(
            x=["事故改善", "事故惡化"],
            y=[acc_improved, acc_worsened],
            ax=axes1[1, 0],
            palette=["#2ecc71", "#e74c3c"],
        )
        axes1[1, 0].set_title(f"事故改善站點比例 ({acc_improved}/{len(df_valid_with_acc)})")
        axes1[1, 0].bar_label(axes1[1, 0].containers[0])

        sns.barplot(
            x=["死傷改善", "死傷惡化"],
            y=[cas_improved, cas_worsened],
            ax=axes1[1, 1],
            palette=["#9b59b6", "#e74c3c"],
        )
        axes1[1, 1].set_title(f"死傷改善站點比例 ({cas_improved}/{len(df_valid_with_acc)})")
        axes1[1, 1].bar_label(axes1[1, 1].containers[0])

        plt.tight_layout()
        fig1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 詳細結果表格

    請勾選欲查看的站點，下方將顯示詳細資訊與地圖。
    """)
    return


@app.cell
def _(df_results, mo):
    """Interactive sortable table with multi-select."""
    df_sorted = df_results.sort_values(
        by=["Reduction_Rate_%", "Casualty_Reduction_%"], ascending=[False, False]
    )

    table_ui = mo.ui.table(
        df_sorted[
            [
                "Camera_ID", "Location", "Setup_Date", "Status",
                "Longitude", "Latitude",
                "Pre_Total", "Post_Total", "Reduction_Rate_%",
                "Pre_Casualty", "Post_Casualty", "Casualty_Reduction_%",
            ]
        ],
        selection="multi",
        pagination=True,
        page_size=15,
    )

    mo.vstack([mo.md("### 站點資料"), table_ui])
    return (table_ui,)


@app.cell
def _(mo, table_ui):
    """Detail cards for selected rows — reactive to table selection."""
    selected_df = table_ui.value

    if len(selected_df) > 0:
        cards_html = ""
        for _, row in selected_df.iterrows():
            acc_color = (
                "#27ae60" if row["Reduction_Rate_%"] > 0
                else "#c0392b" if row["Reduction_Rate_%"] < 0
                else "#7f8c8d"
            )
            cas_color = (
                "#8e44ad" if row["Casualty_Reduction_%"] > 0
                else "#c0392b" if row["Casualty_Reduction_%"] < 0
                else "#7f8c8d"
            )

            cards_html += f"""
            <div style="
                background: white;
                border-left: 5px solid {acc_color};
                border-radius: 5px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 15px;
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                align-items: center;
            ">
                <div style="flex: 2; min-width: 200px;">
                    <div style="font-size: 0.8em; color: #95a5a6; margin-bottom: 2px;">{row['Camera_ID']} | {str(row['Setup_Date'])[:10]}</div>
                    <div style="font-size: 1.2em; font-weight: bold; color: #2c3e50;">{row['Location']}</div>
                    <div style="font-size: 0.85em; color: {acc_color}; margin-top: 5px;">
                        狀態: <strong>{row['Status']}</strong>
                    </div>
                </div>

                <div style="flex: 1; min-width: 140px; text-align: center; border-right: 1px solid #eee;">
                    <div style="font-size: 0.8em; color: #7f8c8d;">事故件數變化</div>
                    <div style="font-weight: bold; font-size: 1.1em; color: {acc_color};">
                        {int(row['Pre_Total'])} -> {int(row['Post_Total'])}
                    </div>
                    <div style="font-size: 0.75em; color: #95a5a6;">(降 {row['Reduction_Rate_%']}%)</div>
                </div>

                <div style="flex: 1; min-width: 140px; text-align: center;">
                    <div style="font-size: 0.8em; color: #7f8c8d;">死傷人數變化</div>
                    <div style="font-weight: bold; font-size: 1.1em; color: {cas_color};">
                        {int(row['Pre_Casualty'])} -> {int(row['Post_Casualty'])}
                    </div>
                    <div style="font-size: 0.75em; color: #95a5a6;">(降 {row['Casualty_Reduction_%']}%)</div>
                </div>
            </div>
            """

        display_content = mo.vstack([
            mo.md(f"### 選定站點詳情 ({len(selected_df)})"),
            mo.Html(
                f"<div style='background: #f4f6f7; padding: 15px; border-radius: 8px;'>{cards_html}</div>"
            ),
        ])
    else:
        display_content = mo.md("_請在上方表格勾選站點以查看詳細資訊_")

    display_content
    return (selected_df,)


@app.cell
def _(
    BUFFER_DIST,
    MAX_ACC_DATE,
    MIN_ACC_DATE,
    TARGET_MONTHS,
    folium,
    joined,
    mo,
    pd,
    selected_df,
):
    """Interactive folium map showing selected cameras, buffer zones, and accidents."""
    if len(selected_df) > 0:
        map_center_lat = selected_df["Latitude"].mean()
        map_center_lon = selected_df["Longitude"].mean()
        map_obj = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=14)

        # Find coordinate columns in joined data
        map_acc_lon_col = (
            "GPS經度" if "GPS經度" in joined.columns
            else next((c for c in joined.columns if "經度" in c and c != "Longitude"), None)
        )
        map_acc_lat_col = (
            "GPS緯度" if "GPS緯度" in joined.columns
            else next((c for c in joined.columns if "緯度" in c and c != "Latitude"), None)
        )

        for _, map_cam_row in selected_df.iterrows():
            map_cam_lat = map_cam_row["Latitude"]
            map_cam_lon = map_cam_row["Longitude"]
            map_cam_id = map_cam_row["Camera_ID"]
            map_setup_date = map_cam_row["Setup_Date"]

            # Camera marker (blue)
            folium.Marker(
                location=[map_cam_lat, map_cam_lon],
                popup=f"<b>{map_cam_id}</b><br>{map_cam_row['Location']}<br>設置日期: {str(map_setup_date)[:10]}",
                icon=folium.Icon(color="blue", icon="camera", prefix="fa"),
                tooltip=f"{map_cam_id}",
            ).add_to(map_obj)

            # Buffer circle
            folium.Circle(
                location=[map_cam_lat, map_cam_lon],
                radius=BUFFER_DIST,
                color="blue",
                fill=True,
                fill_opacity=0.1,
                weight=2,
            ).add_to(map_obj)

            # Analysis period
            if not pd.isna(map_setup_date):
                map_t_day = map_setup_date
                map_pre_start = max(
                    map_t_day - pd.DateOffset(months=TARGET_MONTHS), MIN_ACC_DATE
                )
                map_pre_end = map_t_day - pd.DateOffset(days=1)
                map_post_start = map_t_day + pd.DateOffset(days=1)
                map_post_end = min(
                    map_t_day + pd.DateOffset(months=TARGET_MONTHS), MAX_ACC_DATE
                )
            else:
                map_pre_start = map_pre_end = map_post_start = map_post_end = None

            # Plot accident points (only within analysis period)
            if "編號" in joined.columns:
                map_cam_accs = joined[joined["編號"] == map_cam_id]
            else:
                map_cam_accs = pd.DataFrame()

            if (
                not map_cam_accs.empty
                and not pd.isna(map_setup_date)
                and map_acc_lat_col
                and map_acc_lon_col
                and map_pre_start is not None
            ):
                for _, map_acc in map_cam_accs.iterrows():
                    try:
                        map_acc_lat = float(map_acc[map_acc_lat_col])
                        map_acc_lon = float(map_acc[map_acc_lon_col])
                        map_acc_date = map_acc.get("acc_date", None)

                        if pd.isna(map_acc_date):
                            continue

                        map_is_pre = (map_acc_date >= map_pre_start) & (
                            map_acc_date <= map_pre_end
                        )
                        map_is_post = (map_acc_date >= map_post_start) & (
                            map_acc_date <= map_post_end
                        )

                        if not (map_is_pre or map_is_post):
                            continue

                        if map_is_pre:
                            map_color = "orange"
                            map_period = "設置前"
                        else:
                            map_color = "green"
                            map_period = "設置後"

                        map_casualty = map_acc.get("Casualty", 0)
                        map_popup_text = f"<b>{map_period}</b><br>日期: {str(map_acc_date)[:10]}<br>死傷: {int(map_casualty) if pd.notna(map_casualty) else 0}人"

                        folium.CircleMarker(
                            location=[map_acc_lat, map_acc_lon],
                            radius=6,
                            color=map_color,
                            fill=True,
                            fill_color=map_color,
                            fill_opacity=0.7,
                            popup=map_popup_text,
                            tooltip=f"{map_period}: {str(map_acc_date)[:10]}",
                        ).add_to(map_obj)
                    except:
                        pass

        map_legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background: white; padding: 10px; border-radius: 5px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3); font-size: 12px;">
            <b>圖例</b><br>
            藍色標記 = 測速照相<br>
            橘色點 = 設置前事故<br>
            綠色點 = 設置後事故
        </div>
        """
        map_obj.get_root().html.add_child(folium.Element(map_legend_html))

        map_html_content = map_obj._repr_html_()
        map_display = mo.vstack([
            mo.md("### 站點地圖"),
            mo.Html(
                f"<div style='height: 500px; border-radius: 8px; overflow: hidden;'>{map_html_content}</div>"
            ),
        ])
    else:
        map_display = mo.md("_選取站點後將顯示地圖_")

    map_display
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 分析說明

    ### 資料篩選條件

    本分析篩選省道上與超速相關之事故資料，篩選邏輯如下：

    1. **事故類型及型態**（可能因超速所導致）：
       - 12: 同向擦撞、13: 追撞、18: 路上翻車/摔倒、
         19-29: 各類撞固定物

    2. **主要肇事因素**（推測為超速違規）：
       - 112.07 前：13 (超速失控)、14 (未依規定減速)、
         16 (未保持行車安全距離)、17 (未保持行車安全間隔)
       - 112.07 後：5、6、7、8（更新後之分類代碼）

    ### 指標定義
    - **Reduction_Rate_%**：原始事故下降率（設置前後月均事故率比較）
    - **DiD_Net_Reduction_%**：差異中之差異淨效果（排除全省趨勢影響）
    - **Casualty**：死傷人數（死亡 + 受傷）
    - **Mann-Whitney U**：無母數檢定（p < 0.05 為顯著）

    > **資料說明**：測速照相位置為屏東縣公開資料。
    > 事故紀錄為模擬資料，用於展示分析框架與互動功能。
    """)
    return


if __name__ == "__main__":
    app.run()
