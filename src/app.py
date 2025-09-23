# app.py 
# -*- coding: utf-8 -*-
import os
import json
import warnings
import re
import traceback
import numpy as np
import pandas as pd
from matrix_builder_processor import process_matrix_builder as mb_process
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from jinja2 import TemplateNotFound
from mass_spectrometry_processor import process_mass_spectrometry_data

# 画图
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 频率谷点
from scipy.interpolate import UnivariateSpline
from scipy.signal import argrelextrema

# NMF / 聚类
from sklearn.decomposition import NMF
from scipy.cluster.hierarchy import linkage, leaves_list

# UMAP 与网络
try:
    import umap  # umap-learn
except Exception:
    umap = None

try:
    import networkx as nx
except Exception:
    nx = None

# Dash / Plotly
try:
    from dash import Dash, dcc, html, Input, Output
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
except Exception:
    Dash = None
    dcc = None
    html = None
    Input = Output = None
    dbc = None
    go = None

warnings.filterwarnings("ignore")

# ---------------------------
# Flask 基础
# ---------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config.setdefault("UPLOAD_FOLDER", "uploads")
app.config.setdefault("OUTPUT_FOLDER", "output")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB

GLOBAL_VIS = {"processor": None}  # Dash 图形所用的全局处理器

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

ensure_dir(app.config["UPLOAD_FOLDER"])
ensure_dir(app.config["OUTPUT_FOLDER"])

# ---------------------------
# 工具：安全数值/ID
# ---------------------------
def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _normalize_id(x):
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s

def _stamp(prefix: str, suffix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}{suffix}"

def _dl(path: str) -> str:
    return f"/download/{os.path.basename(path)}"

# ---------------------------
# 稳健读取：强度矩阵 / 频率表 自动识别
# ---------------------------
def read_intensity_or_frequency(csv_path: str):
    """
    返回 (mode, df)
    - mode == "intensity"：df 行为样本，列为 m/z（float），值为强度
    - mode == "frequency"：df 包含列 ["mz","Frequency_Count","Frequency_Rate"]
    """
    # 先尝试频率表
    try:
        tmp = pd.read_csv(csv_path)
        lower = {c.lower(): c for c in tmp.columns}
        if "mz" in lower and ("frequency_rate" in lower or "frequency_count" in lower):
            colmap = {}
            for c in tmp.columns:
                if c.lower() == "mz": colmap[c] = "mz"
                elif c.lower() == "frequency_rate": colmap[c] = "Frequency_Rate"
                elif c.lower() == "frequency_count": colmap[c] = "Frequency_Count"
            tmp = tmp.rename(columns=colmap)
            tmp["mz"] = pd.to_numeric(tmp["mz"], errors="coerce")
            if "Frequency_Rate" in tmp.columns:
                tmp["Frequency_Rate"] = pd.to_numeric(tmp["Frequency_Rate"], errors="coerce")
            if "Frequency_Count" in tmp.columns:
                tmp["Frequency_Count"] = pd.to_numeric(tmp["Frequency_Count"], errors="coerce")
            tmp = tmp.dropna(subset=["mz"]).sort_values("mz").reset_index(drop=True)
            return "frequency", tmp
    except Exception:
        pass

    # 尝试强度矩阵（第一列为索引/ID）
    inten = pd.read_csv(csv_path, index_col=0)
    # 列名应为 m/z，可转 float
    try:
        inten.columns = inten.columns.astype(float)
    except Exception as e:
        raise ValueError("该 CSV 不是频率表，也不是有效的强度矩阵（列名无法解析为 m/z 浮点数）")
    # 统一转数值
    inten = inten.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return "intensity", inten

# -------------------------------------------------------------
# 类：NMF 评估（重构误差 & 解释方差）
# -------------------------------------------------------------
class NMFEvaluator:
    def __init__(self, data_file, output_dir):
        self.data_file = data_file
        self.output_dir = ensure_dir(output_dir)
        self.matrix_df = None
        self.data_matrix = None

    def load_data(self):
        self.matrix_df = pd.read_csv(self.data_file, index_col=0)
        self.matrix_df = self.matrix_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        self.data_matrix = self.matrix_df.values

    def evaluate(self, max_components=20, min_components=2, step=1):
        if self.data_matrix is None:
            self.load_data()

        n_samples, n_features = self.data_matrix.shape
        max_components = min(max_components, min(n_samples, n_features))
        components_range = range(min_components, max_components + 1, step)

        # Reconstruction error
        reconstruction_errors = []
        for n in components_range:
            model = NMF(n_components=n, init="random", random_state=42, max_iter=1000)
            model.fit_transform(self.data_matrix)
            reconstruction_errors.append(model.reconstruction_err_)

        plt.figure(figsize=(10, 6))
        plt.plot(list(components_range), reconstruction_errors, marker="o", linestyle="-")
        plt.xlabel("Number of Components (n_components)")
        plt.ylabel("Reconstruction Error")
        plt.title("Choosing Optimal n_components for CNMF")
        plt.grid()
        error_plot_path = os.path.join(self.output_dir, "reconstruction_error.png")
        plt.savefig(error_plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        # Explained variance (proxy via W variance)
        explained_variances = []
        total_variance = np.var(self.data_matrix)
        for n in components_range:
            nmf_model = NMF(n_components=n, init="random", random_state=42, max_iter=1000)
            W = nmf_model.fit_transform(self.data_matrix)
            component_var = np.var(W, axis=0)
            explained_variances.append(np.sum(component_var) / total_variance if total_variance > 0 else 0)

        plt.figure(figsize=(10, 6))
        plt.plot(list(components_range), explained_variances, marker="o", linestyle="-")
        plt.xlabel("Number of Components (n_components)")
        plt.ylabel("Explained Variance Ratio (proxy)")
        plt.title("Choosing Optimal n_components using Explained Variance")
        plt.grid()
        variance_plot_path = os.path.join(self.output_dir, "explained_variance.png")
        plt.savefig(variance_plot_path, dpi=200, bbox_inches="tight")
        plt.close()

# -------------------------------------------------------------
# 类：UMAP 可视化处理（使用 W 矩阵）
# -------------------------------------------------------------
class NMFProcessor:
    def __init__(self, w_matrix_file, metadata_file, gnps_graphml_path, output_dir, n_components=11):
        self.w_matrix_file = w_matrix_file
        self.metadata_file = metadata_file
        self.gnps_graphml_path = gnps_graphml_path
        self.output_dir = ensure_dir(output_dir)
        self.n_components = int(n_components)

        self.W_df = None
        self.edges_df = pd.DataFrame(columns=["Node 1", "Node 2", "Cosine Score"])
        self.color_map = {}

        self.custom_palette = [
            "#E46A6A", "#64B5F6", "#81C784", "#FFD54F",
            "#C37BCF", "#4DB6AC", "#F27AA2", "#FCBB74",
            "#A1887F", "#7D949F", "#7986CB", "#DCE775", "#B3E5FC",
            "#FF8A80", "#E0F7FA"
        ]

    def load_w(self):
        self.W_df = pd.read_csv(self.w_matrix_file, index_col=0)
        self.W_df = self.W_df.apply(pd.to_numeric, errors="coerce").fillna(0)
        numeric_cols = self.W_df.select_dtypes(include="number").columns
        if len(numeric_cols) < self.n_components:
            self.n_components = len(numeric_cols)
        return self.W_df

    def process_umap(self, n_neighbors=10, min_dist=1.0, random_state=42, diff_threshold=0.05):
        if umap is None:
            raise RuntimeError("缺少依赖：umap-learn。请先安装 `pip install umap-learn`")

        def _comp_num(name: str) -> int:
            m = re.search(r'(\d+)$', str(name))
            return int(m.group(1)) if m else 10 ** 9
        self.load_w()
        comp_cols = [c for c in self.W_df.columns if re.match(r'^Component_\d+$', str(c))]
        comp_cols = sorted(comp_cols, key=_comp_num)
        numeric_cols = comp_cols[:self.n_components]

        # Top components & node type
        top_n = 3
        self.W_df["Top_Components"] = self.W_df[numeric_cols].apply(
            lambda row: row.nlargest(top_n).index.tolist(), axis=1
        )
        self.W_df["Top_Contributions"] = self.W_df[numeric_cols].apply(
            lambda row: row.nlargest(top_n).values.tolist(), axis=1
        )

        def determine_node_type(row):
            c1 = row["Top_Contributions"][0]
            c2 = row["Top_Contributions"][1] if len(row["Top_Contributions"]) > 1 else 0
            ratio = abs(c1 - c2) / (c1 + c2) if (c1 + c2) > 0 else 1
            return "multi" if ratio < diff_threshold else "single"

        self.W_df["Node_Type"] = self.W_df.apply(determine_node_type, axis=1)

        # UMAP
        reducer = umap.UMAP(n_neighbors=int(n_neighbors), min_dist=float(min_dist),
                            metric="euclidean", random_state=int(random_state))
        embedding = reducer.fit_transform(self.W_df[numeric_cols])
        self.W_df["UMAP_1"] = embedding[:, 0]
        self.W_df["UMAP_2"] = embedding[:, 1]

        # metadata（可选）
        if self.metadata_file and os.path.exists(self.metadata_file):
            meta = pd.read_csv(self.metadata_file)
            if "row ID" in meta.columns:
                meta["row ID"] = meta["row ID"].map(_normalize_id)
                self.W_df.index = self.W_df.index.map(_normalize_id)
                meta_map = meta.set_index("row ID")[["row m/z", "row retention time"]].to_dict(orient="index")
                self.W_df["m/z"] = self.W_df.index.map(lambda i: _safe_float(meta_map.get(i, {}).get("row m/z")))
                self.W_df["Retention Time"] = self.W_df.index.map(lambda i: _safe_float(meta_map.get(i, {}).get("row retention time")))
            else:
                self.W_df["m/z"] = np.nan
                self.W_df["Retention Time"] = np.nan
        else:
            self.W_df["m/z"] = np.nan
            self.W_df["Retention Time"] = np.nan

        # 颜色映射
        all_components = comp_cols
        self.color_map = {comp: self.custom_palette[i % len(self.custom_palette)]
                          for i, comp in enumerate(all_components)}
        self.W_df["Primary_Component"] = self.W_df["Top_Components"].apply(lambda xs: xs[0] if xs else None)
        self.W_df["Color"] = self.W_df["Primary_Component"].map(self.color_map)

        # GNPS 边
        if self.gnps_graphml_path and os.path.exists(self.gnps_graphml_path):
            if nx is None:
                raise RuntimeError("缺少依赖：networkx。请先安装 `pip install networkx`")

            g = nx.read_graphml(self.gnps_graphml_path)
            rows = []
            for u, v, ed in g.edges(data=True):
                rows.append({
                    "Node 1": _normalize_id(u),
                    "Node 2": _normalize_id(v),
                    "Cosine Score": _safe_float(ed.get("cosine_score", 1.0), 1.0),
                })
            df_e = pd.DataFrame(rows)
            nodes = set(self.W_df.index)
            self.edges_df = df_e[
                df_e["Node 1"].isin(nodes) & df_e["Node 2"].isin(nodes)
            ].copy()
            self.edges_df.to_csv(os.path.join(self.output_dir, "edges_df.csv"), index=False)

        try:
            vc = self.W_df["Top_Components"].str[0].value_counts()
            print("[DEBUG] Primary component counts:\n", vc)
        except Exception:
            pass

        all_nodes_path = os.path.join(self.output_dir, f"all_nodes_info-{self.n_components}.csv")
        self.W_df.to_csv(all_nodes_path)
        return self.W_df

    def plot_graph(self, highlight_samples=None, display_option="None", cosine_range=None):
        if go is None:
            raise RuntimeError("缺少依赖：plotly。请先安装 `pip install plotly`")
        if highlight_samples is None: highlight_samples = []
        if cosine_range is None: cosine_range = [0.6, 1.0]

        need = ["Top_Components", "Primary_Component", "UMAP_1", "UMAP_2", "Node_Type", "m/z", "Retention Time"]
        for c in need:
            if c not in self.W_df.columns:
                return go.Figure()

        fig = go.Figure()

        def _comp_num(name: str) -> int:
            m = re.search(r'(\d+)$', str(name))
            return int(m.group(1)) if m else 10 ** 9

        # 图层：各成分的点
        all_components = sorted(
            list(self.W_df["Top_Components"].explode().dropna().unique()),
            key=_comp_num
        )

        for comp in all_components:
            dfc = self.W_df[self.W_df["Primary_Component"] == comp]
            if dfc.empty: continue
            color = self.color_map.get(comp, "gray")

            hover_text = dfc.apply(
                lambda r: (
                    f"Sample ID: {r.name}<br>"
                    f"Component: {' & '.join(r['Top_Components'][:2]) if r['Node_Type']=='multi' else r['Primary_Component']}<br>"
                    f"m/z: {('%.4f' % r['m/z']) if pd.notna(r['m/z']) else 'N/A'}<br>"
                    f"Retention Time: {('%.2f' % r['Retention Time']) if pd.notna(r['Retention Time']) else 'N/A'}"
                ), axis=1
            )

            if display_option == "Node ID":
                text_info = dfc.index.astype(str)
            elif display_option == "PEPMASS":
                text_info = dfc["m/z"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
            elif display_option == "Retention Time":
                text_info = dfc["Retention Time"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
            elif display_option == "All":
                def _fmt_row(r):
                    a = f"{r['m/z']:.4f}" if pd.notna(r["m/z"]) else "N/A"
                    b = f"{r['Retention Time']:.2f}" if pd.notna(r["Retention Time"]) else "N/A"
                    return f"{r.name}, {a}, {b}"
                text_info = dfc.apply(_fmt_row, axis=1)
            else:
                text_info = ""

            # multi 节点用第二成分颜色描边
            line_colors, line_widths = [], []
            for idx in dfc.index:
                row = self.W_df.loc[idx]
                if row["Node_Type"] == "multi" and len(row["Top_Components"]) > 1:
                    second = row["Top_Components"][1]
                    line_colors.append(self.color_map.get(second, "black"))
                    line_widths.append(4)
                else:
                    line_colors.append("rgba(0,0,0,0)")
                    line_widths.append(0)

            fig.add_trace(go.Scatter(
                x=dfc["UMAP_1"], y=dfc["UMAP_2"],
                mode="markers+text" if display_option != "None" else "markers",
                marker=dict(
                    size=12, color=color, opacity=0.85,
                    line=dict(width=line_widths, color=line_colors)
                ),
                text=text_info, textposition="top center",
                textfont=dict(size=8),
                hoverinfo="text", hovertext=hover_text,
                name=str(comp)
            ))

        # 高亮样本
        if highlight_samples:
            ids = [_normalize_id(x) for x in highlight_samples if x in self.W_df.index]
            if ids:
                h = self.W_df.loc[ids]
                fig.add_trace(go.Scatter(
                    x=h["UMAP_1"], y=h["UMAP_2"],
                    mode="markers",
                    marker=dict(
                        size=14,
                        color=h["Primary_Component"].map(self.color_map),
                        opacity=1.0,
                        line=dict(width=4, color="red")
                    ),
                    hoverinfo="text",
                    hovertext=h.apply(
                        lambda r: (
                            f"🔍 FOUND!<br>Sample ID: {r.name}<br>"
                            f"Component: {' & '.join(r['Top_Components'][:2]) if r['Node_Type']=='multi' else r['Primary_Component']}<br>"
                            f"m/z: {('%.4f' % r['m/z']) if pd.notna(r['m/z']) else 'N/A'}<br>"
                            f"Retention Time: {('%.2f' % r['Retention Time']) if pd.notna(r['Retention Time']) else 'N/A'}"
                        ), axis=1
                    ),
                    showlegend=False
                ))

        # GNPS 边
        cmin, cmax = cosine_range
        if not self.edges_df.empty:
            fedges = self.edges_df[
                (self.edges_df["Cosine Score"] >= cmin) & (self.edges_df["Cosine Score"] <= cmax)
            ]
            for _, ed in fedges.iterrows():
                n1, n2, sc = ed["Node 1"], ed["Node 2"], ed["Cosine Score"]
                if n1 in self.W_df.index and n2 in self.W_df.index:
                    fig.add_trace(go.Scatter(
                        x=[self.W_df.loc[n1, "UMAP_1"], self.W_df.loc[n2, "UMAP_1"]],
                        y=[self.W_df.loc[n1, "UMAP_2"], self.W_df.loc[n2, "UMAP_2"]],
                        mode="lines",
                        line=dict(width=max(1.0, sc * 2.0), color = "gray"),
                        opacity=float(np.clip(sc, 0.2, 1.0)),
                        hoverinfo="text",
                        hovertext=f"Cosine Score: {sc:.2f}",
                        showlegend=False
                    ))

        fig.update_layout(
            width=1000, height=650, template="plotly_white",
            title="UMAP Visualization with Interactive Search",
            xaxis_title="UMAP Component 1",
            yaxis_title="UMAP Component 2",
            hovermode="closest",
            legend_title="Components",
            xaxis=dict(showgrid=False, zeroline=False, showline=False),
            yaxis=dict(showgrid=False, zeroline=False, showline=False),
            paper_bgcolor="white", plot_bgcolor="white"
        )
        return fig


# -------------------------------------------------------------
# Dash 子应用（/umap_visualization/），并提供 /umap 的别名
# -------------------------------------------------------------
if Dash is not None:
    dash_app = Dash(
        __name__,
        server=app,
        url_base_pathname="/umap_visualization/",
        external_stylesheets=[dbc.themes.BOOTSTRAP] if dbc else None,
    )

    dash_app.layout = html.Div([
        html.H3("Interactive UMAP visualization", className="text-center mt-3"),
        dcc.Loading(
            id="loading",
            children=[dcc.Graph(id="umap-graph", style={"width": "100%", "height": "650px"})],
            type="circle",
        ),
        html.Div([
            html.Label("Node ID:"), dcc.Input(id="node-id", type="text", style={"margin": "10px"}),
            html.Label("PEPMASS:"), dcc.Input(id="pepmass", type="text", style={"margin": "10px"}),
            html.Label("Show:"),
            dcc.Dropdown(
                id="display-option",
                options=[{"label": k, "value": k} for k in ["None", "Node ID", "PEPMASS", "Retention Time", "All"]],
                value="None", style={"width": "220px", "margin": "10px"}
            ),
            html.Label("Cosine Range:"),
            dcc.RangeSlider(
                id="cosine-range", min=0.0, max=1.0, step=0.01, value=[0.6, 1.0],
                marks={0: "0", 0.5: "0.5", 1: "1"}, tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], style={"padding": "10px"}),
        html.Div(id="umap-error", style={"color": "red", "textAlign": "center"})
    ])

    @dash_app.callback(
        [Output("umap-graph", "figure"), Output("umap-error", "children")],
        [Input("node-id", "value"), Input("pepmass", "value"),
         Input("display-option", "value"), Input("cosine-range", "value")]
    )
    def update_umap(node_id, pepmass, display_option, cosine_range):
        proc = GLOBAL_VIS.get("processor")
        if proc is None:
            return go.Figure(), "请先通过 /process_nmf_processor 或 /nmf/process 初始化可视化数据。"
        try:
            highlights = []
            if node_id and str(node_id).strip():
                nid = _normalize_id(node_id)
                if nid in proc.W_df.index:
                    highlights.append(nid)
            if pepmass and str(pepmass).strip():
                mz = _safe_float(pepmass, None)
                if mz is not None and "m/z" in proc.W_df.columns:
                    matches = proc.W_df[np.isclose(proc.W_df["m/z"], mz, atol=0.01)].index.tolist()
                    highlights.extend(matches)

            highlights = list(set(highlights))
            fig = proc.plot_graph(highlight_samples=highlights, display_option=display_option, cosine_range=cosine_range)
            return fig, ""
        except Exception as e:
            return go.Figure(), f"绘图失败：{e}"

    @app.route("/umap")
    def umap_alias():
        return redirect("/umap_visualization/", code=302)

else:
    @app.route("/umap")
    @app.route("/umap_visualization/")
    def umap_unavailable():
        return "UMAP 可视化不可用：缺少 dash/plotly 依赖。请安装：pip install dash plotly", 501


# -------------------------------------------------------------
# 视图页（渲染你的模板）
# -------------------------------------------------------------
@app.route("/")
def index():
    try:
        return render_template("index.html")
    except TemplateNotFound:
        return "<h3>主页</h3><p>请在 templates/ 放置 index.html。可访问：/matrix_builder、/mass_spectrometry、/umap、/nmf</p>"

@app.route("/process_mass_spectrometry", methods=["POST"])
def process_mass_spectrometry():
    try:
        mgf = request.files.get("mgf_file")
        csv = request.files.get("csv_file")

        if not mgf or not csv:
            return jsonify({"error": "请同时上传 MGF 与 CSV 文件"}), 400

        up = ensure_dir(app.config["UPLOAD_FOLDER"])
        out = ensure_dir(app.config["OUTPUT_FOLDER"])

        mgf_path = os.path.join(up, mgf.filename)
        csv_path = os.path.join(up, csv.filename)
        mgf.save(mgf_path)
        csv.save(csv_path)

        tolerance = float(request.form.get("tolerance", 0.05))
        intensity_threshold = float(request.form.get("intensity_threshold", 1e6))
        features = request.form.getlist("features[]")
        features = [float(f) for f in features] if features else []

        water_mass = float(request.form.get("water_mass", 18.01528))
        water_count = request.form.get("water_count", "1")
        tolerance_dehydration = float(request.form.get("tolerance_dehydration", 0.05))
        intensity_threshold_dehydration = float(request.form.get("intensity_threshold_dehydration", 70000))
        mz_threshold = float(request.form.get("mz_threshold", 150))

        results = process_mass_spectrometry_data(
            mgf_file_path=mgf_path,
            input_csv=csv_path,
            output_path=os.path.join(out, "processed"),
            tolerance=tolerance,
            intensity_threshold=intensity_threshold,
            features=features,
            water_mass=water_mass,
            water_count=water_count,
            tolerance_dehydration=tolerance_dehydration,
            intensity_threshold_dehydration=intensity_threshold_dehydration,
            mz_threshold=mz_threshold
        )

        def make_download_link(path):
            fname = os.path.basename(path)
            return f"/download/{fname}"

        return jsonify({
            "result": {
                "score_csv": make_download_link(results["score_csv"]),
                "filtered_quant_csv": make_download_link(results["filtered_quant_csv"]),
                "h2o_score_csv": make_download_link(results["h2o_score_csv"]),
                "h2o_quant_csv": make_download_link(results["h2o_quant_csv"]),
                "filtered_mgf": make_download_link(results["filtered_mgf"]),
                "filtered_spectra_count": results["filtered_spectra_count"],
                "matched_count": results["matched_count"]
            }
        })
    except Exception as e:
        print("[ERROR] /process_mass_spectrometry:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/mass_spectrometry")
def mass_spectrometry_page():
    try:
        return render_template("mass_spectrometry.html")
    except TemplateNotFound:
        return "<h3>质谱数据预处理页面（模板缺失）</h3>"

@app.route("/matrix_builder")
def matrix_builder_page():
    try:
        return render_template("matrix_builder.html")
    except TemplateNotFound:
        return "<h3>数据矩阵构建页面（模板缺失）</h3>"

@app.route("/nmf")
def nmf_page():
    try:
        return render_template("nmf.html")
    except TemplateNotFound:
        html_fallback = """
        <html><head><meta charset="utf-8"><title>NMF 控制台</title></head>
        <body style="font-family: sans-serif; max-width: 900px; margin: 40px auto;">
          <h2>NMF 控制台（占位页）</h2>
          <p>建议创建 <code>templates/nmf.html</code> 自定义你的界面。</p>
        </body></html>
        """
        return html_fallback


# -------------------------------------------------------------
# A. 频率/谷点识别
# -------------------------------------------------------------
@app.route("/freq_valley_analysis", methods=["POST"])
def freq_valley_analysis():
    try:
        if "intensity_matrix_file" not in request.files:
            return jsonify({"error": "请上传 Step⑥ 强度矩阵 CSV（字段名 intensity_matrix_file）"}), 400
        f = request.files["intensity_matrix_file"]
        if not f.filename:
            return jsonify({"error": "CSV 文件不能为空"}), 400

        print("[DEBUG] 收到参数：", request.form.to_dict())
        print("[DEBUG] 上传文件：", request.files)

        # 参数
        bin_size_da = _safe_float(request.form.get("bin_size_da", 20), 20)
        top_quantile = _safe_float(request.form.get("bin20_quantile", 0.999), 0.999)
        spline_s = _safe_float(request.form.get("spline_s", 0.05), 0.05)

        upload_dir = ensure_dir(app.config["UPLOAD_FOLDER"])
        out_dir = ensure_dir(app.config["OUTPUT_FOLDER"])

        step6_csv = os.path.join(upload_dir, _stamp("uploaded_step6_matrix", ".csv"))
        f.save(step6_csv)

        # ---------- 自动识别输入 ----------
        mode, obj = read_intensity_or_frequency(step6_csv)
        print(f"[DEBUG] 识别类型: {mode}, 维度: {obj.shape}")

        # 输出文件（时间戳，避免被覆盖）
        freq_out = os.path.join(out_dir, _stamp("step6_transformation_frag_frequency_corrected", ".csv"))
        freq_kept_out = os.path.join(out_dir, _stamp("step6_transformation_frag_frequency_filtered", ".csv"))
        valleys_out = os.path.join(out_dir, _stamp("step6_transformation_mz_valleys", ".csv"))
        plot_png = os.path.join(out_dir, _stamp("frag_frequency_valleys_plot", ".png"))

        # Step 1: 频率表
        if mode == "frequency":
            frequency_df = obj.copy()
            # 保底：缺了 Frequency_Count 就用 Rate*样本数估（未知样本数则留空）
            if "Frequency_Count" not in frequency_df.columns:
                frequency_df["Frequency_Count"] = np.nan
            frequency_df = frequency_df[["mz", "Frequency_Count", "Frequency_Rate"]]
        else:
            inten_df = obj.copy()
            counts = (inten_df > 0).sum(axis=0)
            rates = counts / max(inten_df.shape[0], 1)
            frequency_df = pd.DataFrame({
                "mz": counts.index.astype(float),
                "Frequency_Count": counts.values,
                "Frequency_Rate": rates.values
            })
        # 清洗 + 排序
        frequency_df = frequency_df.dropna(subset=["mz"]).copy()
        frequency_df["mz"] = pd.to_numeric(frequency_df["mz"], errors="coerce")
        frequency_df["Frequency_Rate"] = pd.to_numeric(frequency_df["Frequency_Rate"], errors="coerce")
        frequency_df = frequency_df.dropna(subset=["mz", "Frequency_Rate"]).sort_values("mz")
        frequency_df.to_csv(freq_out, index=False)

        # Step 2: 分箱 + 分位选择
        frequency_df["mz_bin"] = (frequency_df["mz"] // bin_size_da).astype(int)
        kept_list = []
        for _, bdf in frequency_df.groupby("mz_bin"):
            if bdf.empty:
                continue
            thr = bdf["Frequency_Rate"].quantile(top_quantile)
            kept_list.append(bdf[bdf["Frequency_Rate"] >= thr])
        kept_df = pd.concat(kept_list).sort_values("mz") if kept_list else pd.DataFrame(columns=frequency_df.columns)
        kept_df.to_csv(freq_kept_out, index=False)

        # Step 3: 样条 + 谷点（去重、保序）
        valley_mz_list = []
        if kept_df.empty:
            valleys = pd.DataFrame(columns=["mz", "Frequency_Rate"])
            valleys.to_csv(valleys_out, index=False)
            plt.figure(figsize=(10, 6))
            plt.title("Smoothed Frequency Curve with Valleys (no points)")
            plt.tight_layout(); plt.savefig(plot_png, dpi=300); plt.close()
        else:
            xy = kept_df[["mz", "Frequency_Rate"]].dropna().sort_values("mz")
            # 同一 m/z 去重聚合（中位数），避免样条报错
            xy = xy.groupby("mz", as_index=False)["Frequency_Rate"].median().sort_values("mz")
            print(f"[DEBUG] 保留点数（唯一 m/z）: {len(xy)}")

            if len(xy) < 4:
                valleys = pd.DataFrame(columns=["mz", "Frequency_Rate"])
                valleys.to_csv(valleys_out, index=False)
                plt.figure(figsize=(10, 6))
                plt.plot(xy["mz"], xy["Frequency_Rate"], "o", label="Retained Points", markersize=3)
                plt.title("Smoothed Frequency Curve with Valleys (insufficient points)")
                plt.tight_layout(); plt.savefig(plot_png, dpi=300); plt.close()
            else:
                x = xy["mz"].values
                y = xy["Frequency_Rate"].values
                try:
                    spline = UnivariateSpline(x, y, s=float(spline_s))
                    xs = np.linspace(x.min(), x.max(), 2000)
                    ys = spline(xs)
                    valley_idx = argrelextrema(ys, np.less)[0]
                    valleys = pd.DataFrame({"mz": xs[valley_idx], "Frequency_Rate": ys[valley_idx]})
                    valleys.to_csv(valleys_out, index=False)
                    valley_mz_list = valleys["mz"].round(4).tolist()

                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y, "o", label="Retained Points", markersize=3)
                    plt.plot(xs, ys, "-", label="Smoothed Fit")
                    if len(valley_idx) > 0:
                        plt.plot(xs[valley_idx], ys[valley_idx], "ro", label="Valleys")
                    plt.xlabel("m/z"); plt.ylabel("Frequency Rate")
                    plt.title("Smoothed Frequency Curve with Valleys")
                    plt.legend(); plt.tight_layout()
                    plt.savefig(plot_png, dpi=300); plt.close()
                except Exception as ee:
                    print("[WARN] 样条拟合失败，退化为仅保存散点：", ee)
                    valleys = pd.DataFrame(columns=["mz", "Frequency_Rate"])
                    valleys.to_csv(valleys_out, index=False)
                    plt.figure(figsize=(10, 6))
                    plt.plot(x, y, "o", label="Retained Points", markersize=3)
                    plt.title("Spline failed; plotted points only")
                    plt.tight_layout(); plt.savefig(plot_png, dpi=300); plt.close()

        return jsonify({"result": {
            "step6_transformation_frag_frequency_corrected_csv_filename": os.path.basename(freq_out),
            "step6_transformation_frag_frequency_corrected_csv_url": _dl(freq_out),
            "step6_transformation_frag_frequency_filtered_csv_filename": os.path.basename(freq_kept_out),
            "step6_transformation_frag_frequency_filtered_csv_url": _dl(freq_kept_out),
            "step6_transformation_mz_valleys_csv_filename": os.path.basename(valleys_out),
            "step6_transformation_mz_valleys_csv_url": _dl(valleys_out),
            "frag_frequency_valleys_plot_png_filename": os.path.basename(plot_png),
            "frag_frequency_valleys_plot_png_url": _dl(plot_png),
            "valley_mz_list": valley_mz_list,
        }})
    except Exception as e:
        print("[ERROR] /freq_valley_analysis:", traceback.format_exc())
        return jsonify({"error": f"频率/谷点识别失败：{e}"}), 500


# -------------------------------------------------------------
# B. 固定阈值分段筛选
# -------------------------------------------------------------
@app.route("/segmented_frequency_filter", methods=["POST"])
def segmented_frequency_filter():
    try:
        if "intensity_matrix_file" not in request.files:
            return jsonify({"error": "请上传 Step⑥ 强度矩阵 CSV（字段名 intensity_matrix_file）"}), 400
        f = request.files["intensity_matrix_file"]
        if not f.filename:
            return jsonify({"error": "CSV 文件不能为空"}), 400

        mz_split_threshold = _safe_float(request.form.get("mz_split_threshold", 300), 300)
        high_freq_rate = _safe_float(request.form.get("high_freq_rate", 0.01), 0.01)
        low_freq_rate = _safe_float(request.form.get("low_freq_rate", 0.1), 0.1)

        upload_dir = ensure_dir(app.config["UPLOAD_FOLDER"])
        out_dir = ensure_dir(app.config["OUTPUT_FOLDER"])
        step6_csv = os.path.join(upload_dir, _stamp("uploaded_step6_matrix", ".csv"))
        f.save(step6_csv)

        mode, data = read_intensity_or_frequency(step6_csv)

        freq_out = os.path.join(out_dir, _stamp("step6_transformation_frag_frequency_corrected", ".csv"))
        final_out = os.path.join(out_dir, _stamp("final_filtered_matrix", ".csv"))

        if mode == "frequency":
            freq_df = data[["mz", "Frequency_Count", "Frequency_Rate"]].copy()
            freq_df = freq_df.dropna(subset=["mz", "Frequency_Rate"])
            freq_df["mz"] = pd.to_numeric(freq_df["mz"], errors="coerce")
            freq_df["Frequency_Rate"] = pd.to_numeric(freq_df["Frequency_Rate"], errors="coerce")
            freq_df = freq_df.dropna(subset=["mz", "Frequency_Rate"]).sort_values("mz")
        else:
            inten_df = data.copy()
            counts = (inten_df > 0).sum(axis=0)
            rates = counts / max(inten_df.shape[0], 1)
            freq_df = pd.DataFrame({"mz": counts.index.astype(float),
                                    "Frequency_Count": counts.values,
                                    "Frequency_Rate": rates.values}).sort_values("mz")

        freq_df.to_csv(freq_out, index=False)

        sel = freq_df[
            ((freq_df["mz"] < mz_split_threshold) & (freq_df["Frequency_Rate"] > low_freq_rate)) |
            ((freq_df["mz"] >= mz_split_threshold) & (freq_df["Frequency_Rate"] > high_freq_rate))
        ]["mz"].tolist()

        if mode == "frequency":
            # 无法返回强度矩阵；写出说明文件避免 404
            pd.DataFrame({"note": ["上传的是频率表，未提供强度矩阵，无法导出筛选后的强度矩阵。"],
                          "kept_mz_count": [len(sel)]}).to_csv(final_out, index=False)
        else:
            kept = data.loc[:, data.columns.isin(sel)].sort_index(axis=1)
            kept.to_csv(final_out)

        return jsonify({"result": {
            "final_filtered_matrix_csv_filename": os.path.basename(final_out),
            "final_filtered_matrix_csv_url": _dl(final_out),
        }})
    except Exception as e:
        print("[ERROR] /segmented_frequency_filter:", traceback.format_exc())
        return jsonify({"error": f"分段筛选失败：{e}"}), 500


# -------------------------------------------------------------
# C. 数据矩阵构建（MGF→Step①–⑥）
# -------------------------------------------------------------
@app.route("/process_matrix_builder", methods=["POST"])
def process_matrix_builder():
    try:
        # 1) 基本校验
        if "mgf_file" not in request.files:
            return jsonify({"error": "请上传 MGF 文件（字段名 mgf_file）"}), 400
        mgf = request.files["mgf_file"]
        if not mgf.filename:
            return jsonify({"error": "MGF 文件不能为空"}), 400

        # 2) 读取参数（与 processor 一致；其余可选阈值需要时同理取出后传入）
        bin_size = _safe_float(request.form.get("bin_size", 0.01), 0.01)
        tolerance = _safe_float(request.form.get("tolerance", 0.02), 0.02)

        # 3) 保存上传 & 准备输出目录
        up = ensure_dir(app.config["UPLOAD_FOLDER"])
        out = ensure_dir(app.config["OUTPUT_FOLDER"])
        mgf_path = os.path.join(up, mgf.filename)
        mgf.save(mgf_path)

        # 4) 调用真正的 Step①–⑦ 流程（返回绝对路径字典）
        results = mb_process(
            mgf_path=mgf_path,
            output_dir=out,
            bin_size=bin_size,
            tolerance=tolerance,
            #如需自定义其它阈值，可开启以下参数并从 form 读取：
            mz_split_threshold=_safe_float(request.form.get("mz_split_threshold", 300), 300),
            high_freq_rate=_safe_float(request.form.get("high_freq_rate", 0.01), 0.01),
            low_freq_rate=_safe_float(request.form.get("low_freq_rate", 0.1), 0.1),
            bin20_quantile=_safe_float(request.form.get("bin20_quantile", 0.999), 0.999),
            spline_s=_safe_float(request.form.get("spline_s", 0.05), 0.05),
        )

        # 5) 将绝对路径映射为 *_filename（供前端 renderFiles 使用）+ *_url（可直接点击）
        payload = {}
        for key, abs_path in results.items():
            fname = os.path.basename(abs_path)
            payload[key + "_filename"] = fname
            payload[key + "_url"] = _dl(abs_path)

        return jsonify({"result": payload})
    except Exception as e:
        print("[ERROR] /process_matrix_builder:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500



# -------------------------------------------------------------
# D. NMF 评估
# -------------------------------------------------------------
@app.route("/process_nmf_evaluate", methods=["POST"])
@app.route("/nmf/evaluate", methods=["POST"])
def route_nmf_evaluate():
    try:
        if "matrix_file" not in request.files:
            return jsonify({"error": "请上传数据矩阵文件（字段名 matrix_file）"}), 400
        matrix_file = request.files["matrix_file"]
        if not matrix_file.filename:
            return jsonify({"error": "矩阵文件不能为空"}), 400

        min_components = int(request.form.get("min_components", 2))
        max_components = int(request.form.get("max_components", 20))
        step = int(request.form.get("step", 1))

        up = ensure_dir(app.config["UPLOAD_FOLDER"])
        out = ensure_dir(app.config["OUTPUT_FOLDER"])
        matrix_path = os.path.join(up, matrix_file.filename)
        matrix_file.save(matrix_path)

        eva = NMFEvaluator(matrix_path, out)
        eva.load_data()
        eva.evaluate(max_components=max_components, min_components=min_components, step=step)

        return jsonify({"result": {
            "reconstruction_error_png_filename": "reconstruction_error.png",
            "explained_variance_png_filename": "explained_variance.png",
        }})
    except Exception as e:
        print("[ERROR] /nmf/evaluate:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------
# E. CNMF 分解
# -------------------------------------------------------------
@app.route("/process_nmf_decomposition", methods=["POST"])
@app.route("/nmf/decompose", methods=["POST"])
def route_nmf_decomposition():
    try:
        if "matrix_file" not in request.files:
            return jsonify({"error": "请上传原始矩阵文件（字段名 matrix_file）"}), 400
        matrix_file = request.files["matrix_file"]
        if not matrix_file.filename:
            return jsonify({"error": "矩阵文件不能为空"}), 400

        n_components = int(request.form.get("n_components", 11))

        up = ensure_dir(app.config["UPLOAD_FOLDER"])
        out = ensure_dir(app.config["OUTPUT_FOLDER"])
        matrix_path = os.path.join(up, matrix_file.filename)
        matrix_file.save(matrix_path)

        # 读取矩阵
        mat_df = pd.read_csv(matrix_path, index_col=0)
        X = mat_df.values
        np.random.seed(42)
        W_init = np.abs(np.random.rand(X.shape[0], n_components))
        H_init = np.abs(np.random.rand(n_components, X.shape[1]))

        model = NMF(n_components=n_components, init="custom", random_state=42, max_iter=1000)
        W = model.fit_transform(X, W=W_init, H=H_init)
        H = model.components_

        comp_names = [f"Component_{i+1}" for i in range(n_components)]
        W_df = pd.DataFrame(W, index=mat_df.index, columns=comp_names)
        H_df = pd.DataFrame(H, index=comp_names, columns=mat_df.columns)

        w_file = os.path.join(out, f"CNMF_W_matrix_n{n_components}.csv")
        h_file = os.path.join(out, f"CNMF_H_matrix_n{n_components}.csv")
        W_df.to_csv(w_file); H_df.to_csv(h_file)

        # top features per component
        tops = []
        for c in comp_names:
            top_mz = H_df.loc[c].nlargest(15).index.tolist()
            for mz in top_mz:
                tops.append([c, mz])
        top_df = pd.DataFrame(tops, columns=["Component", "Top_m/z"])
        top_df.to_csv(os.path.join(out, f"top_features_per_component-{n_components}.csv"), index=False)

        # 层次聚类 + 删除低贡献
        Z = linkage(W_df, method="ward", metric="euclidean")
        order = leaves_list(Z)
        W_sorted = W_df.iloc[order, :]
        low_mask = W_sorted.sum(axis=1) < 0.01
        W_filtered = W_sorted[~low_mask]

        palette = [
            "#E46A6A", "#64B5F6", "#81C784", "#FFD54F",
            "#C37BCF", "#4DB6AC", "#F27AA2", "#FCBB74",
            "#A1887F", "#7D949F", "#7986CB", "#DCE775", "#B3E5FC", "#FF8A80", "#E0F7FA"
        ]
        color_map = {c: palette[i % len(palette)] for i, c in enumerate(comp_names)}

        # H 折线
        plt.figure(figsize=(12, 6), dpi=300)
        for c in comp_names:
            plt.plot(H_df.columns, H_df.loc[c, :], label=c, linewidth=1.2, alpha=0.8, color=color_map[c])
        plt.xlabel("m/z（原始顺序）"); plt.ylabel("Component Weight"); plt.title("CNMF 成分与 m/z 关系")
        xtick_int = max(len(H_df.columns) // 15, 1)
        plt.xticks(range(0, len(H_df.columns), xtick_int), H_df.columns[::xtick_int], rotation=45, fontsize=8)
        plt.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        h_png = os.path.join(out, f"CNMF_H_Matrix_Visualization_n{n_components}.png")
        plt.savefig(h_png, dpi=300, bbox_inches="tight"); plt.close()

        # W 堆叠条形
        plt.figure(figsize=(14, 6), dpi=300)
        W_norm = W_filtered.div(W_filtered.sum(axis=1), axis=0)
        ax = W_norm.plot(kind="bar", stacked=True, width=0.8,
                         color=[color_map[c] for c in W_df.columns], figsize=(14, 6))
        ax.set_xlabel("样本（已聚类）"); ax.set_ylabel("成分贡献比例"); ax.set_title("样本与成分的关系（聚类后）")
        xtick_int2 = max(len(W_norm) // 20, 1)
        ax.set_xticks(range(0, len(W_norm), xtick_int2))
        ax.set_xticklabels(W_norm.index[::xtick_int2], rotation=45, fontsize=8)
        ax.legend(title="Component", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        w_png = os.path.join(out, f"CNMF_W_Matrix_Visualization_n{n_components}.png")
        plt.savefig(w_png, dpi=300, bbox_inches="tight"); plt.close()

        return jsonify({"result": {
            "w_matrix_csv_filename": os.path.basename(w_file),
            "h_matrix_csv_filename": os.path.basename(h_file),
            "h_matrix_plot_png_filename": os.path.basename(h_png),
            "w_matrix_plot_png_filename": os.path.basename(w_png),
            "top_features_csv_filename": os.path.basename(f"top_features_per_component-{n_components}.csv")
        }})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------
# F. UMAP 可视化预处理（加载 W、元数据、GNPS）—— 路由 + 别名
# -------------------------------------------------------------
@app.route("/process_nmf_processor", methods=["POST"])
@app.route("/nmf/process", methods=["POST"])
def route_nmf_processor():
    try:
        if "w_matrix_file" not in request.files:
            return jsonify({"error": "请上传 W 矩阵 CSV（字段名 w_matrix_file）"}), 400
        wfile = request.files["w_matrix_file"]
        if not wfile.filename:
            return jsonify({"error": "W 矩阵文件不能为空"}), 400

        metadata_path = None
        if "metadata_file" in request.files and request.files["metadata_file"].filename:
            mfile = request.files["metadata_file"]
            metadata_path = os.path.join(ensure_dir(app.config["UPLOAD_FOLDER"]), mfile.filename)
            mfile.save(metadata_path)

        gnps_path = None
        if "gnps_file" in request.files and request.files["gnps_file"].filename:
            gfile = request.files["gnps_file"]
            gnps_path = os.path.join(ensure_dir(app.config["UPLOAD_FOLDER"]), gfile.filename)
            gfile.save(gnps_path)

        n_components = int(request.form.get("n_components", 11))
        up = ensure_dir(app.config["UPLOAD_FOLDER"])
        out = ensure_dir(app.config["OUTPUT_FOLDER"])
        w_path = os.path.join(up, wfile.filename)
        wfile.save(w_path)

        proc = NMFProcessor(w_matrix_file=w_path, metadata_file=metadata_path,
                            gnps_graphml_path=gnps_path, output_dir=out, n_components=n_components)
        proc.process_umap(n_neighbors=10, min_dist=1.0, random_state=42)

        GLOBAL_VIS["processor"] = proc  # 给 Dash 用

        return jsonify({"result": {
            "visualization_url": "/umap_visualization/",
            "nodes_info": f"all_nodes_info-{n_components}.csv",
            "edges_df": "edges_df.csv"
        }})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------
# G. UMAP 可视化（可调 n_neighbors、min_dist）—— 路由 + 别名
# -------------------------------------------------------------
@app.route("/process_nmf_visualization", methods=["POST"])
@app.route("/nmf/visualize", methods=["POST"])
def route_nmf_visualization():
    try:
        if "w_matrix_file" not in request.files:
            return jsonify({"error": "请上传 W 矩阵 CSV（字段名 w_matrix_file）"}), 400
        wfile = request.files["w_matrix_file"]
        if not wfile.filename:
            return jsonify({"error": "W 矩阵文件不能为空"}), 400

        metadata_path = None
        if "metadata_file" in request.files and request.files["metadata_file"].filename:
            mfile = request.files["metadata_file"]
            metadata_path = os.path.join(ensure_dir(app.config["UPLOAD_FOLDER"]), mfile.filename)
            mfile.save(metadata_path)

        gnps_path = None
        if "gnps_file" in request.files and request.files["gnps_file"].filename:
            gfile = request.files["gnps_file"]
            gnps_path = os.path.join(ensure_dir(app.config["UPLOAD_FOLDER"]), gfile.filename)
            gfile.save(gnps_path)

        n_neighbors = int(request.form.get("n_neighbors", 10))
        min_dist = float(request.form.get("min_dist", 1.0))
        random_state = int(request.form.get("random_state", 42))
        n_components = int(request.form.get("n_components", 11))

        up = ensure_dir(app.config["UPLOAD_FOLDER"])
        out = ensure_dir(app.config["OUTPUT_FOLDER"])
        w_path = os.path.join(up, wfile.filename)
        wfile.save(w_path)

        proc = NMFProcessor(w_matrix_file=w_path, metadata_file=metadata_path,
                            gnps_graphml_path=gnps_path, output_dir=out, n_components=n_components)
        proc.process_umap(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)

        GLOBAL_VIS["processor"] = proc

        return jsonify({"result": {
            "visualization_url": "/umap_visualization/",
            "nodes_info": f"all_nodes_info-{n_components}.csv",
            "edges_df": "edges_df.csv"
        }})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------------------
# 下载
# -------------------------------------------------------------
@app.route("/download/<path:filename>")
def download_file(filename):
    out = ensure_dir(app.config["OUTPUT_FOLDER"])
    file_path = os.path.join(out, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "文件不存在"}), 404
    return send_from_directory(out, filename, as_attachment=True)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------
if __name__ == "__main__":
    # Windows 中文字体/负号（可选）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    app.run(host="0.0.0.0", port=5000, debug=True)


