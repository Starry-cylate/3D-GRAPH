import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from scipy import integrate

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_numpy_function(expr_str, variables):
    try:
        expr = sp.sympify(expr_str)
    except Exception as e:
        return None, f"表达式解析失败: {e}"
    symbols = [sp.Symbol(v) for v in variables]
    try:
        return sp.lambdify(symbols, expr, modules=["numpy", "math"]), None
    except Exception as e:
        return None, f"函数编译失败: {e}"


# ---------------------------------------------------------------------------
# 页面配置 & CSS（仿 GeoGebra 视觉风格）
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="微积分多维可视化",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* === 全局 === */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Microsoft YaHei', 'PingFang SC', sans-serif;
        color: #2c2c2c;
    }

    /* === 主标题 === */
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin: 0;
        padding: 0.4rem 0 0 0;
        letter-spacing: 1px;
        border-bottom: 2px solid #e0e0e0;
    }

    /* === 侧边栏：代数区 === */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #ddd;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.4rem;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.25rem;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-size: 0.88rem;
        font-weight: 500;
        color: #444;
    }
    /* 侧边栏 section 分隔 */
    .sidebar-section {
        border: 1px solid #e8e8e8;
        border-radius: 8px;
        padding: 0.8rem 0.9rem 0.5rem 0.9rem;
        margin-bottom: 0.8rem;
        background: #fafafa;
    }
    .sidebar-section h4 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #555;
        margin: 0 0 0.6rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* === 输入框（仿 GeoGebra 代数条） === */
    input[type="text"] {
        border: 1px solid #bbb !important;
        border-radius: 4px !important;
        padding: 0.4rem 0.7rem !important;
        font-family: 'Consolas', 'Fira Code', 'Courier New', monospace !important;
        font-size: 0.9rem !important;
        background: #fff !important;
        transition: border-color 0.2s !important;
    }
    input[type="text"]:focus {
        border-color: #1A5F7A !important;
        box-shadow: 0 0 0 2px rgba(26, 95, 122, 0.12) !important;
    }

    [data-testid="stNumberInput"] input {
        border: 1px solid #bbb !important;
        border-radius: 4px !important;
        background: #fff !important;
        font-size: 0.88rem !important;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: #1A5F7A !important;
        box-shadow: 0 0 0 2px rgba(26, 95, 122, 0.12) !important;
    }

    /* === 结果卡片 === */
    .result-card {
        background: #f5f7fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 0.8rem 0 0 0;
    }
    .result-card .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #888;
        margin-bottom: 0.2rem;
    }
    .result-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1A5F7A;
        font-variant-numeric: tabular-nums;
    }

    /* === Latex 公式居中 === */
    .latex-container {
        text-align: center;
        margin: 0.4rem 0 0.2rem 0;
    }

    /* === divider === */
    hr {
        margin: 0.4rem 0;
        border-color: #e8e8e8;
    }
</style>
""", unsafe_allow_html=True)

# ---- 全局标题 ----
st.markdown('<div class="app-title">微积分多维可视化</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 侧边栏：代数区（GeoGebra 风格）
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 代数区")

    dimension = st.radio(
        "视图",
        options=["一元函数积分 (2D 面积)", "二元函数积分 (3D 体积)"],
        label_visibility="collapsed",
    )

    st.markdown("---")

# ====================================================================
# 一元函数积分 (2D)
# ====================================================================
if dimension == "一元函数积分 (2D 面积)":

    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h4>函数</h4>', unsafe_allow_html=True)

        func_str = st.text_input(
            "f(x) =",
            value="x**2 - 2*x + 2",
            key="1d_func",
            label_visibility="collapsed",
        )
        st.caption("例: sin(x), exp(x), x**2 - 2*x + 2")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><h4>积分区间 [a, b]</h4>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            a = st.number_input("下界 a", value=0.0, step=0.5, key="1d_a")
        with c2:
            b = st.number_input("上界 b", value=3.0, step=0.5, key="1d_b")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 主区域：几何绘图区 ----------
    f, err = make_numpy_function(func_str, ["x"])

    if err:
        st.error(err)
    elif a >= b:
        st.warning("积分上界 b 必须大于下界 a。")
    else:
        integral_val, _ = integrate.quad(f, a, b)

        # ---- 公式展示：独立在图表上方 ----
        st.latex(
            r"\int_{%s}^{%s} \left(%s\right) \, dx = %.6f"
            % (f"{a:.4g}", f"{b:.4g}", func_str, integral_val)
        )

        # ---- 构建 2D 图表 ----
        margin = max((b - a) * 0.3, 0.5)
        x_min = a - margin
        x_max_ = b + margin

        x_dense = np.linspace(x_min, x_max_, 600)
        y_dense = f(x_dense)
        if np.isscalar(y_dense):
            y_dense = np.full_like(x_dense, y_dense)

        x_fill = np.linspace(a, b, 400)
        y_fill = f(x_fill)
        if np.isscalar(y_fill):
            y_fill = np.full_like(x_fill, y_fill)

        # 计算 y 范围，让 x 轴居中
        y_all = np.concatenate([y_dense, [0]])
        y_pad = max((np.max(y_all) - np.min(y_all)) * 0.15, 0.5)
        y_min_range = min(np.min(y_all) - y_pad, -0.5)
        y_max_range = max(np.max(y_all) + y_pad,  0.5)

        fig = go.Figure()

        # 积分填充区域
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fill, x_fill[::-1]]),
            y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
            fill="toself",
            fillcolor="rgba(26, 95, 122, 0.18)",
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ))

        # 函数曲线 — GeoGebra 经典深蓝
        fig.add_trace(go.Scatter(
            x=x_dense,
            y=y_dense,
            mode="lines",
            line=dict(color="#1A5F7A", width=2.6),
            hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra></extra>",
            showlegend=False,
        ))

        # 积分边界虚线 + 端点标记
        for boundary, color in [(a, "#CC3333"), (b, "#339933")]:
            fb = float(f(boundary))

            # 虚线
            y0 = min(0, fb)
            y1 = max(0, fb)
            fig.add_trace(go.Scatter(
                x=[boundary, boundary],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=1.6, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ))
            # 端点圆点
            fig.add_trace(go.Scatter(
                x=[boundary, boundary],
                y=[0, fb],
                mode="markers",
                marker=dict(size=7, color=color, line=dict(width=1.5, color="white")),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.update_xaxes(
            title="x",
            range=[x_min, x_max_],
            showgrid=True, gridcolor="#E8E8E8", gridwidth=1,
            zeroline=True, zerolinecolor="#333", zerolinewidth=1.8,
            showline=True, linecolor="#333", linewidth=1.5,
            mirror=False,
        )
        fig.update_yaxes(
            title="f(x)",
            range=[y_min_range, y_max_range],
            showgrid=True, gridcolor="#E8E8E8", gridwidth=1,
            zeroline=True, zerolinecolor="#333", zerolinewidth=1.8,
            showline=True, linecolor="#333", linewidth=1.5,
            mirror=False,
            scaleanchor="x",
            scaleratio=1,
        )

        fig.update_layout(
            showlegend=False,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            font=dict(
                family="Inter, Microsoft YaHei, PingFang SC, sans-serif",
                size=13,
                color="#333",
            ),
            xaxis=dict(constrain="domain"),
            yaxis=dict(constrain="domain"),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # 数值结果卡片
        st.markdown(f"""
        <div class="result-card">
            <div class="label">积分结果</div>
            <div class="value">{integral_val:.8f}</div>
        </div>
        """, unsafe_allow_html=True)

# ====================================================================
# 二元函数积分 (3D)
# ====================================================================
else:

    with st.sidebar:
        st.markdown('<div class="sidebar-section"><h4>函数</h4>', unsafe_allow_html=True)

        func_str = st.text_input(
            "f(x, y) =",
            value="sin(x) * cos(y) + 2",
            key="2d_func",
            label_visibility="collapsed",
        )
        st.caption("例: sin(x)*cos(y) + 2, x**2 + y**2")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><h4>X 轴范围</h4>', unsafe_allow_html=True)
        cx1, cx2 = st.columns(2)
        with cx1:
            x_min = st.number_input("下界", value=-np.pi, step=0.5, key="2d_xmin")
        with cx2:
            x_max = st.number_input("上界", value=np.pi, step=0.5, key="2d_xmax")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section"><h4>Y 轴范围</h4>', unsafe_allow_html=True)
        cy1, cy2 = st.columns(2)
        with cy1:
            y_min = st.number_input("下界", value=-np.pi, step=0.5, key="2d_ymin")
        with cy2:
            y_max = st.number_input("上界", value=np.pi, step=0.5, key="2d_ymax")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 主区域：几何绘图区 ----------
    f, err = make_numpy_function(func_str, ["x", "y"])

    if err:
        st.error(err)
    elif x_min >= x_max or y_min >= y_max:
        st.warning("积分上界必须大于下界。")
    else:
        N = 80
        x_grid = np.linspace(x_min, x_max, N)
        y_grid = np.linspace(y_min, y_max, N)
        X, Y = np.meshgrid(x_grid, y_grid)

        try:
            Z = f(X, Y)
            if np.isscalar(Z):
                Z = np.full_like(X, Z)
        except Exception as e:
            st.error(f"函数求值失败: {e}")
            st.stop()

        integral_val, _ = integrate.dblquad(
            lambda y, x: f(x, y),
            y_min, y_max,
            lambda y: x_min, lambda y: x_max,
        )

        # ---- 公式展示 ----
        st.latex(
            r"\iint_{[%s,%s] \times [%s,%s]} \left(%s\right) \, dx \, dy = %.6f"
            % (f"{x_min:.4g}", f"{x_max:.4g}", f"{y_min:.4g}", f"{y_max:.4g}", func_str, integral_val)
        )

        # ---- 构建 3D 图表 ----
        fig = go.Figure()

        # 顶面
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0.0, "#1A5F7A"],
                [0.3, "#2E8B8B"],
                [0.6, "#5B9BD5"],
                [0.8, "#8CB5E0"],
                [1.0, "#C5D9F0"],
            ],
            opacity=0.90,
            showscale=True,
            colorbar=dict(
                title="",
                thickness=14,
                len=0.5,
                outlinewidth=0,
                tickfont=dict(size=11, color="#555"),
            ),
            contours=dict(
                z=dict(
                    show=True,
                    usecolormap=True,
                    highlightcolor="rgba(255,255,255,0.6)",
                    project=dict(z=True),
                    width=1,
                ),
            ),
            showlegend=False,
        ))

        # 底面
        Z_bottom = np.zeros_like(Z)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_bottom,
            colorscale=[[0, "#e0e0e0"], [1, "#e0e0e0"]],
            opacity=0.30,
            showscale=False,
            showlegend=False,
        ))

        # 四个垂直侧壁
        Y_side, T = np.meshgrid(y_grid, np.linspace(0, 1, 30))
        X_side_g, T_g = np.meshgrid(x_grid, np.linspace(0, 1, 30))

        side_defs = [
            (x_min,  "y", Y_side, T,   "#1A5F7A"),
            (x_max,  "y", Y_side, T,   "#CC3333"),
            (y_min,  "x", X_side_g, T_g, "#339933"),
            (y_max,  "x", X_side_g, T_g, "#D4A017"),
        ]

        for val, axis, grid_a, grid_t, color in side_defs:
            if axis == "y":
                X_side = np.full_like(grid_a, val)
                Y_side = grid_a
                f_vals = f(val, grid_a)
            else:
                X_side = grid_a
                Y_side = np.full_like(grid_a, val)
                f_vals = f(grid_a, val)

            if np.isscalar(f_vals):
                f_vals = np.full_like(grid_a, f_vals)
            Z_side = f_vals * grid_t

            fig.add_trace(go.Surface(
                x=X_side, y=Y_side, z=Z_side,
                colorscale=[[0, color], [1, color]],
                opacity=0.38,
                showscale=False,
                showlegend=False,
            ))

        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(
                    title="x",
                    gridcolor="#E8E8E8",
                    backgroundcolor="rgba(255,255,255,1)",
                    zerolinecolor="#333",
                    showline=True,
                    linecolor="#333",
                    linewidth=1.2,
                ),
                yaxis=dict(
                    title="y",
                    gridcolor="#E8E8E8",
                    backgroundcolor="rgba(255,255,255,1)",
                    zerolinecolor="#333",
                    showline=True,
                    linecolor="#333",
                    linewidth=1.2,
                ),
                zaxis=dict(
                    title="f(x, y)",
                    gridcolor="#E8E8E8",
                    backgroundcolor="rgba(255,255,255,1)",
                    zerolinecolor="#333",
                    showline=True,
                    linecolor="#333",
                    linewidth=1.2,
                ),
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.0)),
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(255,255,255,1)",
            font=dict(
                family="Inter, Microsoft YaHei, PingFang SC, sans-serif",
                size=13,
                color="#333",
            ),
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        # 数值结果卡片
        st.markdown(f"""
        <div class="result-card">
            <div class="label">积分结果</div>
            <div class="value">{integral_val:.8f}</div>
        </div>
        """, unsafe_allow_html=True)
