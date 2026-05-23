import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from scipy import integrate

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_numpy_function(expr_str, variables):
    """将用户输入的表达式字符串编译为可调用的 Python 函数。"""
    try:
        expr = sp.sympify(expr_str)
    except Exception as e:
        return None, f"表达式解析失败: {e}"

    symbols = [sp.Symbol(v) for v in variables]
    try:
        f_lambdified = sp.lambdify(symbols, expr, modules=["numpy", "math"])
    except Exception as e:
        return None, f"函数编译失败: {e}"

    return f_lambdified, None


# ---------------------------------------------------------------------------
# 自定义 CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="微积分多维可视化",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* ---- 全局 ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Microsoft YaHei', sans-serif;
    }

    /* ---- 主标题渐变 ---- */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.6rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    .main-subtitle {
        text-align: center;
        color: #8e8ea0;
        font-size: 0.95rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ---- 结果卡片 ---- */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.6rem 2rem;
        margin: 1.2rem 0 0.5rem 0;
        color: #fff;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
    }
    .result-card .label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.85;
        margin-bottom: 0.3rem;
    }
    .result-card .value {
        font-size: 2.2rem;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.5px;
    }
    .result-card .formula {
        font-size: 0.9rem;
        opacity: 0.75;
        margin-top: 0.2rem;
    }

    /* ---- 侧边栏美化 ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fc 0%, #eef0f7 100%);
        border-right: 1px solid #e0e3eb;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding: 0.6rem 1rem;
        border-radius: 10px;
        transition: all 0.2s ease;
        font-weight: 500;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(102, 126, 234, 0.08);
    }

    /* ---- 输入框 ---- */
    input[type="text"] {
        border: 2px solid #e0e3eb !important;
        border-radius: 10px !important;
        padding: 0.6rem 1rem !important;
        font-family: 'Consolas', 'Fira Code', 'Courier New', monospace !important;
        font-size: 0.95rem !important;
        transition: border-color 0.25s ease !important;
    }
    input[type="text"]:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }

    /* ---- 数字输入框 ---- */
    [data-testid="stNumberInput"] input {
        border: 2px solid #e0e3eb !important;
        border-radius: 10px !important;
        transition: border-color 0.25s ease !important;
    }
    [data-testid="stNumberInput"] input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.15) !important;
    }

    /* ---- 提示卡片 ---- */
    .tip-card {
        background: #f8f9fc;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        font-size: 0.82rem;
        color: #5a5f72;
    }
    .tip-card strong {
        color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# ---- 顶部标题 ----
st.markdown('<div class="main-title">微积分多维可视化</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="main-subtitle">交互式定积分 & 二重积分几何直观演示</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 侧边栏
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ 参数配置")
    st.markdown("---")

    dimension = st.radio(
        "选择积分维度",
        options=["一元函数积分 (2D 面积)", "二元函数积分 (3D 体积)"],
        label_visibility="collapsed",
    )

    st.markdown("---")

# ====================================================================
# 一元函数积分 (2D)
# ====================================================================
if dimension == "一元函数积分 (2D 面积)":

    with st.sidebar:
        st.markdown("#### 📐 函数与区间")

        func_str = st.text_input(
            "输入一元函数 f(x)",
            value="x**2 - 2*x + 2",
            key="1d_func",
            help="支持 Python 数学语法，如 sin(x), exp(x), log(x), sqrt(x) 等",
        )

        c1, c2 = st.columns(2)
        with c1:
            a = st.number_input("积分下界 a", value=0.0, step=0.5, key="1d_a")
        with c2:
            b = st.number_input("积分上界 b", value=3.0, step=0.5, key="1d_b")

        st.markdown(
            '<div class="tip-card">'
            '<strong>提示</strong>：拖拽图表可缩放，双击重置视图。'
            '</div>',
            unsafe_allow_html=True,
        )

    # ---- 主区域 ----
    f, err = make_numpy_function(func_str, ["x"])

    if err:
        st.error(f"❌ {err}")
    elif a >= b:
        st.warning("⚠️ 积分上界 b 必须大于下界 a，请重新设置。")
    else:
        x_vals = np.linspace(a, b, 300)
        y_vals = f(x_vals)
        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals)

        integral_val, _ = integrate.quad(f, a, b)

        # ---- 绘图 ----
        margin = max((b - a) * 0.3, 0.5)
        x_dense = np.linspace(a - margin, b + margin, 600)
        y_dense = f(x_dense)
        if np.isscalar(y_dense):
            y_dense = np.full_like(x_dense, y_dense)

        x_fill = np.linspace(a, b, 400)
        y_fill = f(x_fill)
        if np.isscalar(y_fill):
            y_fill = np.full_like(x_fill, y_fill)

        fig = go.Figure()

        # 积分区域（先画，作为底层）
        fig.add_trace(go.Scatter(
            x=np.concatenate([x_fill, x_fill[::-1]]),
            y=np.concatenate([y_fill, np.zeros_like(y_fill)]),
            fill="toself",
            fillcolor="rgba(102, 126, 234, 0.22)",
            line=dict(width=0),
            name="积分区域",
            hoverinfo="skip",
        ))

        # 函数曲线
        fig.add_trace(go.Scatter(
            x=x_dense, y=y_dense,
            mode="lines",
            name=f"f(x) = {func_str}",
            line=dict(color="#667eea", width=3),
            hovertemplate="x = %{x:.4f}<br>f(x) = %{y:.4f}<extra></extra>",
        ))

        # 积分边界虚线
        for boundary, label, color in [(a, "a", "#e74c3c"), (b, "b", "#2ecc71")]:
            fb = float(f(boundary))
            fig.add_trace(go.Scatter(
                x=[boundary, boundary],
                y=[0, fb],
                mode="lines",
                name=f"x = {label}",
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate=f"x = {boundary:.4f}<extra></extra>",
            ))
            # 边界端点圆点
            fig.add_trace(go.Scatter(
                x=[boundary, boundary],
                y=[0, fb],
                mode="markers",
                marker=dict(size=7, color=color),
                showlegend=False,
                hoverinfo="skip",
            ))

        # 零线
        fig.add_hline(y=0, line=dict(color="#b0b8c8", width=1))

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>∫</b><sub>[{a:.2g}, {b:.2g}]</sub> "
                    f"({func_str}) dx"
                ),
                font=dict(size=18, family="Inter, Microsoft YaHei, sans-serif"),
                x=0.5,
            ),
            xaxis_title="x",
            yaxis_title="f(x)",
            hovermode="x unified",
            template="plotly_white",
            plot_bgcolor="#fafbfc",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, Microsoft YaHei, sans-serif"),
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.12,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#e0e3eb",
                borderwidth=1,
                font=dict(size=12),
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # 结果卡片
        st.markdown(f"""
        <div class="result-card">
            <div class="label">定积分结果</div>
            <div class="value">{integral_val:.8f}</div>
            <div class="formula">
                ∫<sub>{a:.4g}</sub><sup>{b:.4g}</sup> ({func_str}) dx
            </div>
        </div>
        """, unsafe_allow_html=True)

# ====================================================================
# 二元函数积分 (3D)
# ====================================================================
else:

    with st.sidebar:
        st.markdown("#### 📊 函数与积分区域")

        func_str = st.text_input(
            "输入二元函数 f(x, y)",
            value="sin(x) * cos(y) + 2",
            key="2d_func",
            help="支持 Python 数学语法，如 sin(x), cos(y), exp(x+y), sqrt(x**2+y**2) 等",
        )

        st.markdown("##### X 轴范围")
        cx1, cx2 = st.columns(2)
        with cx1:
            x_min = st.number_input("x 下界", value=-np.pi, step=0.5, key="2d_xmin")
        with cx2:
            x_max = st.number_input("x 上界", value=np.pi, step=0.5, key="2d_xmax")

        st.markdown("##### Y 轴范围")
        cy1, cy2 = st.columns(2)
        with cy1:
            y_min = st.number_input("y 下界", value=-np.pi, step=0.5, key="2d_ymin")
        with cy2:
            y_max = st.number_input("y 上界", value=np.pi, step=0.5, key="2d_ymax")

        st.markdown(
            '<div class="tip-card">'
            '<strong>提示</strong>：鼠标拖拽旋转视角，滚轮缩放，右键平移。'
            '</div>',
            unsafe_allow_html=True,
        )

    # ---- 主区域 ----
    f, err = make_numpy_function(func_str, ["x", "y"])

    if err:
        st.error(f"❌ {err}")
    elif x_min >= x_max or y_min >= y_max:
        st.warning("⚠️ 积分上界必须大于下界，请重新设置。")
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
            st.error(f"❌ 函数求值失败: {e}")
            st.stop()

        integral_val, _ = integrate.dblquad(
            lambda y, x: f(x, y),
            y_min, y_max,
            lambda y: x_min, lambda y: x_max,
        )

        # ---- 3D 绘图 ----
        fig = go.Figure()

        # 顶面
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[
                [0.0, "#667eea"],
                [0.3, "#764ba2"],
                [0.6, "#e74c3c"],
                [0.8, "#f39c12"],
                [1.0, "#2ecc71"],
            ],
            name="顶面 f(x, y)",
            showscale=True,
            opacity=0.92,
            colorbar=dict(
                title="f(x, y)",
                thickness=15,
                len=0.5,
                outlinewidth=0,
            ),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white",
                       project=dict(z=True), width=1),
            ),
        ))

        # 底面 z=0
        Z_bottom = np.zeros_like(Z)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_bottom,
            colorscale=[[0, "#bcc3d0"], [1, "#bcc3d0"]],
            name="底面 z = 0",
            showscale=False,
            opacity=0.35,
        ))

        # 四个侧壁
        Y_side, T = np.meshgrid(y_grid, np.linspace(0, 1, 30))

        side_params = [
            ("x", x_min, Y_side, T, "#636efa"),
            ("x", x_max, Y_side, T, "#e74c3c"),
            ("y", y_min, np.meshgrid(x_grid, np.linspace(0, 1, 30))[0], np.meshgrid(x_grid, np.linspace(0, 1, 30))[1], "#2ecc71"),
            ("y", y_max, np.meshgrid(x_grid, np.linspace(0, 1, 30))[0], np.meshgrid(x_grid, np.linspace(0, 1, 30))[1], "#f39c12"),
        ]

        for axis, val, grid_a, grid_t, color in side_params:
            if axis == "x":
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
                name=f"{axis} = {val:.2g}",
                showscale=False,
                opacity=0.45,
            ))

        fig.update_layout(
            title=dict(
                text=(
                    f"<b>∬</b><sub>[{x_min:.2g},{x_max:.2g}]×[{y_min:.2g},{y_max:.2g}]</sub> "
                    f"({func_str}) dx dy"
                ),
                font=dict(size=18, family="Inter, Microsoft YaHei, sans-serif"),
                x=0.5,
            ),
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="f(x, y)",
                camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
                xaxis=dict(
                    gridcolor="#e8ecf2",
                    backgroundcolor="rgba(0,0,0,0)",
                ),
                yaxis=dict(
                    gridcolor="#e8ecf2",
                    backgroundcolor="rgba(0,0,0,0)",
                ),
                zaxis=dict(
                    gridcolor="#e8ecf2",
                    backgroundcolor="rgba(0,0,0,0)",
                ),
            ),
            template="plotly_white",
            margin=dict(l=0, r=0, t=50, b=0),
            font=dict(family="Inter, Microsoft YaHei, sans-serif"),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="#e0e3eb",
                borderwidth=1,
                font=dict(size=11),
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # 结果卡片
        st.markdown(f"""
        <div class="result-card">
            <div class="label">二重积分结果</div>
            <div class="value">{integral_val:.8f}</div>
            <div class="formula">
                ∬<sub>[{x_min:.4g},{x_max:.4g}]×[{y_min:.4g},{y_max:.4g}]</sub>
                ({func_str}) dx dy
            </div>
        </div>
        """, unsafe_allow_html=True)
