import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from scipy import integrate
import re

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_numpy_function(expr_str, variables):
    """将用户输入的表达式字符串编译为可调用的 Python 函数。

    Args:
        expr_str:  数学表达式字符串，如 "x**2 - 2*x + 2"
        variables: 符号变量列表，如 ['x'] 或 ['x', 'y']

    Returns:
        (func, error_msg)
        func:     可调用函数 f(x) 或 f(x, y)，失败时为 None
        error_msg: 错误信息字符串，成功时为 None
    """
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
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(page_title="微积分多维可视化", layout="wide")
st.title("微积分多维可视化")

# ---- 侧边栏：选择维度 ------------------------------------------------
st.sidebar.header("参数配置")
dimension = st.sidebar.radio(
    "选择积分维度",
    options=["一元函数积分 (2D 面积)", "二元函数积分 (3D 体积)"],
)

# ====================================================================
# 一元函数积分 (2D)
# ====================================================================
if dimension == "一元函数积分 (2D 面积)":

    st.sidebar.subheader("函数与积分区间")

    func_str = st.sidebar.text_input(
        "输入一元函数 f(x)",
        value="x**2 - 2*x + 2",
        key="1d_func",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        a = st.number_input("下界 a", value=0.0, step=0.5, key="1d_a")
    with col2:
        b = st.number_input("上界 b", value=3.0, step=0.5, key="1d_b")

    # ---- 计算 & 绘图 --------------------------------------------------
    f, err = make_numpy_function(func_str, ["x"])

    if err:
        st.error(err)
    elif a >= b:
        st.warning("上界 b 必须大于下界 a。")
    else:
        # 采样点
        x_vals = np.linspace(a, b, 300)
        y_vals = f(x_vals)

        if np.isscalar(y_vals):
            y_vals = np.full_like(x_vals, y_vals)

        # 用 scipy 计算定积分
        integral_val, _ = integrate.quad(f, a, b)

        # ------------- 绘图 -------------
        x_dense = np.linspace(
            min(a - 1, -1), max(b + 1, 1), 600
        )
        y_dense = f(x_dense)
        if np.isscalar(y_dense):
            y_dense = np.full_like(x_dense, y_dense)

        x_fill = np.linspace(a, b, 400)
        y_fill = f(x_fill)
        if np.isscalar(y_fill):
            y_fill = np.full_like(x_fill, y_fill)

        fig = go.Figure()

        # 函数曲线
        fig.add_trace(go.Scatter(
            x=x_dense, y=y_dense,
            mode="lines",
            name=f"f(x) = {func_str}",
            line=dict(color="#636EFA", width=2.5),
        ))

        # 积分区域 (fill to zero)
        fig.add_trace(go.Scatter(
            x=x_fill, y=y_fill,
            mode="lines",
            name="积分区域",
            fill="tozeroy",
            fillcolor="rgba(99, 110, 250, 0.25)",
            line=dict(color="rgba(99, 110, 250, 0.6)", width=1),
        ))

        # 竖直线标记积分边界
        for boundary, boundary_label in [(a, "a"), (b, "b")]:
            fig.add_trace(go.Scatter(
                x=[boundary, boundary],
                y=[0, f(boundary)],
                mode="lines",
                name=f"x = {boundary_label}",
                line=dict(color="#EF553B", width=1.5, dash="dot"),
            ))

        fig.add_hline(y=0, line=dict(color="gray", width=1))

        fig.update_layout(
            title=(
                f"∫[{a:.2g}, {b:.2g}] ({func_str}) dx "
                f"= <b>{integral_val:.6f}</b>"
            ),
            xaxis_title="x",
            yaxis_title="f(x)",
            hovermode="x unified",
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        # 数值结果卡片
        st.metric(
            label="定积分结果",
            value=f"{integral_val:.8f}",
            delta=f"∫[{a:.2g}, {b:.2g}] ({func_str}) dx",
        )

# ====================================================================
# 二元函数积分 (3D)
# ====================================================================
else:

    st.sidebar.subheader("函数与积分区域")

    func_str = st.sidebar.text_input(
        "输入二元函数 f(x, y)",
        value="sin(x) * cos(y) + 2",
        key="2d_func",
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        x_min = st.number_input("x 下界", value=-np.pi, step=0.5, key="2d_xmin")
        x_max = st.number_input("x 上界", value=np.pi, step=0.5, key="2d_xmax")
    with col2:
        y_min = st.number_input("y 下界", value=-np.pi, step=0.5, key="2d_ymin")
        y_max = st.number_input("y 上界", value=np.pi, step=0.5, key="2d_ymax")

    # ---- 计算 & 绘图 --------------------------------------------------
    f, err = make_numpy_function(func_str, ["x", "y"])

    if err:
        st.error(err)
    elif x_min >= x_max or y_min >= y_max:
        st.warning("上界必须大于下界。")
    else:
        # 网格分辨率：自适应调整避免卡顿
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

        # scipy 二重积分
        integral_val, _ = integrate.dblquad(
            lambda y, x: f(x, y),
            y_min, y_max,
            lambda y: x_min, lambda y: x_max,
        )

        # ------------- 3D 绘图 -------------
        fig = go.Figure()

        # 1) 函数顶面
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale="Viridis",
            name="顶面 f(x, y)",
            showscale=True,
            opacity=0.9,
        ))

        # 2) z=0 底面
        Z_bottom = np.zeros_like(Z)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_bottom,
            colorscale=[[0, "lightgray"], [1, "lightgray"]],
            name="底面 z=0",
            showscale=False,
            opacity=0.4,
        ))

        # 3) 四个垂直侧壁
        # x = x_min 侧壁
        Y_side, Z_side = np.meshgrid(y_grid, np.linspace(0, 1, 30))
        X_side = np.full_like(Y_side, x_min)
        f_y = f(x_min, Y_side)
        if np.isscalar(f_y):
            f_y = np.full_like(Y_side, f_y)
        Z_side_scaled = f_y * Z_side
        fig.add_trace(go.Surface(
            x=X_side, y=Y_side, z=Z_side_scaled,
            colorscale=[[0, "#636EFA"], [1, "#636EFA"]],
            name=f"x = {x_min:.2g}",
            showscale=False,
            opacity=0.5,
        ))

        # x = x_max 侧壁
        X_side = np.full_like(Y_side, x_max)
        f_y = f(x_max, Y_side)
        if np.isscalar(f_y):
            f_y = np.full_like(Y_side, f_y)
        Z_side_scaled = f_y * Z_side
        fig.add_trace(go.Surface(
            x=X_side, y=Y_side, z=Z_side_scaled,
            colorscale=[[0, "#EF553B"], [1, "#EF553B"]],
            name=f"x = {x_max:.2g}",
            showscale=False,
            opacity=0.5,
        ))

        # y = y_min 侧壁
        X_side2, Z_side2 = np.meshgrid(x_grid, np.linspace(0, 1, 30))
        Y_side2 = np.full_like(X_side2, y_min)
        f_x = f(X_side2, y_min)
        if np.isscalar(f_x):
            f_x = np.full_like(X_side2, f_x)
        Z_side_scaled2 = f_x * Z_side2
        fig.add_trace(go.Surface(
            x=X_side2, y=Y_side2, z=Z_side_scaled2,
            colorscale=[[0, "#00CC96"], [1, "#00CC96"]],
            name=f"y = {y_min:.2g}",
            showscale=False,
            opacity=0.5,
        ))

        # y = y_max 侧壁
        Y_side2 = np.full_like(X_side2, y_max)
        f_x = f(X_side2, y_max)
        if np.isscalar(f_x):
            f_x = np.full_like(X_side2, f_x)
        Z_side_scaled2 = f_x * Z_side2
        fig.add_trace(go.Surface(
            x=X_side2, y=Y_side2, z=Z_side_scaled2,
            colorscale=[[0, "#AB63FA"], [1, "#AB63FA"]],
            name=f"y = {y_max:.2g}",
            showscale=False,
            opacity=0.5,
        ))

        fig.update_layout(
            title=(
                f"∫∫[{x_min:.2g},{x_max:.2g}]×[{y_min:.2g},{y_max:.2g}] "
                f"({func_str}) dx dy = <b>{integral_val:.6f}</b>"
            ),
            scene=dict(
                xaxis_title="x",
                yaxis_title="y",
                zaxis_title="f(x, y)",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            ),
            template="plotly_white",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.metric(
            label="二重积分结果",
            value=f"{integral_val:.8f}",
            delta=(f"∫∫[{x_min:.2g},{x_max:.2g}]"
                   f"×[{y_min:.2g},{y_max:.2g}] ({func_str}) dx dy"),
        )
