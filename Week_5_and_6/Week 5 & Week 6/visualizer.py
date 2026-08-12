import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from dataset import GENRES


def plot_pca_2d(titles, Z, exp_var, highlighted_title=None):
    """
    Creates an interactive Plotly 2D Scatter Plot of movies projected onto Principal Components 1 & 2.
    """
    df_pca = pd.DataFrame({
        "Title": titles,
        "PC1": Z[:, 0],
        "PC2": Z[:, 1]
    })

    pc1_label = f"PC 1 ({exp_var[0]*100:.1f}% Variance)"
    pc2_label = f"PC 2 ({exp_var[1]*100:.1f}% Variance)"

    df_pca["Is_Selected"] = df_pca["Title"].apply(
        lambda t: "Selected Movie" if (highlighted_title and t.lower() == highlighted_title.lower()) else "Movie"
    )

    fig = px.scatter(
        df_pca,
        x="PC1",
        y="PC2",
        text="Title",
        color="Is_Selected",
        color_discrete_map={"Selected Movie": "#FF4B4B", "Movie": "#00C9A7"},
        hover_name="Title",
        labels={"PC1": pc1_label, "PC2": pc2_label},
        title="🎬 2D PCA Movie Projection (Eigenvalue / Eigenvector Space)"
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(size=12, opacity=0.85, line=dict(width=1.5, color="white"))
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, sans-serif", size=13),
        title=dict(font=dict(size=18, color="#FFFFFF")),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        height=550
    )

    return fig


def plot_covariance_heatmap(C, genres=GENRES):
    """
    Plots heatmap of the 8x8 Genre Covariance Matrix C.
    """
    fig = px.imshow(
        C,
        x=genres,
        y=genres,
        color_continuous_scale="Viridis",
        title="📊 Covariance Matrix (Genre Dependencies)",
        labels=dict(x="Genre Feature", y="Genre Feature", color="Covariance"),
        text_auto=".2f"
    )

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, sans-serif"),
        height=450
    )
    return fig


def plot_loss_curve(history):
    """
    Plots Gradient Descent Loss vs Epochs curve.
    """
    epochs = [step["epoch"] for step in history]
    losses = [step["loss"] for step in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs,
        y=losses,
        mode="lines+markers",
        name="MSE Loss",
        line=dict(color="#FF4B4B", width=3),
        marker=dict(size=6, color="#FFA0A0")
    ))

    fig.update_layout(
        title="📉 Gradient Descent Loss Convergence",
        xaxis_title="Epoch Number",
        yaxis_title="Mean Squared Error Loss (L)",
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, sans-serif"),
        height=400
    )
    return fig


def plot_user_weights(weights, genres=GENRES):
    """
    Bar chart showing learned user preference weight per genre feature.
    """
    colors = ["#00C9A7" if w >= 0 else "#FF4B4B" for w in weights]

    fig = go.Figure(go.Bar(
        x=genres,
        y=weights,
        marker_color=colors,
        text=[f"{w:+.2f}" for w in weights],
        textposition="auto"
    ))

    fig.update_layout(
        title="🎯 Learned User Preference Vector (w)",
        xaxis_title="Genre Feature",
        yaxis_title="Learned Weight (w_j)",
        template="plotly_dark",
        plot_bgcolor="rgba(15, 23, 42, 0.6)",
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        font=dict(family="Inter, sans-serif"),
        height=400
    )
    return fig
