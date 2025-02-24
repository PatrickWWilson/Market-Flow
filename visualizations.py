"""Visualization functions for MarketFlow Signals using Plotly and Streamlit."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from typing import Dict


def createOpportunityGauge(mos: float, title: str = "Crypto Market Opportunity Index (CMOI)") -> go.Figure:
    """Create a circular gauge for the Market Opportunity Score."""
    signal = "Buy" if mos > 70 else "Hold" if 40 <= mos <= 70 else "Sell"
    fig = px.pie(values=[mos, 100 - mos], 
                 names=['Opportunity', 'Caution'], 
                 hole=0.4, color_discrete_sequence=['green', 'red'],
                 title=f"{title}: {int(mos)} ({signal})")
    fig.update_layout(showlegend=False, 
                      polar=dict(radialaxis=dict(visible=False, range=[0, 100])),
                      annotations=[dict(text=str(int(mos)), x=0.5, y=0.5, font_size=20, showarrow=False)])
    return fig


def createMosTrendChart(mosHistory: pd.Series) -> None:
    """Display a 7-day trend line for the Market Opportunity Score."""
    st.line_chart(mosHistory, title="7-Day MOS Trend")


def createClusterScatter(clusters: pd.DataFrame, title: str = "Clusters in PCA Space") -> go.Figure:
    """Create a PCA scatter plot of clusters."""
    fig = px.scatter(clusters, x='pc1', y='pc2', color='cluster', hover_data=clusters.columns[:-2],
                     title=title, labels={'pc1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%})',
                                         'pc2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'})
    return fig


def createClusterDriftAnimation(clustersHistory: List[pd.DataFrame]) -> go.Figure:
    """Create an animated scatter plot of cluster drift over 7 days."""
    driftData = pd.concat([df.assign(day=i) for i, df in enumerate(clustersHistory)], ignore_index=True)
    fig = px.scatter(driftData, x='pc1', y='pc2', color='cluster', animation_frame='day', 
                     hover_data=['coin'], title="Cluster Drift Animation (7 Days)")
    return fig


def createFeatureHeatmap(clusters: pd.DataFrame, title: str = "Feature Correlations by Cluster") -> go.Figure:
    """Create a heatmap of feature correlations within clusters."""
    correlations = clusters.groupby('cluster').corr()
    fig = px.imshow(correlations, text_auto=True, title=title)
    return fig


def displayPortfolioSimulation(equityCurve: pd.Series, metrics: Dict[str, float]) -> None:
    """Display theoretical portfolio simulation results."""
    st.line_chart(equityCurve, title="Theoretical Portfolio Equity Curve")
    st.write(f"Final Value: ${equityCurve.iloc[-1]:,.2f}")
    st.write(f"Max Drawdown: {metrics['drawdown']:.2f}%")
    st.write(f"Sharpe Ratio: {metrics['sharpe']:.2f}")
    st.write("Note: This is a theoretical simulation for analysis only, not a trading recommendation.")