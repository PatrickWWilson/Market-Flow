"""Main Streamlit application for MarketFlow."""

import streamlit as st
from cryptoData import dataFetcher
from mlAnalysis import calculateClusters, calculateMarketOpportunityScore
from visualizations import createOpportunityGauge, createMosTrendChart, createClusterScatter, createClusterDriftAnimation, \
                          createFeatureHeatmap, displayPortfolioSimulation
import pandas as pd
import numpy as np
from typing import List, Dict


def main():
    """Run the MarketFlow Signals web application."""
    st.title("MarketFlow Signals (MFS)")

    # Top 10 coins (CCXT pairs)
    coins = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BNB/USDT', 'ADA/USDT', 
             'SOL/USDT', 'DOGE/USDT', 'DOT/USDT', 'AVAX/USDT', 'SHIB/USDT']

    # Fetch data and extract features
    data = []
    priceTrends = {}
    for pair in coins:
        try:
            features, trend = dataFetcher.extractFeatures(pair)
            data.append(features)
            priceTrends[pair.split('/')[0]] = trend
        except Exception as e:
            st.write(f"Error fetching {pair}: {e}")
            data.append([np.nan] * 5)

    featureDf = pd.DataFrame(data, 
                             columns=['volatility', 'momentum', 'volumeSkew', 'autocorrelation', 'meanReversionStrength'],
                             index=[p.split('/')[0] for p in coins])
    featureDf.fillna(featureDf.median(), inplace=True)

    # Calculate MOS and clusters
    mos, clusters = calculateMarketOpportunityScore(featureDf)
    clusterDf, _ = calculateClusters(featureDf)
    clusterDf.index = featureDf.index

    # Display Opportunity Gauge
    st.subheader("Crypto Market Opportunity Index (CMOI)")
    st.plotly_chart(createOpportunityGauge(mos), use_container_width=True)
    st.write("Score Explanation: >70 = Buy (bullish structure), 40-70 = Hold (mixed), <40 = Sell (bearish structure).")

    # 7-day MOS trend (simulated, store in CSV for real use)
    st.subheader("7-Day MOS Trend")
    mosHistory = pd.Series([mos] * 7, index=pd.date_range(end=pd.Timestamp.now(), periods=7))  # Placeholder
    createMosTrendChart(mosHistory)

    # Cluster Insights
    st.subheader("Cluster Insights")
    for cluster in clusterDf['cluster'].unique():
        clusterData = clusterDf[clusterDf['cluster'] == cluster]
        st.write(f"Cluster {int(cluster)}: {', '.join(clusterData.index)}")
        st.write(clusterData.mean().drop(['cluster', 'pc1', 'pc2']).to_frame().T)

    # Filters
    filterBy = st.multiselect("Filter Clusters", ["High Volatility", "High Momentum", "Strong Mean Reversion"])
    filteredData = clusterDf
    if "High Volatility" in filterBy:
        filteredData = filteredData[filteredData['volatility'] > filteredData['volatility'].mean()]
    if "High Momentum" in filterBy:
        filteredData = filteredData[filteredData['momentum'] > filteredData['momentum'].mean()]
    if "Strong Mean Reversion" in filterBy:
        filteredData = filteredData[filteredData['meanReversionStrength'].abs() > filteredData['meanReversionStrength'].abs().mean()]

    # Display table with trends
    for coin, row in filteredData.iterrows():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"{coin}: Vol={row['volatility']:.2f}, Mom={row['momentum']:.2f}, Cluster={int(row['cluster'])}")
        with col2:
            st.line_chart(priceTrends[coin], height=100)

    # PCA Visualization
    st.plotly_chart(createClusterScatter(filteredData))

    # Cluster Drift Animation (7-day history, simulated)
    st.subheader("Cluster Drift Over 7 Days")
    # Simulate 7 days of cluster data (in practice, store in CSV or database)
    clusterHistory = []
    for i in range(7):
        dayData = []
        for pair in coins:
            coin = pair.split('/')[0]
            try:
                feats, _ = dataFetcher.extractFeatures(pair)
                dayData.append(feats)
            except Exception as e:
                st.write(f"Error fetching {pair} for day {i}: {e}")
                dayData.append([np.nan] * 5)
        dayDf = pd.DataFrame(dayData, 
                             columns=['volatility', 'momentum', 'volumeSkew', 'autocorrelation', 'meanReversionStrength'],
                             index=coins)
        dayDf.fillna(dayDf.median(), inplace=True)
        _, dayClusters = calculateClusters(dayDf)
        dayClusters.index = coins
        clusterHistory.append(dayClusters.assign(day=i))
    st.plotly_chart(createClusterDriftAnimation(clusterHistory))

    # Feature Heatmap
    st.subheader("Feature Correlations by Cluster")
    st.plotly_chart(createFeatureHeatmap(filteredData))

    # Portfolio Simulation
    st.subheader("Portfolio Simulation (Theoretical)")
    buyThreshold = st.number_input("Buy Threshold (MOS >)", min_value=0, max_value=100, value=75)
    sellThreshold = st.number_input("Sell Threshold (MOS <)", min_value=0, max_value=100, value=45)
    initialValue = st.number_input("Initial Portfolio Value ($)", min_value=1000, value=10000)

    if st.button("Simulate Portfolio"):
        # Fetch 30-day historical data
        histData = {}
        for pair in coins:
            try:
                ohlcv = dataFetcher.fetchOhlcvData(pair, limit=30)
                if ohlcv is not None and not ohlcv.empty:
                    histData[pair.split('/')[0]] = ohlcv['close']
            except Exception as e:
                st.write(f"Error fetching history for {pair}: {e}")

        if histData:
            # Simulate portfolio
            portfolio = initialValue
            weights = np.ones(len(coins)) / len(coins)  # Equal weight initially
            cash = portfolio
            positions = {coin: 0 for coin in coins}
            equityCurve = [portfolio]
            mosHistory = []  # Simulate MOS over 30 days

            for i in range(1, len(histData['BTC'])):
                # Compute MOS for this day (simplified placeholder)
                dayData = []
                for pair in coins:
                    coin = pair.split('/')[0]
                    dayFeats, _ = dataFetcher.extractFeatures(pair)
                    dayData.append(dayFeats)
                dayDf = pd.DataFrame(dayData, 
                                     columns=['volatility', 'momentum', 'volumeSkew', 'autocorrelation', 'meanReversionStrength'],
                                     index=coins)
                dayDf.fillna(dayDf.median(), inplace=True)
                dayMos, _ = calculateMarketOpportunityScore(dayDf)
                mosHistory.append(dayMos)

                # Buy/Sell logic
                if mosHistory[-1] > buyThreshold and cash > 0:
                    invest = cash / len(coins)
                    for coin in coins:
                        positions[coin] += invest / histData[coin].iloc[i]
                    cash = 0
                elif mosHistory[-1] < sellThreshold and any(positions.values()):
                    cash = sum(pos * histData[coin].iloc[i] for coin, pos in positions.items())
                    for coin in positions:
                        positions[coin] = 0

                # Portfolio value
                totalValue = cash + sum(pos * histData[coin].iloc[i] for coin, pos in positions.items())
                equityCurve.append(totalValue)

            # Metrics
            equityDf = pd.Series(equityCurve, index=histData['BTC'].index)
            returns = equityDf.pct_change().dropna()
            sharpe = returns.mean() / returns.std() * np.sqrt(252)  # Annualized
            drawdown = (equityDf / equityDf.cumulmax() - 1).min() * 100  # Max drawdown %

            displayPortfolioSimulation(equityDf, {
                'drawdown': drawdown,
                'sharpe': sharpe
            })


if __name__ == "__main__":
    main()