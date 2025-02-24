"""Machine learning functions for clustering and calculating Market Opportunity Score (MOS)."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, List


def calculateClusters(features: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Perform K-Means clustering and PCA on quantitative features."""
    scaler = StandardScaler()
    scaledFeatures = scaler.fit_transform(features.drop('cluster', axis=1, errors='ignore'))
    kMeans = KMeans(n_clusters=3, random_state=42)
    clusters = kMeans.fit_predict(scaledFeatures)

    pca = PCA(n_components=2)
    pcaData = pca.fit_transform(scaledFeatures)
    result = features.copy()
    result['cluster'] = clusters
    result['pc1'] = pcaData[:, 0]
    result['pc2'] = pcaData[:, 1]

    return clusters, result


def calculateMarketOpportunityScore(data: pd.DataFrame) -> Tuple[float, np.ndarray]:
    """Calculate the Market Opportunity Score (0-100) based on cluster tightness and feature alignment."""
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(data.drop('cluster', axis=1, errors='ignore'))
    kMeans = KMeans(n_clusters=3, random_state=42)
    clusters = kMeans.fit_predict(scaledData)

    # Cluster tightness (average intra-cluster distance)
    tightness = []
    for cluster in range(3):
        clusterPoints = scaledData[clusters == cluster]
        if len(clusterPoints) > 1:
            centroid = kMeans.cluster_centers_[cluster]
            dist = np.mean(np.linalg.norm(clusterPoints - centroid, axis=1))
            tightness.append(dist)
        else:
            tightness.append(0)
    avgTightness = np.mean(tightness) if tightness else 0

    # Feature alignment (correlation of momentum across clusters)
    momentum = data['momentum'].values
    momCorrs = []
    for cluster in range(3):
        clusterMom = momentum[clusters == cluster]
        if len(clusterMom) > 1:
            momCorrs.append(np.corrcoef(clusterMom, momentum)[0, 1] if len(clusterMom) == len(momentum) else 0)
    momAlignment = np.mean(momCorrs) if momCorrs else 0

    # MOS: 0-100, scaled by tightness (inverse) and alignment
    baseScore = (1 - avgTightness / np.max([1e-10, np.std(scaledData)])) * 50  # Lower tightness = higher score
    alignmentBoost = momAlignment * 50  # Higher correlation = more bullish
    mos = max(0, min(100, baseScore + alignmentBoost))

    return mos, clusters