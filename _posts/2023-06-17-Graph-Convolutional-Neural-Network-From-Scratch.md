---
title:  "Graph Convolutional Neural Network (GCN) from Scratch"
mathjax: true
layout: post
categories: media
---

## Introduction
In this post, I will implement perhaps the most influential Graph Neural Network (GNN) architecture, known as the Graph Convolutional Neural Network (GCN)

## Graph Convolutional Neural Network (GCN)

### Why GCN?
GCN was originally proposed to address the limitation of Manifold Regularization, a Semi-Supervised Learning (SSL) technique based on graphs. The objective of Manifold Regularization is encoded in the following loss function:

$$
\mathcal{L} = |f(X)-Y| + λ\mathcal{L_{reg}}
$$
<br>
where
$$
\mathcal{L_{reg}}=f(X)Lf(X) 
$$
<br>

which is a sum of the supervised loss $$|f(X)-Y|$$ and the semi-supervised loss $$\mathcal{L_{reg}}$$.

The semi-supervised loss $$\mathcal{L_{reg}}$$ enforces the assumption that similar data will belong to similar labels. Therefore, penalizing the model if it predicts different labels between pairs of similar data. 

The similarity of the data is encoded using the Laplacian Matrix $$L=D-A$$ of an undirected graph $$G=(V,E)$$ with $$N$$ vertices $$V$$, edges $$(v_i, v_j) \in E$$, and an adjacency matrix $$A\in \mathbb{R}^{N \times N}$$. $$D$$ is the degree matrix such that $$D_{ii}=∑_{j}A_{ii}$$.

A limitation of the Manifold Regularization approach is that the Laplacian Matrix can only encode the similarity between the node of each graph, when in reality, the relationship between nodes can be more complex. For example, in citation networks, the citation between papers describes interest in a certain topic, which could not be modelled by the similarity between the papers.

