# Scaling Graph Neural Networks (GNNs)

This repository explores techniques for scaling Graph Neural Networks to handle large, real-world graphs efficiently. GNNs are powerful tools for tasks such as node classification, link prediction, and graph classification, but they face challenges when scaling to large graphs due to high memory consumption and computational complexity. In this project, we implement and benchmark various GNN scaling techniques.

## Table of Contents

- [Introduction](#introduction)
- [Techniques for Scaling GNNs](#techniques-for-scaling-gnns)
- [Implementation](#implementation)
- [Dataset](#dataset)
- [Benchmark Results](#benchmark-results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Graph Neural Networks (GNNs) have become popular for processing graph-structured data, but they are often difficult to scale to large datasets. This project addresses the challenge of scaling GNNs by exploring a variety of techniques such as sampling, sparse tensor operations, and distributed training.

The main objectives of this project are:
1. To implement scalable versions of popular GNN architectures.
2. To evaluate performance on large datasets.
3. To compare the effectiveness of various scaling techniques.

## Techniques for Scaling GNNs

1. **Graph Sampling Techniques**
   - **Neighbor Sampling**: Sample a fixed number of neighbors for each node during training.
   - **GraphSAGE**: A GNN variant that uses sampling to aggregate information from a fixed-size neighborhood.
   - **Subgraph Sampling**: Train on smaller subgraphs sampled from the full graph (e.g., Cluster-GCN).
   
2. **Sparse Tensor Operations**
   - Use sparse matrix representations to reduce memory consumption and improve computational efficiency.

3. **Graph Partitioning**
   - Partition the large graph into smaller subgraphs that can be processed independently.
   - Techniques like **METIS** and **LEMON** can be used to partition large graphs.

4. **Distributed Training**
   - Train GNNs across multiple machines using frameworks like **DGL** and **PyTorch Geometric**.
   - **Data Parallelism** and **Model Parallelism** strategies are implemented to handle large-scale training.

5. **Memory Optimization**
   - Use mixed precision training and gradient checkpointing to reduce memory overhead.

## Implementation

The project contains the following key implementations:

- **GNN Architectures**: GCN, GAT, GraphSAGE, and Cluster-GCN.
- **Scaling Techniques**: We have implemented neighbor sampling, subgraph sampling, and partitioning.
- **Training Scripts**: Scripts for single-machine and distributed training.

### Key Libraries
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Deep Graph Library (DGL)](https://www.dgl.ai/)
- [METIS Graph Partitioning](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)

## Dataset

We use the following datasets to test the scalability of the GNNs:

1. **Open Graph Benchmark (OGB)**: Datasets such as OGBN-Products, OGBN-Arxiv, and OGBN-Papers100M.
2. **Reddit**: A large-scale social network dataset.
3. **Amazon2M**: A large graph from the Amazon product co-purchasing network.

Download datasets via the provided scripts in the `/data` folder.

### Distributed Training Example

For distributed training using PyTorch Distributed Data Parallel (DDP):

## Benchmark Results

| Model    | Dataset        | Sampling    | GPU(s) | Accuracy | Training Time |
|----------|----------------|-------------|--------|----------|---------------|
| GCN      | OGBN-Arxiv      | Full Batch  | 1      | 71.2%    | 2h 34m        |
| GraphSAGE| OGBN-Products   | Neighbor    | 4      | 78.6%    | 3h 12m        |
| GAT      | Reddit          | Subgraph    | 1      | 89.4%    | 1h 46m        |
| Cluster-GCN | Amazon2M     | Partition   | 8      | 75.3%    | 5h 10m        |

## Contributing

We welcome contributions! Please feel free to submit pull requests or open issues to report bugs or request features. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Added feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
