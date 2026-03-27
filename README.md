# MEES-PINN

> From the School of Artifical Intelligence, Shenzhen University

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Introduction

**MEES-PINN** is a plug-in platform for testing Physics-Informed Neural Networks (PINNs). It includes **24 PDE benchmark problems** and various baseline methods, covering both Gradient Descent (GD) and Evolutionary Algorithm (EA) approaches.

## ✨ Features

### Gradient Descent Methods
- SGD 
- Batch SGD
- Lion (SOTA optimizer)

### Evolutionary Algorithm Methods
- GA 
- PSO 
- CMA-ES
- NSGA-II
- And more...

### 🚀 Our Method: AMNES

We propose a novel ES-category method named **AMNES**, which combines:
- **Global search capability** from Genetic Algorithms
- **Local search capability** from Gradient Descent

## 📊 Experiments

We have conducted comprehensive experiments comparing our method against GA and GD baselines across all benchmarks in this platform. The results demonstrate that **AMNES has superior capacity to solve PINN problems**.

## 📝 Citation

If you use this project in your research, please cite our paper:


```
@ARTICLE{chenmees2026,
  author={Chen, Fanke and Zhou, Fengrong and Pan, Yinghui and Lu, Yifan and Luo, Chengwen},
  journal={IEEE Transactions on Evolutionary Computation}, 
  title={A Modular Framework with an Adaptive Momentum-based Evolutionary Strategy for Physics-Informed Neural Networks}, 
  year={2026},
  keywords={Training;Optimization;Benchmark testing;Neural networks;Evolutionary computation;Deep learning;Convergence;Accuracy;Scalability;Partial differential equations;PINN;Evolutionary Strategy;Benchmarks},
  doi={10.1109/TEVC.2026.3676646}}
```

---

<p align="center">
  Made with ❤️ by Shenzhen University
</p>
