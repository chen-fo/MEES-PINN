# MEES-PINN

> From the Big Data National Laboratory, Shenzhen University

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Introduction

**MEES-PINN** is a plug-in platform for testing Physics-Informed Neural Networks (PINNs). It includes **24 PDE benchmark problems** and various baseline methods, covering both Gradient Descent (GD) and Evolutionary Algorithm (EA) approaches.

## ✨ Features

### Gradient Descent Methods
- GD (Gradient Descent)
- SGD (Stochastic Gradient Descent)
- Batch SGD
- Lion (SOTA optimizer)

### Evolutionary Algorithm Methods
- ES (Evolution Strategy)
- GA (Genetic Algorithm)
- PSO (Particle Swarm Optimization)
- CMA-ES
- NSGA-II (Multi-objective optimization)
- And more...

### 🚀 Our Method: AMNES

We propose a novel ES-category method named **AMNES**, which combines:
- **Global search capability** from Genetic Algorithms
- **Local search capability** from Gradient Descent

## 📊 Experiments

We have conducted comprehensive experiments comparing our method against GA and GD baselines across all benchmarks in this platform. The results demonstrate that **AMNES has superior capacity to solve PINN problems**.

## 📝 Citation

If you use this project in your research, please cite our paper:


> **📢 Stay tuned for the official citation!**

## 🙏 Acknowledgements

Special thanks to the support of **Tsinghua University's open source project**.

---

<p align="center">
  Made with ❤️ by Big Data National Laboratory, Shenzhen University
</p>
