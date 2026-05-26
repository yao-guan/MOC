# MOC: Multi-Order Communication in LLM-based Multi-Agent Systems

## Overview

**MOC** is a multi-order communication mechanism for LLM-based multi-agent systems that works with various agent topologies, such as random graphs, predefined graphs, and task-adaptive graphs, to enable multi-hop information exchange and reduce redundant message passing.

## Quick Start

### Install packages

```bash
conda create -n moc python=3.10
conda activate moc
pip install -r requirements.txt
```

### LLM Setup

MOC supports two LLM backends: API-based models and local Ollama models.

#### API Mode 

Add API keys in `template.env` and rename it to `.env`:

```python
BASE_URL = ""
API_KEY = "" 
```

#### Local Mode (Ollama)

Install [Ollama](https://ollama.com/) and pull the models:

```bash
ollama pull gemma2:27b  
ollama pull gemma2:9b    # distillation model
```

### Run MOC on MMLU with Random Graph

```bash
python experiments/run_experiment.py --domain mmlu --mode Random --edge_density 0.7 --agent_nums 7 --random_dag_seed 42 --neighbor_hops 2
```

<!-- ## Citation

Your support would be greatly appreciated if you find MOC interesting or useful. Please acknowledge our work by citing the paper and giving this repository a star. Feel free to open an issue if you have any questions. -->

<!-- ```bibtex
@inproceedings{
anonymous2026moc,
title={{MOC}: Multi-Order Communication in {LLM}-based Multi-Agent Systems},
author={Anonymous},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
}
``` -->

## Acknowledgement

This code refers to [GDesigner](https://github.com/yanweiyue/GDesigner).
