"""
Unified MOC experiment runner.
Supports all datasets: MMLU, AQuA, GSM8K, SVAMP, HumanEval, MMLU-Pro.
"""
import warnings

# ignore "pkg_resources is deprecated" warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
import math
import asyncio
import argparse
import random
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Union, Literal

from src.graph.graph import Graph
from src.utils.const import MOC_ROOT
from src.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from datasets.data_process import (
    MMLUDataset, AQuADataset, GSM8KDataset, 
    SVAMPDataset, HumanEvalDataset,MMLUProDataset
)
from experiments.common import (
    train_batch, evaluate_batch, dataloader,
    save_batch_results, print_batch_summary, save_final_summary,save_graph_structure
)

# set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ========== Dataset Config Registry ==========
DATASET_CONFIG = {
    'mmlu': {
        'class': MMLUDataset,
        'default_agent': 'AnalyzeAgent',
        'default_decision': 'FinalRefer',
        'test_split': 'test',
        'n_size':5,
        'prompt_domain': 'mmlu',  
    },
    'mmlu_pro': {
        'class': MMLUProDataset,
        'default_agent': 'AnalyzeAgent',
        'default_decision': 'FinalRefer',
        'test_split': 'test',
        'n_size':20,
        'prompt_domain': 'mmlu_pro',  
    },
    'gsm8k': {
        'class': GSM8KDataset,
        'default_agent': 'MathSolver',
        'default_decision': 'FinalRefer',
        'test_split': 'test',
        'n_size':300,
        'prompt_domain': 'gsm8k',  
    },
    'svamp': {
        'class': SVAMPDataset,
        'default_agent': 'MathSolver',
        'default_decision': 'FinalRefer',
        'test_split': 'test',
        'n_size':300,
        'prompt_domain': 'gsm8k',  
    },
    'aqua': {
        'class': AQuADataset,
        'default_agent': 'MathSolver',
        'default_decision': 'FinalRefer',
        'test_split': 'test',
        'n_size':254,
        'prompt_domain': 'aqua',  
    },
    'humaneval': {
        'class': HumanEvalDataset,
        'default_agent': 'CodeWriting',
        'default_decision': 'FinalWriteCode', 
        'test_split': 'test',
        'n_size':164,
        'prompt_domain': 'humaneval',  
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="MOC")
    # Dataset
    parser.add_argument('--domain', type=str, required=True, choices=['mmlu','mmlu_pro','aqua', 'gsm8k', 'svamp', 'humaneval'], help="Dataset name")
    parser.add_argument('--n_size', type=int, default=None, help="Dataset size (for loading preprocessed data)")
    # Model and Agent
    parser.add_argument('--llm_name', type=str, default="gemma2:27b", help="Model name")
    parser.add_argument('--agent_names', nargs='+', type=str, default=None, help='Agent name list (uses dataset default if not specified)')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=7, help='Number of each agent type')
    parser.add_argument('--decision_method', type=str, default=None, help="Final decision method (uses dataset default if not specified)")
    parser.add_argument('--use_cot', action='store_true', help='Use Chain-of-Thought prompting')
    
    # Graph Structure
    parser.add_argument('--mode', type=str, default='Chain', choices=['FullConnected', 'Random', 'Chain'], help="Graph topology mode")
    parser.add_argument('--edge_density', type=float, default=0.5, help='Edge density for random graph (0-1, default 0.5)')
    parser.add_argument('--random_dag_seed', type=int, default=42, help='Random seed for graph structure')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds per query")
    
    # Neighbor Summary
    parser.add_argument('--neighbor_hops', type=int, default=1, help='Number of neighbor hops')
    parser.add_argument('--use_neighbor_summary', action='store_true', default=True, help='Use neighbor summary')
    parser.add_argument('--ism_r', type=float, default=1.0, help='ISM algorithm r parameter')
    parser.add_argument('--ism_epsilon', type=float, default=0.01, help='ISM algorithm epsilon parameter')
    parser.add_argument('--ism_kppa', type=int, default=45, help='ISM algorithm kppa parameter')

    args = parser.parse_args()

    domain_config = DATASET_CONFIG[args.domain]
    if args.agent_names is None:
        default_agent = domain_config['default_agent']
        total_agents = args.agent_nums[0]
        args.agent_names = [default_agent]
        args.agent_nums = [total_agents]

    if args.decision_method is None:
        args.decision_method = domain_config['default_decision']

    if args.n_size is None:
        args.n_size = domain_config['n_size']
    
    if args.agent_nums is not None and len(args.agent_names) != len(args.agent_nums):
        parser.error("agent_names and agent_nums must have the same length")
    
    return args


def get_kwargs(mode: str, N: int, edge_density: float = 0.5, random_dag_seed: int = 42):
    """Generate kwargs for graph structure."""
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None
    local_random = random.Random(random_dag_seed)

    if mode == 'FullConnected':
        order = list(range(N))   
        local_random.shuffle(order)
        pos = {v: k for k, v in enumerate(order)}
        fixed_spatial_masks = [
            [1 if (i != j and pos[i] < pos[j]) else 0 for j in range(N)]
            for i in range(N) ]
        fixed_temporal_masks = [[1 if i==j else 0 for j in range(N)] for i in range(N)]
    elif mode == 'Random':
        order = list(range(N))
        local_random.shuffle(order)
        pos = {v: k for k, v in enumerate(order)}
        dag_candidates = [(i, j) for i in range(N) for j in range(N)
                        if i != j and pos[i] < pos[j]]
        max_dag_edges = len(dag_candidates)  # = N*(N-1)/2
        target_edges = math.ceil(max_dag_edges * edge_density)
        target_edges = max(target_edges, N - 1)
        target_edges = min(target_edges, max_dag_edges)
        spatial_edges = set((order[k], order[k + 1]) for k in range(N - 1))
        remaining = list(set(dag_candidates) - spatial_edges)
        need = target_edges - len(spatial_edges)
        if need > 0:
            spatial_edges.update(local_random.sample(remaining, min(need, len(remaining))))
        fixed_spatial_masks = [[1 if (i, j) in spatial_edges else 0 for j in range(N)]
                            for i in range(N)]
        fixed_temporal_masks = [[1 if i==j else 0 for j in range(N)] for i in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==j else 0 for j in range(N)] for i in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs
    }


async def main():
    args = parse_args()
    run_start_time = time.time()
    domain_config = DATASET_CONFIG[args.domain]
    dataset_class = domain_config['class']
    
    print(f"Loading {args.domain} dataset...")
    test_dataset = dataset_class(split=domain_config['test_split'], n_size=args.n_size)
    
    print(f"Test set size: {len(test_dataset)}")

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names), args.edge_density, args.random_dag_seed)

    print(f"Creating Graph: mode={args.mode}, agents={agent_names}")
    prompt_domain = domain_config.get('prompt_domain', args.domain)
    graph = Graph(
        domain=prompt_domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        use_neighbor_summary=args.use_neighbor_summary,
        neighbor_hops=args.neighbor_hops,
        ism_r=args.ism_r,
        ism_epsilon=args.ism_epsilon,
        ism_kppa=args.ism_kppa,
        use_cot=args.use_cot,
        **kwargs
    )

    graph_dir = Path(f"{MOC_ROOT}/graph_structures/{args.domain}/{args.mode}")
    save_graph_structure(graph_dir, graph, args.mode)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{MOC_ROOT}/result/{args.domain}")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = result_dir / f"{args.llm_name}_{current_time}_detail.json"
    summary_file = result_dir / f"{args.llm_name}_{args.mode}_{current_time}.json"

    print("="*80)
    print("Starting evaluation...")
    print("="*80)

    num_test_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size
    total_solved = 0
    total_executed = 0
    
    for i_batch in range(num_test_batches):
        batch_data = dataloader(test_dataset, args.batch_size, i_batch)
        if len(batch_data) == 0:
            break
        
        start_time = time.time()
        results, accuracy = await evaluate_batch(
            graph, batch_data, test_dataset, args.num_rounds
        )
        batch_time = time.time() - start_time

        batch_correct = sum(1 for r in results if r['Correct'])
        total_solved += batch_correct
        total_executed += len(results)

        save_batch_results(result_file, results, total_solved, total_executed)

        print(f"\nBatch {i_batch} " + "="*60)
        print(f"  Time: {batch_time:.3f}s")
        print(f"  Batch Accuracy: {accuracy:.4f}")
        print(f"  Cumulative Accuracy: {total_solved}/{total_executed} = {total_solved/total_executed:.4f}")

    final_accuracy = total_solved / total_executed if total_executed > 0 else 0
    
    config = {
        "domain": args.domain,
        "mode": args.mode,
        "use_cot": args.use_cot,  
        "edge_density": args.edge_density,
        "llm_name": args.llm_name,
        "agent_nums": args.agent_nums,
        "agent_names": agent_names,
        "decision_method": args.decision_method,
        "num_rounds": args.num_rounds,
        "batch_size": args.batch_size,
        "use_neighbor_summary": args.use_neighbor_summary,
        "neighbor_hops": args.neighbor_hops,
        "ism_r": args.ism_r,
        "ism_epsilon": args.ism_epsilon,
        "ism_kppa": args.ism_kppa,
        "random_dag_seed": args.random_dag_seed,
        "timestamp": current_time,
    }
    
    metrics = {
        "accuracy": round(final_accuracy, 4),
        "total_solved": total_solved,
        "total_executed": total_executed,
        "total_runtime_sec": round(time.time() - run_start_time, 3),
    }
    
    save_final_summary(summary_file, config, metrics)


if __name__ == "__main__":
    asyncio.run(main())