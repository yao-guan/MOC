"""
Common training and evaluation logic.
"""
import time
import copy
import asyncio
import torch
from typing import List, Dict, Any
from pathlib import Path

from src.utils.globals import Cost, PromptTokens, CompletionTokens, ConpressedPromptTokens, ConpressedCompletionTokens


async def train_batch(graph, batch_data, dataset_instance, num_rounds: int, optimizer=None):
    """
    Train on a single batch.

    Args:
        graph: Graph instance.
        batch_data: Batch data (list of DataFrame rows).
        dataset_instance: Dataset instance (provides record_to_input and postprocess_answer).
        num_rounds: Number of inference rounds.
        optimizer: Optimizer (no gradient update if None).

    Returns:
        accuracy: Accuracy for this batch.
        loss: Loss for this batch.
        utilities: List of correctness values per sample.
    """
    tasks = []
    answers = []
    
    for record in batch_data:
        realized_graph = copy.deepcopy(graph)
        realized_graph.gcn = graph.gcn
        realized_graph.mlp = graph.mlp
        
        input_dict = dataset_instance.record_to_input(record)
        answer = dataset_instance.record_to_target_answer(record)
        answers.append(answer)
        
        tasks.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
    
    raw_results = await asyncio.gather(*tasks)
    raw_answers, log_probs = zip(*raw_results)
    
    loss_list: List[torch.Tensor] = []
    utilities: List[float] = []
    correct_count = 0
    
    is_code_task = dataset_instance.get_domain() == 'humaneval'
    
    if is_code_task:
        from src.tools.coding.python_executor import PyExecutor
        executor = PyExecutor()
        
        for pred_answer, log_prob, test_case in zip(raw_answers, log_probs, answers):
            code = dataset_instance.postprocess_answer(pred_answer)
            is_correct, _, _ = executor.execute(code, [test_case], timeout=100)
            correct_count += int(is_correct)
            
            utility = float(is_correct)
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
    else:
        for pred_answer, log_prob, true_answer in zip(raw_answers, log_probs, answers):
            predicted = dataset_instance.postprocess_answer(pred_answer)
            is_correct = (predicted == true_answer)
            correct_count += int(is_correct)
            
            utility = 1.0 if is_correct else 0.0
            utilities.append(utility)
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
    
    total_loss = torch.mean(torch.stack(loss_list))
    
    if optimizer is not None:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    accuracy = correct_count / len(batch_data)
    
    return accuracy, total_loss.item(), utilities


async def evaluate_batch(graph, batch_data, dataset_instance, num_rounds: int):
    """
    Evaluate on a single batch.

    Returns:
        results: List of detailed result dicts per sample.
        accuracy: Accuracy for this batch.
    """
    tasks = []
    answers = []
    questions = []

    for record in batch_data:
        realized_graph = copy.deepcopy(graph)
        realized_graph.gcn = graph.gcn
        realized_graph.mlp = graph.mlp
        
        input_dict = dataset_instance.record_to_input(record)
        answer = dataset_instance.record_to_target_answer(record)
        
        questions.append(input_dict.get('task', str(record.get('question', ''))))
        answers.append(answer)
        
        tasks.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))
    
    raw_results = await asyncio.gather(*tasks)
    raw_answers, log_probs = zip(*raw_results)

    results = []
    correct_count = 0
    
    is_code_task = dataset_instance.get_domain() == 'humaneval'
    
    if is_code_task:
        from src.tools.coding.python_executor import PyExecutor
        executor = PyExecutor()
        
        for question, pred_answer, test_case in zip(questions, raw_answers, answers):
            code = dataset_instance.postprocess_answer(pred_answer)
            is_correct, _, _ = executor.execute(code, [test_case], timeout=100)
            correct_count += int(is_correct)
            
            results.append({
                "Question": question,
                "Generated Code": code,
                "Test Case": test_case,
                "Correct": is_correct
            })
    else:
        for question, pred_answer, true_answer in zip(questions, raw_answers, answers):
            predicted = dataset_instance.postprocess_answer(pred_answer)
            is_correct = (predicted == true_answer)
            correct_count += int(is_correct)
            
            results.append({
                "Question": question,
                "Predicted": predicted,
                "True Answer": true_answer,
                "Correct": is_correct
            })
    
    accuracy = correct_count / len(batch_data)
    
    return results, accuracy


def dataloader(data_list, batch_size: int, batch_idx: int):
    """Simple data loader, returns a list of Series."""
    start = batch_idx * batch_size
    end = min(start + batch_size, len(data_list))
    return [data_list[i] for i in range(start, end)]


def save_batch_results(result_file: Path, results: List[Dict[str, Any]], 
                      total_solved: int, total_executed: int):
    """
    Save batch results to file.
    """
    import json

    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    
    for result in results:
        result["Total solved"] = total_solved
        result["Total executed"] = total_executed
        result["Accuracy"] = total_solved / total_executed
    
    existing_data.extend(results)
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=4, ensure_ascii=False)


def print_batch_summary(batch_idx: int, batch_time: float, accuracy: float, 
                       loss: float, utilities: List[float]):
    """Print batch summary info."""
    print(f"Batch {batch_idx} " + "="*80)
    print(f"  Time: {batch_time:.3f}s")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Utilities: {utilities}")
    print(f"  Cost: ${Cost.instance().value:.4f}")
    print(f"  Prompt Tokens: {int(PromptTokens.instance().value):,}")
    print(f"  Completion Tokens: {int(CompletionTokens.instance().value):,}")


def save_final_summary(summary_file: Path, config: Dict[str, Any], 
                      metrics: Dict[str, Any]):
    """
    Save final experiment summary.

    Args:
        summary_file: Output path.
        config: Experiment configuration.
        metrics: Experiment metrics (accuracy, total_solved, etc.).
    """
    import json
    
    summary_data = {
        **config,
        **metrics,
        "total_cost_usd": round(Cost.instance().value, 4),
        "prompt_tokens": int(PromptTokens.instance().value),
        "completion_tokens": int(CompletionTokens.instance().value),
        "total_tokens": int(PromptTokens.instance().value + CompletionTokens.instance().value),
        "conpressed_prompt_tokens": int(ConpressedPromptTokens.instance().value),
        "conpressed_completion_tokens": int(ConpressedCompletionTokens.instance().value),
        "conpressed_total_tokens": int(ConpressedPromptTokens.instance().value + ConpressedCompletionTokens.instance().value),
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"Final Results:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    if 'total_solved' in metrics:
        print(f"  Correct/Total: {metrics['total_solved']}/{metrics['total_executed']}")
    if 'total_runtime_sec' in metrics:
        print(f"  Total Runtime: {metrics['total_runtime_sec']:.3f}s")
    print(f"  Total Cost: ${Cost.instance().value:.4f}")
    print(f"  Prompt Tokens: {int(PromptTokens.instance().value):,}")
    print(f"  Completion Tokens: {int(CompletionTokens.instance().value):,}")
    print(f"  Total Tokens: {int(PromptTokens.instance().value + CompletionTokens.instance().value):,}")
    print(f"  Compressed Prompt Tokens: {int(ConpressedPromptTokens.instance().value):,}")
    print(f"  Compressed Completion Tokens: {int(ConpressedCompletionTokens.instance().value):,}")
    print(f"  Compressed Total Tokens: {int(ConpressedPromptTokens.instance().value + ConpressedCompletionTokens.instance().value):,}")
    print(f"{'='*80}")
    print(f"Results saved to: {summary_file}")


def save_graph_structure(graph_dir: Path, graph, mode: str):
    """Save graph structure to file."""
    import json
    
    graph_dir.mkdir(parents=True, exist_ok=True)

    nodes_info = []
    for node_id, node in graph.nodes.items():
        nodes_info.append({
            'id': node_id,
            'agent_name': node.__class__.__name__,
            'role': getattr(node, 'role', None),
            'description': getattr(node, 'profile', '')[:100] if hasattr(node, 'profile') else ''
        })

    N = len(nodes_info)
    if hasattr(graph, 'spatial_masks'):
        spatial_masks_tensor = graph.spatial_masks.view(N, N)
        spatial_masks = [[int(spatial_masks_tensor[i][j].item()) for j in range(N)] for i in range(N)]
    else:
        spatial_masks = [[0]*N for _ in range(N)]
        
    def format_adjacency_matrix(matrix, nodes_info):
        if not matrix:
            return "Empty matrix"
        
        N = len(matrix)
        lines = []
        
        role_labels = [nodes_info[i]['role'] if nodes_info[i]['role'] else 'Normal' for i in range(N)]
        max_label_width = max(len(f"Node{i}({role_labels[i]})") for i in range(N))
        max_label_width = max(max_label_width, 15)
        
        header = " " * (max_label_width + 2) + "  ".join(f"{i:2d}" for i in range(N))
        lines.append(header)
        lines.append(" " * (max_label_width + 2) + "  ".join("--" for _ in range(N)))

        for i in range(N):
            role = role_labels[i]
            label = f"Node{i}({role})"
            row_str = f"{label:<{max_label_width}} |" + "  ".join(f" {matrix[i][j]}" for j in range(N))
            lines.append(row_str)
        
        return "\n".join(lines)
    
    formatted_matrix = format_adjacency_matrix(spatial_masks, nodes_info)
    
    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    matrix_file = graph_dir / f'graph_structure_{timestamp}.txt'
    with open(matrix_file, 'w', encoding='utf-8') as f:
        f.write(f"Graph Mode: {mode}\n")
        f.write(f"Number of Nodes: {N}\n")

        f.write("Node Information:\n")
        f.write(f"{'ID':<5} | {'Role':<15} | {'Agent Name':<20} | Description\n")
        f.write("-" * 80 + "\n")
        for node in nodes_info:
            desc = (node['description'][:50] + '...') if len(node['description']) > 50 else node['description']
            role_str = node['role'] if node['role'] else 'Normal'
            f.write(f"{node['id']:<5} | {role_str:<15} | {node['agent_name']:<20} | {desc}\n")

        total_edges = sum(sum(row) for row in spatial_masks)
        total_positions_without_diagonal = N * (N - 1)  
        edge_density = total_edges / total_positions_without_diagonal if total_positions_without_diagonal > 0 else 0
        
        f.write(f"Total Edges: {total_edges}, Density (excluding diagonal): {edge_density:.3f}\n")
        f.write(formatted_matrix)

    
    print(f"\n{'='*80}")
    print(f"Adjacency Matrix Visualization:\n")
    print(formatted_matrix)
    print(f"\nDetails saved to: {matrix_file}")
    print(f"{'='*80}\n")