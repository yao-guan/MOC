"""
Direct inference script - calls LLM directly without graph structure.
Used as baseline for all datasets: MMLU, AQuA, GSM8K, SVAMP, HumanEval, MMLU-Pro.
"""
import os
import sys
import time
import json
import asyncio
import argparse
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from src.utils.const import MOC_ROOT
from src.utils.globals import Time, Cost, PromptTokens, CompletionTokens
from src.llm.llm_registry import LLMRegistry
from src.prompt.prompt_set_registry import PromptSetRegistry
from datasets.data_process import (
    MMLUDataset, AQuADataset, GSM8KDataset, 
    SVAMPDataset, HumanEvalDataset, MMLUProDataset
)


# ========== Dataset Config Registry ==========
DATASET_CONFIG = {
    'mmlu': {
        'class': MMLUDataset,
        'test_split': 'test',
        'n_size': 5,
        'prompt_domain': 'mmlu',
    },
    'mmlu_pro': {
        'class': MMLUProDataset,
        'test_split': 'test',
        'n_size': 20,
        'prompt_domain': 'mmlu_pro',
    },
    'gsm8k': {
        'class': GSM8KDataset,
        'test_split': 'test',
        'n_size': 300,
        'prompt_domain': 'gsm8k',
    },
    'svamp': {
        'class': SVAMPDataset,
        'test_split': 'test',
        'n_size': 300,
        'prompt_domain': 'gsm8k',
    },
    'aqua': {
        'class': AQuADataset,
        'test_split': 'test',
        'n_size': 254,
        'prompt_domain': 'aqua',
    },
    'humaneval': {
        'class': HumanEvalDataset,
        'test_split': 'test',
        'n_size': 164,
        'prompt_domain': 'humaneval',
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Direct LLM Inference without Graph Structure")
    
    parser.add_argument('--domain', type=str, required=True,
                        choices=['mmlu', 'mmlu_pro', 'aqua', 'gsm8k', 'svamp', 'humaneval'],
                        help="Dataset name")
    parser.add_argument('--n_size', type=int, default=None,
                        help="Dataset size")
    
    parser.add_argument('--llm_name', type=str, default="gemma-chat",
                        help="Model name")
    parser.add_argument('--use_cot', action='store_true',
                        help='Use Chain-of-Thought prompting')
    
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size")
    
    args = parser.parse_args()
    
    if args.n_size is None:
        args.n_size = DATASET_CONFIG[args.domain]['n_size']
    
    return args


async def process_single_question(llm, prompt_set, question_data, use_cot=False):
    """Process a single question."""
    system_prompt = prompt_set.get_direct_constraint(use_cot)
    
    task = question_data.get('task', '')
    user_prompt = f"{task}"
    
    message = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ]
    
    try:
        response = await llm.agen(message)

        extracted_answer = prompt_set.postprocess_answer(response)
        
        ground_truth = question_data.get('answer', '')
        is_correct = (extracted_answer == ground_truth)
        
        return {
            'task': task,
            'ground_truth': ground_truth,
            'llm_response': response,
            'extracted_answer': extracted_answer,
            'is_correct': is_correct
        }
    except Exception as e:
        print(f"Error processing question: {str(e)}")
        return {
            'task': task,
            'ground_truth': question_data.get('answer', ''),
            'llm_response': f"Error: {str(e)}",
            'extracted_answer': '',
            'is_correct': False
        }


async def process_batch(llm, prompt_set, batch_data, use_cot=False):
    """Process a batch of questions."""
    tasks = [process_single_question(llm, prompt_set, q, use_cot) for q in batch_data]
    results = await asyncio.gather(*tasks)
    return results


def save_batch_results(result_file, batch_results):
    """Save batch results to detailed results file."""
    if result_file.exists():
        with open(result_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.extend(batch_results)

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


def save_final_summary(summary_file, config, metrics):
    """Save final summary."""
    summary = {
        'config': config,
        'metrics': metrics,
        'cost': {
            'total_cost': Cost.instance().value,
            'prompt_tokens': PromptTokens.instance().value,
            'completion_tokens': CompletionTokens.instance().value
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\nFinal summary saved to: {summary_file}")


async def main():
    args = parse_args()

    domain_config = DATASET_CONFIG[args.domain]
    dataset_class = domain_config['class']
    prompt_domain = domain_config['prompt_domain']

    print(f"Loading {args.domain} dataset...")
    test_dataset = dataset_class(split=domain_config['test_split'], n_size=args.n_size)
    print(f"Test set size: {len(test_dataset)}")

    print(f"Initializing model: {args.llm_name}")
    llm = LLMRegistry.get(args.llm_name)
    prompt_set = PromptSetRegistry.get(prompt_domain)

    current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    
    result_dir = Path(f"{MOC_ROOT}/result/{args.domain}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    cot_suffix = "_cot" if args.use_cot else ""
    result_file = result_dir / f"{args.llm_name}_direct{cot_suffix}_{current_time}_detail.json"
    summary_file = result_dir / f"{args.llm_name}_direct{cot_suffix}_{current_time}.json"
    
    print("="*80)
    print("Starting direct inference...")
    print(f"Using CoT: {args.use_cot}")
    print("="*80)

    num_batches = (len(test_dataset) + args.batch_size - 1) // args.batch_size
    total_solved = 0
    total_executed = 0
    
    for i_batch in range(num_batches):
        start_idx = i_batch * args.batch_size
        end_idx = min((i_batch + 1) * args.batch_size, len(test_dataset))
        batch_data = [test_dataset[i] for i in range(start_idx, end_idx)]
        
        print(f"\nProcessing batch {i_batch + 1}/{num_batches} (questions {start_idx + 1}-{end_idx})...")

        batch_results = await process_batch(llm, prompt_set, batch_data, args.use_cot)

        batch_solved = sum(1 for r in batch_results if r['is_correct'])
        batch_executed = len(batch_results)
        total_solved += batch_solved
        total_executed += batch_executed

        batch_acc = batch_solved / batch_executed if batch_executed > 0 else 0
        print(f"Batch accuracy: {batch_solved}/{batch_executed} = {batch_acc:.4f}")

        save_batch_results(result_file, batch_results)

        current_acc = total_solved / total_executed if total_executed > 0 else 0
        print(f"Current overall accuracy: {total_solved}/{total_executed} = {current_acc:.4f}")

    final_accuracy = total_solved / total_executed if total_executed > 0 else 0
    
    config = {
        "mode": "direct",
        "domain": args.domain,
        "llm_name": args.llm_name,
        "use_cot": args.use_cot,
        "batch_size": args.batch_size,
        "n_size": args.n_size,
        "timestamp": current_time,
    }
    
    metrics = {
        "accuracy": round(final_accuracy, 4),
        "total_solved": total_solved,
        "total_executed": total_executed,
    }
    
    save_final_summary(summary_file, config, metrics)
    
    print("\n" + "="*80)
    print("Testing complete!")
    print(f"Final accuracy: {final_accuracy:.4f} ({total_solved}/{total_executed})")
    print(f"Detailed results: {result_file}")
    print(f"Summary file: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())