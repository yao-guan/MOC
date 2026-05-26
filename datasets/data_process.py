import json
import os
import ast
import re
import random
from abc import ABC, abstractmethod
from typing import Union, List, Any, Dict

import pandas as pd


"""
Format Alignment: Regardless of whether the raw data is CSV (MMLU), JSONL (AQuA), or JSON (SVAMP),
different _load_data implementations ultimately unify everything into a familiar pd.DataFrame structure.

Prompt Standardization: The record_to_input function converts structured data into LLM-understandable
text, dynamically generating wrong_answer (negative reference) for comparative experiments.

Robust Post-processing: Answer extraction logic differs completely between math problems (GSM8K) and
multiple-choice (MMLU). postprocess_answer implements priority-based intelligent extraction
(keyword -> Boxed format -> regex matching), greatly improving accuracy calculation reliability.
"""

"""
===========================================================================================
Dataset Distribution & Sampling Strategy
===========================================================================================

1. MMLU (Massive Multitask Language Understanding):
   - Structure: 57 subject categories.
   - Strategy: Stratified Sampling, n samples per category.
   - Scale: Total = 57 * n.

2. MMLU-Pro (Enhanced MMLU):
   - Structure: 14 more challenging reasoning categories.
   - Strategy: Stratified Sampling, n samples per category.
   - Scale: Total = 14 * n.

3. SVAMP (Synthesized Variations of Arithmetic Math Problems):
   - Structure: Semantic variants of arithmetic problems.
   - Strategy: Full test set, 300 problems total.
   - Note: Small set but many logic traps; good for testing sensitivity to wording changes.

4. AQuA-RAT (Algebra Question Answering with Rationales):
   - Structure: Algebraic word problems.
   - Strategy: Full test set, 254 problems total.
   - Note: A-E multiple-choice format with complex reasoning chains.

5. GSM8K (Grade School Math 8K):
   - Structure: Grade-school math word problems.
   - Strategy: Randomly sample n from 1319 original test items.
   - Note: Large train/test pool; fixed random seed ensures sampling consistency.

===========================================================================================
All random sampling uses a fixed random seed (random_state=888) to ensure reproducibility.
===========================================================================================
"""

# ==========================================
# 1. Abstract Base Class: Common logic for all datasets
# ==========================================
class BaseDataset(ABC):
    def __init__(self, domain: str, split: str, n_size: int, data_ext: str = "csv"):
        self._domain = domain
        self._split = split
        self._data_path = f"datasets/test/{domain}_{split}_n{n_size}.{data_ext}"
        self._total_df: pd.DataFrame = self._load_data(self._data_path)

    @abstractmethod
    def _load_data(self, data_path: str) -> pd.DataFrame:
        pass

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.Series:
        return self._total_df.iloc[index]

    @staticmethod
    @abstractmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        pass

    @abstractmethod
    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        pass

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        pass
    
    @staticmethod
    def get_domain() -> str:
        pass


# ==========================================
# 2. MMLU Dataset Classes
# ==========================================
class MMLUDataset(BaseDataset):
    def __init__(self, split: str = 'test', n_size: int = 5):
        super().__init__('mmlu', split, n_size)

    def _load_data(self, data_path: str) -> pd.DataFrame:
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Missing: {data_path}")
        df = pd.read_csv(data_path)
        print(f"data_path: {data_path}, Loaded dataset: {len(df)} questions")
        return df

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        demo_question = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
        )
        all_options = ['A', 'B', 'C', 'D']
        correct_answer = str(record['correct_answer']).strip().upper()
        wrong_options = [opt for opt in all_options if opt != correct_answer]
        wrong_answer = random.choice(wrong_options) if wrong_options else "N/A"
        
        return {"task": demo_question, "wrong_answer": wrong_answer}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer_upper = answer.strip().upper()
        answer_upper = answer_upper.replace("OPTION", "").replace(":", "").strip()
        
        match = re.search(r'[A-D]', answer_upper)
        return match.group(0) if match else ""

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        return str(record['correct_answer']).strip().upper()

    @staticmethod
    def get_domain() -> str:
        return 'mmlu'


# ==========================================
# 3. MMLU-Pro Dataset Class
# ==========================================
class MMLUProDataset(BaseDataset):
    def __init__(self, split: str = 'test', n_size: int = 10):
        super().__init__('mmlu_pro', split, n_size)

    def _load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        if 'options' in df.columns and isinstance(df['options'].iloc[0], str):
            df['options'] = df['options'].apply(ast.literal_eval)
        print(f"data_path: {data_path}, Loaded MMLU-Pro: {len(df)} questions")
        return df

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        task_description = (
            f"{record['question']}\n"
            f"Option A: {record['A']}\n"
            f"Option B: {record['B']}\n"
            f"Option C: {record['C']}\n"
            f"Option D: {record['D']}\n"
            f"Option E: {record['E']}\n"
            f"Option F: {record['F']}\n"
            f"Option G: {record['G']}\n"
            f"Option H: {record['H']}\n"
            f"Option I: {record['I']}\n"
            f"Option J: {record['J']}\n"
        )
            
        all_options = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        correct_answer = str(record['answer']).strip().upper()
        wrong_options = [opt for opt in all_options if opt != correct_answer]
        wrong_answer = random.choice(wrong_options) if wrong_options else "N/A"
        
        return {"task": task_description, "wrong_answer": wrong_answer}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        answer_upper = answer.strip().upper()
        answer_upper = answer_upper.replace("OPTION", "").replace(":", "").strip()

        match = re.search(r'[A-J]', answer_upper)
        return match.group(0) if match else ""

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        return str(record['answer']).strip().upper()

    @staticmethod
    def get_domain() -> str:
        return 'mmlu_pro'


# ==========================================
# 4. AQuA Dataset Class (JSONL)
# ==========================================
class AQuADataset(BaseDataset):
    def __init__(self, split: str = 'test', n_size: int = 200):
        super().__init__('aqua', split, n_size, data_ext="jsonl")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_json(data_path, lines=True)
        print(f"data_path: {data_path}, Loaded AQuA: {len(df)} questions")
        return df

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        options = record['options']
        task_description = f"Question: {record['question']}\n"
        for opt in options:
            task_description += f"{opt}\n"
            
        correct_answer = str(record['correct']).strip().upper()
        all_labels = ['A', 'B', 'C', 'D', 'E']
        wrong_options = [label for label in all_labels if label != correct_answer]
        wrong_answer = random.choice(wrong_options)
        
        return {"task": task_description, "wrong_answer": wrong_answer}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        
        match = re.search(r'The answer is ([A-E])', answer, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        answer_upper = answer.strip().upper()
        answer_upper = answer_upper.replace("OPTION", "").replace(":", "").strip()
        matches = re.findall(r'\b([A-E])\b', answer_upper)
        return matches[-1] if matches else ""

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        return str(record['correct']).strip().upper()

    @staticmethod
    def get_domain() -> str:
        return 'aqua'


# ==========================================
# 5. Math Datasets (GSM8K & SVAMP)
# ==========================================
class MathDataset(BaseDataset):
    """Common numeric result cleaning utility for math datasets."""
    @staticmethod
    def _clean_numeric_result(text: str) -> str:
        text = text.replace(",", "")
        numbers = re.findall(r'-?\d*\.?\d+', text)
        if not numbers:
            return ""
        
        pred = numbers[-1].rstrip('.')
        try:
            val = float(pred)
            if val == int(val):
                return str(int(val))
            return str(val)
        except (ValueError, TypeError):
            return pred

class GSM8KDataset(MathDataset):
    def __init__(self, split: str = 'test', n_size: int = 200):
        super().__init__('gsm8k', split, n_size)

    def _load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_csv(data_path)
        print(f"data_path: {data_path}, Loaded GSM8K: {len(df)} questions")
        return df

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        try:
            correct_val = float(str(record['answer']).split('####')[-1].strip().replace(',', ''))
            wrong_answer = str(correct_val + random.choice([-1, 1, 2]))
        except (ValueError, IndexError):
            wrong_answer = "N/A"
        return {"task": f"Question: {record['question']}\n", "wrong_answer": wrong_answer}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return self._clean_numeric_result(answer)

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        raw_answer = str(record['answer']).split('####')[-1].replace(',', '').strip()
        return MathDataset._clean_numeric_result(raw_answer)

    @staticmethod
    def get_domain() -> str:
        return 'gsm8k'


class SVAMPDataset(MathDataset):
    def __init__(self, split: str = 'test', n_size: int = 300):
        super().__init__('svamp', split, n_size, data_ext="json")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        with open(data_path, 'r', encoding='utf-8') as f:
            df = pd.DataFrame(json.load(f))
        print(f"data_path: {data_path}, Loaded SVAMP: {len(df)} questions")
        return df

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        context = f"Context: {record['Body']}\nQuestion: {record['Question']}\nAnswer: "
        return {"task": context, "wrong_answer": "N/A"}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return self._clean_numeric_result(answer)

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        return MathDataset._clean_numeric_result(str(record['Answer']))

    @staticmethod
    def get_domain() -> str:
        return 'svamp'


# ==========================================
# 6. Code Generation Dataset (HumanEval)
# ==========================================
class HumanEvalDataset(BaseDataset):
    """HumanEval code generation dataset."""
    def __init__(self, split: str = 'test', n_size: int = 164):
        super().__init__('humaneval', split, n_size, data_ext="jsonl")

    def _load_data(self, data_path: str) -> pd.DataFrame:
        df = pd.read_json(data_path, lines=True)
        print(f"data_path: {data_path}, Loaded HumanEval: {len(df)} problems")
        return df

    @staticmethod
    def record_to_input(record: pd.Series) -> Dict[str, Any]:
        """Return the prompt directly as the task."""
        return {"task": record['prompt']}

    def postprocess_answer(self, answer: Union[str, List[str]]) -> str:
        """Clean generated code (remove markdown code block markers)."""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        
        answer = answer.lstrip("```python\n").lstrip("```\n").rstrip("\n```")
        return answer

    @staticmethod
    def record_to_target_answer(record: pd.Series) -> str:
        """Target answer is the test case."""
        return record['test']

    @staticmethod
    def get_domain() -> str:
        return 'humaneval'