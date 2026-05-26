from typing import Optional
from class_registry import ClassRegistry

from src.llm.llm import LLM


class LLMRegistry:
    registry = ClassRegistry()

    @classmethod
    def register(cls, *args, **kwargs):
        return cls.registry.register(*args, **kwargs)
    
    @classmethod
    def keys(cls):
        return cls.registry.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        if model_name is None or model_name=="":
            model_name = "gpt-4o"

        if model_name == 'mock':
            model = cls.registry.get(model_name)
        elif 'deepseek' in model_name.lower():  # DeepSeek models
            model = cls.registry.get('DeepSeekChat', model_name)
        elif any(ollama_model in model_name.lower() for ollama_model in [
            'llama', 'phi', 'gemma', 'qwen', 'mistral'
        ]):
            model = cls.registry.get('OllamaChat', model_name)
        else: 
            model = cls.registry.get('GPTChat', model_name)

        return model
