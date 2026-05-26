import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Union, Optional

from src.llm.format import Message
from src.llm.llm import LLM
from src.llm.llm_registry import LLMRegistry
from src.llm.price import cost_count_llama


@LLMRegistry.register('LlamaChat')
class LlamaChat(LLM):
    """
    Local Llama model class, supports loading local model files directly.

    Supported models: Llama 3.3-70B and other large language models.
    """

    def __init__(self, model_path: str, device_map: str = "auto", load_in_8bit: bool = False):
        """
        Initialize local Llama model.

        Args:
            model_path: Path to local model files, e.g. "/path/to/llama-3.3-70b".
            device_map: Device mapping strategy, "auto" for automatic, "cuda:0" for specific GPU.
            load_in_8bit: Whether to use 8-bit quantization (saves VRAM).
        """
        self.model_path = model_path
        self.device_map = device_map
        self.load_in_8bit = load_in_8bit
        self.model_name = "llama-local"  
        
        print(f"[LlamaChat] Loading model: {model_path}")
        print(f"[LlamaChat] Device map: {device_map}")
        print(f"[LlamaChat] 8-bit quantization: {load_in_8bit}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                use_fast=False
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            load_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "device_map": device_map,
                "trust_remote_code": True,
                "torch_dtype": torch.float16,
            }
            
            if load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            self.model.eval()
            
            print(f"[LlamaChat] Model loaded successfully")
            
        except Exception as e:
            print(f"[LlamaChat] Model loading failed: {e}")
            raise e

    def _convert_messages_to_prompt(self, messages) -> str:
        """Convert message list to Llama-format prompt."""
        if isinstance(messages, str):
            return messages
        
        prompt_parts = []
        for msg in messages:
            role = msg.role if hasattr(msg, 'role') else msg.get('role', 'user')
            content = msg.content if hasattr(msg, 'content') else msg.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"<|system|>\n{content}</s>\n")
            elif role == 'user':
                prompt_parts.append(f"<|user|>\n{content}</s>\n")
            elif role == 'assistant':
                prompt_parts.append(f"<|assistant|>\n{content}</s>\n")

        prompt_parts.append("<|assistant|>\n")
        return "".join(prompt_parts)

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Async generation (delegates to sync implementation)."""
        return self.gen(messages, max_tokens, temperature, num_comps)

    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Synchronous generation."""
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        prompt = self._convert_messages_to_prompt(messages)
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096
            )
            
            if self.device_map == "auto":
                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
            else:
                input_ids = inputs.input_ids.to(self.device_map)
                attention_mask = inputs.attention_mask.to(self.device_map)
            
            prompt_tokens = input_ids.shape[1]
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_tokens = outputs.shape[1] - prompt_tokens
            response_text = self.tokenizer.decode(
                outputs[0][prompt_tokens:],
                skip_special_tokens=True
            )
            
            cost_count_llama(prompt_tokens, generated_tokens, self.model_name)
            
            return response_text.strip()
            
        except Exception as e:
            print(f"[LlamaChat] Generation failed: {e}")
            raise e