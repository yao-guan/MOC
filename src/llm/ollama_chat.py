import ollama
from typing import List, Union, Optional

from src.llm.format import Message
from src.llm.llm import LLM
from src.llm.llm_registry import LLMRegistry
from src.llm.price import cost_count_gemma

@LLMRegistry.register('OllamaChat')
class OllamaChat(LLM):
    """
    Generic Ollama model class, supports all downloaded Ollama models.

    Supported model families:
    - Llama: llama3.2:1b, llama3.2:3b, llama3.1:8b
    - Phi: phi3:mini, phi3:medium
    - Gemma: gemma2:2b, gemma2:9b, gemma2:27b
    - And any other models downloaded via ollama pull
    """

    def __init__(self, model_name: str = "gemma2:9b"):
        """
        Initialize Ollama model.

        Args:
            model_name: Model name in Ollama, e.g. "llama3.2:3b", "phi3:mini", "gemma2:9b".
        """
        self.model_name = model_name
        
        try:
            ollama.list()
            print(f"[OllamaChat] Initialized, using model: {model_name}")
        except Exception as e:
            print(f"[OllamaChat] Warning: Cannot connect to Ollama service: {e}")
            print(f"Please ensure the Ollama service is running")

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        """Async generation."""
        
        if max_tokens is None:
            max_tokens = self.DEFAULT_MAX_TOKENS
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
        if num_comps is None:
            num_comps = self.DEFUALT_NUM_COMPLETIONS

        ollama_messages = self._convert_messages(messages)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                },
            )

            response_text = response['message']['content']
            prompt_tokens = response.get('prompt_eval_count', 0)
            completion_tokens = response.get('eval_count', 0)
            cost_count_gemma(prompt_tokens,completion_tokens)

            return response_text

        except ollama.ResponseError as e:
            print(f"[OllamaChat] Ollama error: {e.error}")
            if e.status_code == 404:
                print(f"Hint: model {self.model_name} not found.")
                print(f"Run: ollama pull {self.model_name}")
                print(f"Or run 'ollama list' to see installed models")
            raise e
        except TimeoutError as e:
            print(f"[OllamaChat] Request timeout: {e}")
            raise e
        except Exception as e:
            print(f"[OllamaChat] Call failed: {e}")
            raise e

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

        ollama_messages = self._convert_messages(messages)

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=ollama_messages,
                stream=False,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                }
            )

            response_text = response['message']['content']
            return response_text

        except ollama.ResponseError as e:
            print(f"[OllamaChat] Ollama error: {e.error}")
            if e.status_code == 404:
                print(f"Hint: model {self.model_name} not found.")
                print(f"Run: ollama pull {self.model_name}")
                print(f"Or run 'ollama list' to see installed models")
            raise e
        except Exception as e:
            print(f"[OllamaChat] Call failed: {e}")
            raise e

    def _convert_messages(self, messages) -> List[dict]:
        """Convert messages to Ollama-compatible format."""
        ollama_messages = []
        
        if isinstance(messages, str):
            ollama_messages = [{'role': 'user', 'content': messages}]
        elif messages and isinstance(messages[0], Message):
            for msg in messages:
                ollama_messages.append({
                    'role': msg.role if msg.role in ['user', 'assistant', 'system'] else 'user',
                    'content': msg.content
                })
        elif messages and isinstance(messages[0], dict):
            ollama_messages = messages
        else:
            ollama_messages = [{'role': 'user', 'content': str(messages)}]
        
        return ollama_messages