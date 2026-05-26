import aiohttp
from typing import List, Union, Optional
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
import os

from src.llm.format import Message
from src.llm.llm import LLM
from src.llm.llm_registry import LLMRegistry
from src.llm.price import cost_count_deepseek

load_dotenv()
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat_deepseek(
    model: str, 
    msg: List[dict],
    temperature: Optional[float] = None):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }
    data = {
        "model": model,
        "messages": msg,
        "stream": False
    }

    if temperature is not None:
        data["temperature"] = temperature
    async with aiohttp.ClientSession() as session:
        async with session.post(DEEPSEEK_BASE_URL, headers=headers, json=data) as response:
            response_text = await response.text()
            if response.status != 200:
                raise Exception(f"API returned status {response.status}: {response_text}")
            
            response_data = await response.json()
            content = response_data['choices'][0]['message']['content']
            
            prompt_tokens = response_data['usage']['prompt_tokens']
            completion_tokens = response_data['usage']['completion_tokens']
            cost_count_deepseek(prompt_tokens, completion_tokens, model)
            
            return content


@LLMRegistry.register('DeepSeekChat')
class DeepSeekChat(LLM):
    def __init__(self, model_name: str = "deepseek-chat"):
        self.model_name = model_name

    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        
        if temperature is None:
            temperature = self.DEFAULT_TEMPERATURE
            
        if isinstance(messages, str):
            messages = [Message(role="user", content=messages)]

        return await achat_deepseek(self.model_name, messages,temperature)
    
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        num_comps: Optional[int] = None,
    ) -> Union[List[str], str]:
        pass
