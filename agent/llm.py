from typing import Dict, List, Tuple
from openai import OpenAI

from .agent_config import llm_api_key

class BaseModel:
    def __init__(self, api_key: str = '') -> None:
        self.api_key = api_key

    def chat(self, prompt: str, history: List[Dict[str, str]], system_prompt: str = "") -> Tuple[
        str, List[Dict[str, str]]]:
        """
        基础聊天接口

        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示

        Returns:
            (模型响应, 更新后的对话历史)
        """
        pass


class LlmService(BaseModel):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def chat(self, prompt: str, history: List[Dict[str, str]] = [], system_prompt: str = "") -> Tuple[
        str, List[Dict[str, str]]]:
        """
        与 LlmService API 进行聊天

        Args:
            prompt: 用户输入
            history: 对话历史
            system_prompt: 系统提示

        Returns:
            (模型响应, 更新后的对话历史)
        """
        # 构建消息列表
        msg = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        ]

        # 添加历史消息
        if history:
            msg.extend(history)

        # 添加当前用户消息
        msg.append({"role": "user", "content": prompt})

        print("==========================\n")
        print(msg)
        print("==========================\n")
        # 调用 API
        response = self.client.chat.completions.create(
            model="deepseek-v4-flash",
            messages=msg,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )

        model_response = response.choices[0].message.content

        # 更新对话历史
        updated_history = msg.copy()
        updated_history.append({"role": "assistant", "content": model_response})

        return model_response, updated_history

    @staticmethod
    def getLLM():
        model = LlmService(api_key=llm_api_key)

        return model


if __name__ == "__main__":
    llm = LlmService.getLLM()
    prompt = "你好"
    response, history = llm.chat(prompt)
    print("Response:", response)
    print("History:", history)
