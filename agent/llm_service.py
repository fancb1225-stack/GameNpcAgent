from typing import Dict, List, Tuple
from openai import OpenAI
from anthropic import Anthropic

from agent_config import deepseek_api_key, openai_api_key

class BaseModel:
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

class DeepSeekV4Pro(BaseModel):
    def __init__(self, api_key: str = deepseek_api_key) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def chat(self, prompt: str, history: List[Dict[str, str]] = [], system_prompt: str = "") -> Tuple[
        str, List[Dict[str, str]]]:
        # 构建消息列表
        msg = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        ]

        # 添加历史消息
        if history:
            msg.extend(history)

        # 添加当前用户消息
        msg.append({"role": "user", "content": prompt})

        # print("==========================\n")
        # print(msg)
        # print("==========================\n")
        # 调用 API
        response = self.client.chat.completions.create(
            model="deepseek-v4-pro",
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

class DeepSeekV4Flash(BaseModel):
    def __init__(self, api_key: str = deepseek_api_key) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def chat(self, prompt: str, history: List[Dict[str, str]] = [], system_prompt: str = "") -> Tuple[
        str, List[Dict[str, str]]]:
        # 构建消息列表
        msg = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        ]

        # 添加历史消息
        if history:
            msg.extend(history)

        # 添加当前用户消息
        msg.append({"role": "user", "content": prompt})

        # print("==========================\n")
        # print(msg)
        # print("==========================\n")
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

class GLM(BaseModel):
    def __init__(self, api_key: str = openai_api_key) -> None:
        self.api_key = api_key
        self.client = Anthropic(api_key=api_key, base_url='https://ark.cn-beijing.volces.com/api/coding')

    def chat(self, prompt: str, history: List[Dict[str, str]] = [], system_prompt: str = "") -> Tuple[
        str, List[Dict[str, str]]]:

        # 构建消息列表
        msg = []

        # 添加历史消息
        if history:
            msg.extend(history)

        # 添加当前用户消息
        msg.append({"role": "user", "content": prompt})

        # 调用 API
        response = self.client.messages.create(
            model="glm-5.1",
            messages=msg,
            max_tokens=2048,
            system=system_prompt or "You are a helpful assistant.",
            stream=False,
        )

        # 只提取最终文本回答，过滤 ThinkingBlock
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        model_response = "\n".join(text_parts).strip()

        # 更新对话历史
        updated_history = msg.copy()
        updated_history.append({
            "role": "assistant",
            "content": model_response
        })

        return model_response, updated_history

class EmbeddingModel:
    def __init__(self, api_key: str = openai_api_key) -> None:
        self.api_key = api_key
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/coding/v3"
        )

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("输入文本为空，无法生成 embedding")

        response = self.client.embeddings.create(
            model="doubao-embedding-vision",
            input=text
        )

        return response.data[0].embedding

class Kimi(BaseModel):
    def __init__(self, api_key: str = openai_api_key) -> None:
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key, base_url='https://ark.cn-beijing.volces.com/api/coding/v3')

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

        # 调用 API
        response = self.client.chat.completions.create(
            model="kimi-k2.6",
            messages=msg,
            stream=False
        )

        model_response = response.choices[0].message.content

        # 更新对话历史
        updated_history = msg.copy()
        updated_history.append({"role": "assistant", "content": model_response})

        return model_response, updated_history

class Minimax(BaseModel):
    def __init__(self, api_key: str = openai_api_key) -> None:
        self.api_key = api_key
        self.client = Anthropic(api_key=api_key, base_url='https://ark.cn-beijing.volces.com/api/coding')

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
        msg = []

        # 添加历史消息
        if history:
            msg.extend(history)

        # 添加当前用户消息
        msg.append({"role": "user", "content": prompt})

        # 调用 API
        response = self.client.messages.create(
            model="minimax-m2.7",
            messages=msg,
            system=system_prompt or "You are a helpful assistant.",
            stream=False,  # 启用流式输出
            max_tokens=2048,  # 最大输出tokens
            temperature=0.2  # 控制输出的随机性
        )

        # 只提取最终文本回答，过滤 ThinkingBlock
        text_parts = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)

        model_response = "\n".join(text_parts).strip()

        # 更新对话历史
        updated_history = msg.copy()
        updated_history.append({
            "role": "assistant",
            "content": model_response
        })

        return model_response, updated_history

class LlmService:
    @staticmethod
    def getLLM():
        return DeepSeekV4Flash()

    @staticmethod
    def getDeepSeek_pro():
        return DeepSeekV4Pro()

    @staticmethod
    def getDeepSeek_flash():
        return DeepSeekV4Flash()

    @staticmethod
    def getGLM():
        return GLM()

    @staticmethod
    def getKimi():
        return Kimi()

    @staticmethod
    def getMinimax():
        return Minimax()

    @staticmethod
    def getEmbeddingModel():
        return EmbeddingModel()


if __name__ == "__main__":
    llm = LlmService.getEmbeddingModel()
    prompt = "你是什么模型"
    response = llm.embed(prompt)
    print("Response:", response[1])
    print(len(response))
