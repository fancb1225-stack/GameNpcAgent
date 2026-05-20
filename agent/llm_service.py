import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
try:
    from anthropic import Anthropic
except ModuleNotFoundError:
    Anthropic = None

try:
    from agent.agent_config import deepseek_api_key, openai_api_key
except ModuleNotFoundError:
    from agent_config import deepseek_api_key, openai_api_key


# 加载本地 .env，Docker 环境中已注入的变量会保持优先级。
load_dotenv()


def _get_env_config(
    api_key_env: str,
    base_url_env: str,
    model_env: str,
    default_api_key: str,
    default_base_url: str,
    default_model: str,
) -> Tuple[str, str, str]:
    # 统一读取模型配置，避免本地和 Docker 走不同配置来源。
    api_key = os.getenv(api_key_env) or default_api_key
    base_url = os.getenv(base_url_env) or default_base_url
    model = os.getenv(model_env) or default_model

    return api_key, base_url, model


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
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        # 优先使用显式参数，其次读取 Docker 或本地 .env 中的配置。
        env_api_key, env_base_url, env_model = _get_env_config(
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL",
            deepseek_api_key,
            "https://api.deepseek.com",
            "deepseek-v4-pro",
        )
        self.api_key = api_key or env_api_key
        self.base_url = base_url or env_base_url
        self.model = model or env_model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
            model=self.model,
            messages=msg,
            stream=False,
            reasoning_effort="high",
            extra_body={"thinking": {"type": "enabled"}}
        )

        model_response = response.choices[0].message.content

        # thinking 模式下 content 可能为 None，实际回答在 reasoning_content
        if not model_response:
            rc = getattr(response.choices[0].message, "reasoning_content", None)
            if rc:
                model_response = rc

        # 更新对话历史
        updated_history = msg.copy()
        updated_history.append({"role": "assistant", "content": model_response or ""})

        return model_response or "", updated_history

class DeepSeekV4Flash(BaseModel):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        # 优先使用显式参数，其次读取 Docker 或本地 .env 中的配置。
        env_api_key, env_base_url, env_model = _get_env_config(
            "LLM_API_KEY",
            "LLM_BASE_URL",
            "LLM_MODEL",
            deepseek_api_key,
            "https://api.deepseek.com",
            "deepseek-v4-flash",
        )
        self.api_key = api_key or env_api_key
        self.base_url = base_url or env_base_url
        self.model = model or env_model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

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
        # 调用 API（flash 模型不支持 reasoning_effort / thinking）
        response = self.client.chat.completions.create(
            model=self.model,
            messages=msg,
            stream=False,
        )

        model_response = response.choices[0].message.content

        # 更新对话历史
        updated_history = msg.copy()
        updated_history.append({"role": "assistant", "content": model_response})

        return model_response, updated_history

class GLM(BaseModel):
    def __init__(self, api_key: str = openai_api_key) -> None:
        if Anthropic is None:
            raise RuntimeError("缺少 anthropic 依赖，无法创建 GLM 模型")
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
        if Anthropic is None:
            raise RuntimeError("缺少 anthropic 依赖，无法创建 Minimax 模型")
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
            temperature=0.6  # 控制输出的随机性
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
    llm = LlmService.getDeepSeek_pro()
    prompt = "你是什么模型"
    response = llm.chat(prompt)
    print("Response:", response[1])
    print(len(response))
