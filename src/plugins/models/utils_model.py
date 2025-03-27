import asyncio
import json
import re
from datetime import datetime
from typing import Tuple, Union
from openai import AsyncClient

import aiohttp
from loguru import logger
from nonebot import get_driver
import base64
from PIL import Image
import io
from ...common.database import Database
from ..chat.config import global_config

driver = get_driver()
config = driver.config


class LLM_request:
    def __init__(self, model, **kwargs):
        # 将大写的配置键转换为小写并从config中获取实际值
        try:
            self.client = AsyncClient(
                api_key=getattr(config, model["key"]),
                base_url=getattr(config, model["base_url"]),
            )
        except AttributeError as e:
            logger.error(f"原始 model dict 信息：{model}")
            logger.error(f"配置错误：找不到对应的配置项 - {str(e)}")
            raise ValueError(f"配置错误：找不到对应的配置项 - {str(e)}") from e
        self.model_name = model["name"]
        self.params = kwargs

        self.pri_in = model.get("pri_in", 0)
        self.pri_out = model.get("pri_out", 0)

        # 获取数据库实例
        self.db = Database.get_instance()
        self._init_database()

    def _init_database(self):
        """初始化数据库集合"""
        try:
            # 创建llm_usage集合的索引
            self.db.db.llm_usage.create_index([("timestamp", 1)])
            self.db.db.llm_usage.create_index([("model_name", 1)])
            self.db.db.llm_usage.create_index([("user_id", 1)])
            self.db.db.llm_usage.create_index([("request_type", 1)])
        except Exception:
            logger.error("创建数据库索引失败")

    def _record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        user_id: str = "system",
        request_type: str = "chat",
        endpoint: str = "/chat/completions",
    ):
        """记录模型使用情况到数据库
        Args:
            prompt_tokens: 输入token数
            completion_tokens: 输出token数
            total_tokens: 总token数
            user_id: 用户ID，默认为system
            request_type: 请求类型(chat/embedding/image等)
            endpoint: API端点
        """
        try:
            usage_data = {
                "model_name": self.model_name,
                "user_id": user_id,
                "request_type": request_type,
                "endpoint": endpoint,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": self._calculate_cost(prompt_tokens, completion_tokens),
                "status": "success",
                "timestamp": datetime.now(),
            }
            self.db.db.llm_usage.insert_one(usage_data)
            logger.info(
                f"Token使用情况 - 模型: {self.model_name}, "
                f"用户: {user_id}, 类型: {request_type}, "
                f"提示词: {prompt_tokens}, 完成: {completion_tokens}, "
                f"总计: {total_tokens}"
            )
        except Exception:
            logger.error("记录token使用情况失败")

    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """计算API调用成本
        使用模型的pri_in和pri_out价格计算输入和输出的成本

        Args:
            prompt_tokens: 输入token数量
            completion_tokens: 输出token数量

        Returns:
            float: 总成本（元）
        """
        # 使用模型的pri_in和pri_out计算成本
        input_cost = (prompt_tokens / 1000000) * self.pri_in
        output_cost = (completion_tokens / 1000000) * self.pri_out
        return round(input_cost + output_cost, 6)

    async def _execute_request(
        self,
        prompt: str = None,
        retry_policy: dict = None,
        image_base64: str = None,
    ):
        """统一请求执行入口
        Args:
            endpoint: API端点路径 (如 "chat/completions")
            prompt: prompt文本
            image_base64: 图片的base64编码
            payload: 请求体数据
            retry_policy: 自定义重试策略
            response_handler: 自定义响应处理器
            user_id: 用户ID
            request_type: 请求类型
        """
        # 合并重试策略
        default_retry = {
            "max_retries": 3,
            "base_wait": 15,
            "retry_codes": [429, 413, 500, 503],
            "abort_codes": [400, 401, 402, 403],
        }
        policy = {**default_retry, **(retry_policy or {})}

        # 常见Error Code Mapping
        error_code_mapping = {
            400: "参数不正确",
            401: "API key 错误，认证失败",
            402: "账号余额不足",
            403: "需要实名,或余额不足",
            404: "Not Found",
            429: "请求过于频繁，请稍后再试",
            500: "服务器内部故障",
            503: "服务器负载过高",
        }
        content = []
        if prompt is not None:
            content.append({"type": "text", "text": prompt})

        if image_base64 is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }
            )
        else:
            # 似乎有些模型还不支持openai的新版本格式，对于这部分使用旧版本处理
            content = prompt

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "max_tokens": global_config.max_response_length,
        }
        logger.info(f"url:{self.client.base_url}, 请求体: {payload}")
        for retry in range(policy["max_retries"]):
            try:
                # TODO: 感觉后续可以基于状态调整temperature
                response = await self.client.chat.completions.create(**payload)
                message = response.choices[0].message
                content = message.content
                reasoning_content = (
                    message.reasoning_content
                    if hasattr(message, "reasoning_content")
                    else ""
                )
                return content, reasoning_content
            except Exception as e:
                if retry < policy["max_retries"] - 1:
                    wait_time = policy["base_wait"] * (2**retry)
                    logger.error(f"请求失败，等待{wait_time}秒后重试... 错误: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.critical(f"请求失败: {str(e)}")
                    logger.critical(
                        f"请求头: {await self._build_headers(no_key=True)} 请求体: {payload}"
                    )
                    raise RuntimeError(f"API请求失败: {str(e)}")

        logger.error("达到最大重试次数，请求仍然失败")
        raise RuntimeError("达到最大重试次数，API请求仍然失败")

    async def generate_response(self, prompt: str) -> Tuple[str, str]:
        """根据输入的提示生成模型的异步响应"""

        content, reasoning_content = await self._execute_request(prompt=prompt)
        return content, reasoning_content

    async def generate_response_for_image(
        self, prompt: str, image_base64: str
    ) -> Tuple[str, str]:
        """根据输入的提示和图片生成模型的异步响应"""

        content, reasoning_content = await self._execute_request(
            prompt=prompt, image_base64=image_base64
        )
        return content, reasoning_content

    async def generate_response_async(
        self, prompt: str, **kwargs
    ) -> Union[str, Tuple[str, str]]:
        """异步方式根据输入的提示生成模型的响应"""
        # 构建请求体
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": global_config.max_response_length,
            **self.params,
        }

        content, reasoning_content = await self._execute_request(
            payload=data, prompt=prompt
        )
        return content, reasoning_content

    async def get_embedding(self, text: str) -> Union[list, None]:
        """异步方法：获取文本的embedding向量

        Args:
            text: 需要获取embedding的文本

        Returns:
            list: embedding向量，如果失败则返回None
        """
        payload = {
            "model": "text-embedding-v3",
            "input": text,
            "dimensions": 1024,
            "encoding_format": "float",
        }
        embedding = await self.client.embeddings.create(**payload)
        return embedding.data[0].embedding


def compress_base64_image_by_scale(
    base64_data: str, target_size: int = 0.8 * 1024 * 1024
) -> str:
    """压缩base64格式的图片到指定大小
    Args:
        base64_data: base64编码的图片数据
        target_size: 目标文件大小（字节），默认0.8MB
    Returns:
        str: 压缩后的base64图片数据
    """
    try:
        # 将base64转换为字节数据
        image_data = base64.b64decode(base64_data)

        # 如果已经小于目标大小，直接返回原图
        if len(image_data) <= 2 * 1024 * 1024:
            return base64_data

        # 将字节数据转换为图片对象
        img = Image.open(io.BytesIO(image_data))

        # 获取原始尺寸
        original_width, original_height = img.size

        # 计算缩放比例
        scale = min(1.0, (target_size / len(image_data)) ** 0.5)

        # 计算新的尺寸
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # 创建内存缓冲区
        output_buffer = io.BytesIO()

        # 如果是GIF，处理所有帧
        if getattr(img, "is_animated", False):
            frames = []
            for frame_idx in range(img.n_frames):
                img.seek(frame_idx)
                new_frame = img.copy()
                new_frame = new_frame.resize(
                    (new_width // 2, new_height // 2), Image.Resampling.LANCZOS
                )  # 动图折上折
                frames.append(new_frame)

            # 保存到缓冲区
            frames[0].save(
                output_buffer,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                optimize=True,
                duration=img.info.get("duration", 100),
                loop=img.info.get("loop", 0),
            )
        else:
            # 处理静态图片
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # 保存到缓冲区，保持原始格式
            if img.format == "PNG" and img.mode in ("RGBA", "LA"):
                resized_img.save(output_buffer, format="PNG", optimize=True)
            else:
                resized_img.save(
                    output_buffer, format="JPEG", quality=95, optimize=True
                )

        # 获取压缩后的数据并转换为base64
        compressed_data = output_buffer.getvalue()
        logger.success(
            f"压缩图片: {original_width}x{original_height} -> {new_width}x{new_height}"
        )
        logger.info(
            f"压缩前大小: {len(image_data)/1024:.1f}KB, 压缩后大小: {len(compressed_data)/1024:.1f}KB"
        )

        return base64.b64encode(compressed_data).decode("utf-8")

    except Exception as e:
        logger.error(f"压缩图片失败: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return base64_data
