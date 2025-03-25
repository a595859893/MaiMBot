import os
import shutil
import sys

import nonebot

from nonebot.adapters.qq import Adapter as QQAdapter  # 避免重复命名
from dotenv import load_dotenv
from loguru import logger

# 获取没有加载env时的环境变量
env_mask = {key: os.getenv(key) for key in os.environ}

uvicorn_server = None


def init_config():
    # 初次启动检测
    if not os.path.exists("config/bot_config.toml"):
        logger.warning("检测到bot_config.toml不存在，正在从模板复制")

        # 检查config目录是否存在
        if not os.path.exists("config"):
            os.makedirs("config")
            logger.info("创建config目录")

        shutil.copy("template/bot_config_template.toml", "config/bot_config.toml")
        logger.info("复制完成，请修改config/bot_config.toml和.env.prod中的配置后重新启动")


def init_env():
    # 初始化.env 默认ENVIRONMENT=prod
    if not os.path.exists(".env"):
        with open(".env", "w") as f:
            f.write("ENVIRONMENT=prod")

        # 检测.env.prod文件是否存在
        if not os.path.exists(".env.prod"):
            logger.error("检测到.env.prod文件不存在")
            shutil.copy("template.env", "./.env.prod")

    # 检测.env.dev文件是否存在，不存在的话直接复制生产环境配置
    if not os.path.exists(".env.dev"):
        logger.error("检测到.env.dev文件不存在")
        shutil.copy(".env.prod", "./.env.dev")

    # 首先加载基础环境变量.env
    if os.path.exists(".env"):
        load_dotenv(".env",override=True)
        logger.success("成功加载基础环境变量配置")


def load_env():
    # 使用闭包实现对加载器的横向扩展，避免大量重复判断
    def prod():
        logger.success("加载生产环境变量配置")
        load_dotenv(".env.prod", override=True)  # override=True 允许覆盖已存在的环境变量

    def dev():
        logger.success("加载开发环境变量配置")
        load_dotenv(".env.dev", override=True)  # override=True 允许覆盖已存在的环境变量

    fn_map = {
        "prod": prod,
        "dev": dev
    }

    env = os.getenv("ENVIRONMENT")
    logger.info(f"[load_env] 当前的 ENVIRONMENT 变量值：{env}")

    if env in fn_map:
        fn_map[env]()  # 根据映射执行闭包函数

    elif os.path.exists(f".env.{env}"):
        logger.success(f"加载{env}环境变量配置")
        load_dotenv(f".env.{env}", override=True)  # override=True 允许覆盖已存在的环境变量

    else:
        logger.error(f"ENVIRONMENT 配置错误，请检查 .env 文件中的 ENVIRONMENT 变量及对应 .env.{env} 是否存在")
        RuntimeError(f"ENVIRONMENT 配置错误，请检查 .env 文件中的 ENVIRONMENT 变量及对应 .env.{env} 是否存在")


def load_logger():
    logger.remove()  # 移除默认配置
    if os.getenv("ENVIRONMENT") == "dev":
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <fg #777777>|</> <level>{level: <7}</level> <fg "
                   "#777777>|</> <cyan>{name:.<8}</cyan>:<cyan>{function:.<8}</cyan>:<cyan>{line: >4}</cyan> <fg "
                   "#777777>-</> <level>{message}</level>",
            colorize=True,
            level=os.getenv("LOG_LEVEL", "DEBUG"),  # 根据环境设置日志级别，默认为DEBUG
        )
    else:
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> <fg #777777>|</> <level>{level: <7}</level> <fg "
                "#777777>|</> <cyan>{name:.<8}</cyan>:<cyan>{function:.<8}</cyan>:<cyan>{line: >4}</cyan> <fg "
                "#777777>-</> <level>{message}</level>",
            colorize=True,
            level=os.getenv("LOG_LEVEL", "INFO"),  # 根据环境设置日志级别，默认为INFO
            filter=lambda record: "nonebot" not in record["name"]
        )



def scan_provider(env_config: dict):
    provider = {}

    # 利用未初始化 env 时获取的 env_mask 来对新的环境变量集去重
    # 避免 GPG_KEY 这样的变量干扰检查
    env_config = dict(filter(lambda item: item[0] not in env_mask, env_config.items()))

    # 遍历 env_config 的所有键
    for key in env_config:
        # 检查键是否符合 {provider}_BASE_URL 或 {provider}_KEY 的格式
        if key.endswith("_BASE_URL") or key.endswith("_KEY"):
            # 提取 provider 名称
            provider_name = key.split("_", 1)[0]  # 从左分割一次，取第一部分

            # 初始化 provider 的字典（如果尚未初始化）
            if provider_name not in provider:
                provider[provider_name] = {"url": None, "key": None}

            # 根据键的类型填充 url 或 key
            if key.endswith("_BASE_URL"):
                provider[provider_name]["url"] = env_config[key]
            elif key.endswith("_KEY"):
                provider[provider_name]["key"] = env_config[key]

    # 检查每个 provider 是否同时存在 url 和 key
    for provider_name, config in provider.items():
        if config["url"] is None or config["key"] is None:
            logger.error(
                f"provider 内容：{config}\n"
                f"env_config 内容：{env_config}"
            )
            raise ValueError(f"请检查 '{provider_name}' 提供商配置是否丢失 BASE_URL 或 KEY 环境变量")

def raw_main():
    load_logger()
    init_config()
    init_env()
    load_env()
    load_logger()

    env_config = {key: os.getenv(key) for key in os.environ}
    scan_provider(env_config)

    # 合并配置
    nonebot.init(**env_config)

    # 注册适配器
    driver = nonebot.get_driver()
    driver.register_adapter(QQAdapter)

    # 加载插件
    nonebot.load_plugins("src/plugins")


if __name__ == "__main__":

    try:
        raw_main()

        nonebot.run()
    except KeyboardInterrupt:
        logger.warning("麦麦会努力做的更好的！正在停止中......")
    except Exception as e:
        logger.error(f"主程序异常: {e}")
    finally:
        logger.info("进程终止完毕，麦麦开始休眠......下次再见哦！")
