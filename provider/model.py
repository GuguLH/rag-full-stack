import os
import threading

from langchain_openai import ChatOpenAI


class ModelProvider:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        # 确保 __init__ 方法不会被直接调用，引导用户使用 get_instance()
        raise RuntimeError('调用 get_instance() 获取实例')

    @classmethod
    def get_instance(cls) -> ChatOpenAI:
        # 第一次检查：如果实例已存在，直接返回
        if cls._instance is None:
            # 加锁，确保线程安全
            with cls._lock:
                # 第二次检查：在锁内部再次检查，防止重复创建
                if cls._instance is None:
                    cls._instance = cls._create_instance()
        return cls._instance

    @classmethod
    def _create_instance(cls) -> ChatOpenAI:
        """
        私有方法，用于创建并初始化 ChatOpenAI 实例。
        """
        import dotenv
        dotenv.load_dotenv()
        api_key = os.getenv("OPEN_ROUTER_API_KEY")
        base_url = os.getenv("OPEN_ROUTER_URL")
        llm = ChatOpenAI(
            model="deepseek/deepseek-chat-v3-0324:free",
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            top_p=0.5
        )
        return llm
