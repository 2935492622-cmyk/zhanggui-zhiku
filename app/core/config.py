"""
app/core/config.py —— 全项目唯一的 .env 读取入口

【解决什么问题】
    每个节点都要读 .env（模型路径、Milvus 地址、切分参数、日志级别……）。
    教程的做法是 app/conf/ 下建 7 个 dataclass（lm_config / embedding_config /
    milvus_config / minio_config / mineru_config / reranker_config / bailian_mcp_config），
    每个模块自己 load_dotenv() + os.getenv()。

    那样写有三个代价，所以这里换成了 pydantic-settings：
      1. load_dotenv() 抄 7 遍，且 import 顺序会影响读到什么；
      2. os.getenv 拿不到值时**静默返回 None**，错误会推迟到很后面才炸，
         排查时你看到的是 "NoneType has no attribute"，不是 "某个配置没配"；
      3. 拿到的全是字符串，bool/float 要手动转（教程里就有
         `os.getenv("BGE_FP16") in ("1","True","true",1)` 这种写法）。

    pydantic-settings 的做法：字段在一个类里声明并带类型，启动时一次性校验并完成
    str→bool/int/float 转换，缺字段或类型不对立刻报错，错误信息直接指到字段名。

【上下游】
    上游：项目根目录的 .env（真实值，已被 .gitignore 忽略，不进仓库）
          .env.example（模板，进仓库，Ava 新环境时 cp .env.example .env 再填 key）
    下游：app/core/logger.py 读 LOG_*；后续所有节点读模型 / Milvus / 切分参数

【用法】
    from app.core.config import settings
    settings.chunk_max_size     # -> 1000（int，不是 "1000"）
    settings.doc_path           # -> 项目根/doc（派生属性，不用自己拼路径）
    settings.has_llm_key        # -> False（.env 里还是占位的 sk-xxxx 时）
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 本文件位置：项目根/app/core/config.py
# parents[0]=app/core  parents[1]=app  parents[2]=项目根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """从 .env 加载的全部配置。字段名 = .env 里的大写下划线名转小写。"""

    # pydantic-settings 的配置：读哪个文件、编码、遇到多余环境变量怎么办
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",  # 用绝对路径，不依赖当前工作目录
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里多出来的字段直接忽略，不报错
    )

    @field_validator("*", mode="before")
    @classmethod
    def _strip_inline_comment(cls, v):
        """
        为什么需要这个：
            python-dotenv **不会**剥离未加引号的行内注释。
            .env 里写 `LOG_FILE_LEVEL=INFO   # 控制台级别`，读到的值是
            "INFO   # 控制台级别"，喂给 loguru 会直接报「级别不存在」。
            这里对所有字符串字段统一砍掉 # 之后的内容，让你在 .env 里随手加注释也不会炸。
        代价：
            真正含 # 的值（比如密码 sk-#abc）会被截断。目前没有这种字段，
            真遇到了就给该字段加引号，dotenv 会自动 unquote。
        """
        if isinstance(v, str) and "#" in v:
            return v.split("#", 1)[0].strip()
        return v

    # ---------------- 运行环境 ----------------
    app_env: str = "dev"  # dev / test / prod
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # ---------------- 大模型（阿里云百炼，OpenAI 兼容模式）----------------
    # 控制台 https://bailian.console.aliyun.com → API-KEY 管理
    dashscope_api_key: str = ""  # 留空 / 占位 -> has_llm_key=False，走降级
    dashscope_base_url: str = "https://dashscope.aliyun.com/compatible-mode/v1"
    llm_model: str = "qwen-flash"  # 意图识别 / 查询改写 / HyDE / 答案生成
    vlm_model: str = "qwen-vl-flash"  # 图片理解（node_md_img 用）
    llm_temperature: float = 0.2  # 问答场景要稳，别调高
    llm_max_tokens: int = 2048

    # ---------------- Embedding（本地 BGE-M3，不走云端）----------------
    # 这是本项目唯一留在本地的模型：百炼的 embedding 只有稠密向量，
    # 而「稠密+稀疏混合」是核心卖点（稀疏向量负责精确命中型号 HAk180 / RS-12）
    bge_m3_path: str = "BAAI/bge-m3"  # 填本地权重目录可离线；留仓库 id 会自动下载
    embedding_device: str = "cuda"  # cuda / cpu；GPU 没配好时先退 cpu
    embedding_fp16: bool = True  # fp16 省一半显存，8GB 显存够用
    embedding_batch_size: int = 8
    embedding_max_len: int = 8192

    # ---------------- 重排序 ----------------
    rerank_model: str = "qwen3-rerank"  # 云端；也可换本地 BAAI/bge-reranker-v2-m3
    rerank_top_n: int = 5  # 最终送给大模型的条数
    rerank_clip_abs: float = 0.5  # 断崖检测：分数绝对差距阈值
    rerank_clip_rel: float = 0.25  # 断崖检测：分数相对差距阈值
    rerank_min_keep: int = 3
    rerank_max_keep: int = 10

    # ---------------- 向量库 Milvus（双层索引）----------------
    # M0-M2 用 Milvus Lite（单文件 ./data/milvus.db，零运维）；上生产再换 standalone URI
    milvus_uri: str = "./data/milvus.db"
    milvus_collection_item: str = "kb_item_names"  # 文档级索引：先筛出是哪本手册
    milvus_collection_chunk: str = "kb_chunks"  # 切片级索引：再定位到具体段落
    top_k_dense: int = 20
    top_k_sparse: int = 20
    rrf_k: int = 60  # RRF 融合公式的 k，行业惯例值

    # ---------------- MongoDB（元数据 / 解析结果 / 切片 / 任务 / 对话历史）----------------
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "zhanggui_zhiku"

    # ---------------- MinIO（原始文件与图片）----------------
    # M0 阶段关掉，用本地目录代替（见 parsed_path）
    minio_enabled: bool = False
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "zhiku"
    minio_secure: bool = False

    # ---------------- 文档切分 ----------------
    # 注意：这三个是「待 Ava 实测调参」的值，不是定论
    chunk_max_size: int = 1000  # 超过这个长度做二次切分
    chunk_min_size: int = 200  # 短于这个长度做贪心合并
    chunk_overlap_ratio: float = 0.15  # 10%~20% 重叠，保证跨切片的上下文连续

    # ---------------- 日志 ----------------
    log_console_enable: bool = True
    log_console_level: str = "INFO"
    log_file_enable: bool = True
    log_file_level: str = "INFO"  # 文件级别可与控制台不同，例如文件只记 ERROR
    log_file_retention: str = "7 days"
    log_fix_position: bool = True  # 修正日志显示的调用位置；压测时可关掉省开销

    # ==================== 派生属性：路径不用到处手拼 ====================
    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def doc_path(self) -> Path:
        """原始语料：88 本设备手册 PDF"""
        return _PROJECT_ROOT / "doc"

    @property
    def data_path(self) -> Path:
        return _PROJECT_ROOT / "data"

    @property
    def parsed_path(self) -> Path:
        """PDF 解析后的 Markdown 落盘位置（node_pdf_to_md 写这里）"""
        return self.data_path / "parsed"

    @property
    def log_path(self) -> Path:
        return _PROJECT_ROOT / "logs"

    @property
    def prompts_path(self) -> Path:
        """提示词模板目录（node_item_name_recognition 等节点会用到）"""
        return _PROJECT_ROOT / "prompts"

    @property
    def has_llm_key(self) -> bool:
        """
        判断大模型 key 是不是真的填了。
        为什么要这个：.env.example 里的占位值是 sk-xxxx...，
        直接用会报 401，但报错信息往往很晚才出现。
        节点里先判 has_llm_key，没 key 就走降级路径（不调 LLM），
        这样你现在没配 key 也能把整条链路跑通。
        """
        key = (self.dashscope_api_key or "").strip()
        return bool(key) and not key.startswith("sk-xxxx")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """带缓存的获取方法：多次调用只解析一次 .env"""
    return Settings()


# 模块级单例：绝大多数地方直接 `from app.core.config import settings`
settings = get_settings()


if __name__ == "__main__":
    # 自检：确认 .env 被正确读进来了（尤其是行内注释没污染值）
    s = settings
    print(f"项目根        : {s.project_root}")
    print(f"语料目录      : {s.doc_path}  (存在={s.doc_path.exists()})")
    print(f"日志级别      : 控制台={s.log_console_level!r} 文件={s.log_file_level!r}")
    print(f"切分参数      : max={s.chunk_max_size} min={s.chunk_min_size} overlap={s.chunk_overlap_ratio}")
    print(f"Embedding     : {s.bge_m3_path} device={s.embedding_device} fp16={s.embedding_fp16}")
    print(f"Milvus        : {s.milvus_uri} -> {s.milvus_collection_item} / {s.milvus_collection_chunk}")
    print(f"大模型 key 已配: {s.has_llm_key}")
