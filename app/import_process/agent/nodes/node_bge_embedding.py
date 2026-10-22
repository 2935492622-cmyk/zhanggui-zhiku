"""真实 BGE-M3：切片和品名均生成 dense/sparse，统一为 Python 标量。"""
import math
from functools import lru_cache
from app.core.config import settings
from ._runtime import import_node


@lru_cache(maxsize=1)
def get_model(path, device, fp16):
    from FlagEmbedding import BGEM3FlagModel
    return BGEM3FlagModel(path, devices=device, use_fp16=fp16)


def validate_vector(dense, sparse):
    dense = [float(v) for v in dense]
    sparse = {int(k): float(v) for k, v in sparse.items()}
    if len(dense) != 1024 or not all(math.isfinite(v) for v in dense) or not any(dense):
        raise ValueError('BGE-M3 dense 必须为非零有限的 1024 维向量')
    if not sparse or any(k < 0 or not math.isfinite(v) or v < 0 for k, v in sparse.items()):
        raise ValueError('sparse 格式错误')
    return {'dense': dense, 'sparse': sparse}


@import_node('node_bge_embedding')
def node_bge_embedding(state):
    state.update(embeddings_content=[], item_embedding={})
    chunks = state['chunks']
    texts = [state['item_name']] + [c['content'] for c in chunks]
    if not chunks or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError('品名或切片为空')
    if settings.embedding_batch_size < 1 or settings.embedding_max_len < 1:
        raise ValueError('Embedding 参数错误')
    model = get_model(settings.bge_m3_path, settings.embedding_device,
                      settings.embedding_fp16 and settings.embedding_device != 'cpu')
    # 防止 encode 静默截断正文。
    for text in texts:
        if len(model.tokenizer(text, truncation=False)['input_ids']) > settings.embedding_max_len:
            raise ValueError('切片超过模型 token 上限，请减小切片长度')
    vectors = []
    size = settings.embedding_batch_size
    for start in range(0, len(texts), size):
        batch = texts[start:start + size]
        output = model.encode(batch, batch_size=size, max_length=settings.embedding_max_len,
                              return_dense=True, return_sparse=True, return_colbert_vecs=False)
        if len(output['dense_vecs']) != len(batch) or len(output['lexical_weights']) != len(batch):
            raise ValueError('向量数量不一致')
        vectors.extend(validate_vector(d, s) for d, s in zip(output['dense_vecs'], output['lexical_weights']))
    state['item_embedding'] = vectors[0]
    state['embeddings_content'] = [dict(chunk, **vector) for chunk, vector in zip(chunks, vectors[1:])]
    return state
