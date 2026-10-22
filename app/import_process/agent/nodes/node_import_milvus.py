"""向现有兼容 Collection 实际 upsert；不建表、不删旧数据、不修改 Schema。

文档 ID 包含来源和内容；重复同版本幂等，不同版本保留，双表无跨表事务。
"""
import hashlib
import importlib
import os
from threading import Lock

_SDK_LOCK = Lock()

def get_sdk():
    # SDK 的隐式默认连接只支持远程 URI；实际连接始终显式使用 settings。
    # 暂时屏蔽默认连接自动初始化，退出后恢复进程环境，不修改 .env。
    with _SDK_LOCK:
        previous = os.environ.get("MILVUS_URI")
        os.environ["MILVUS_URI"] = ""
        try:
            return importlib.import_module("pymilvus")
        finally:
            if previous is None:
                os.environ.pop("MILVUS_URI", None)
            else:
                os.environ["MILVUS_URI"] = previous

from app.core.config import settings
from app.utils.task_utils import update_task_status, set_task_result, TASK_STATUS_COMPLETED
from ._runtime import import_node
from .node_bge_embedding import validate_vector


def get_client():
    return get_sdk().MilvusClient(uri=settings.milvus_uri, timeout=30)


def check_schema(client, collection, pk, rows):
    DataType = get_sdk().DataType
    if not client.has_collection(collection_name=collection):
        raise ValueError('Collection 不存在，请先提供或创建经确认的 Schema')
    schema = client.describe_collection(collection_name=collection)
    fields = {f['name']: f for f in schema['fields']}
    if schema.get('auto_id') or pk not in fields or not fields[pk].get('is_primary'):
        raise ValueError('主键配置不兼容')
    expected = {pk: DataType.VARCHAR, 'dense': DataType.FLOAT_VECTOR,
                'sparse': DataType.SPARSE_FLOAT_VECTOR}
    for key, value in rows[0].items():
        expected.setdefault(key, DataType.INT64 if isinstance(value, int) else DataType.VARCHAR)
    for key, dtype in expected.items():
        if key not in fields or fields[key]['type'] != dtype:
            raise ValueError('Collection 字段类型不兼容')
    if int(fields['dense'].get('params', {}).get('dim', 0)) != 1024:
        raise ValueError('Collection 向量维度不兼容')
    for key, field in fields.items():
        if key not in rows[0] and not field.get('nullable') and 'default_value' not in field:
            raise ValueError('Collection 存在额外必填字段')
    for row in rows:
        for key, value in row.items():
            if isinstance(value, str) and len(value.encode('utf-8')) > int(fields[key].get('params', {}).get('max_length', 65535)):
                raise ValueError('字符串超过 Collection 长度限制')


@import_node('node_import_milvus')
def node_import_milvus(state):
    state['import_result'] = {}
    chunks = state['embeddings_content']
    if not chunks or not state['item_name'].strip():
        raise ValueError('缺少入库数据')
    document_id = chunks[0]['metadata']['document_id']
    common = dict(file_title=state['file_title'], item_name=state['item_name'])
    item_rows = [dict(pk=document_id, **common,
                      **validate_vector(**state['item_embedding']))]
    chunk_rows = []
    for chunk in chunks:
        meta = chunk['metadata']
        if meta['document_id'] != document_id:
            raise ValueError('混合了多个文档')
        chunk_rows.append(dict(chunk_id=hashlib.sha256(f"{document_id}:{meta['part']}".encode()).hexdigest(),
            content=chunk['content'], title=meta['title'], parent_title=meta['parent_title'],
            part=meta['part'], **common, **validate_vector(chunk['dense'], chunk['sparse'])))
    if len({r['chunk_id'] for r in chunk_rows}) != len(chunk_rows):
        raise ValueError('切片 ID 重复')
    if settings.milvus_collection_item == settings.milvus_collection_chunk:
        raise ValueError('两层 Collection 不能同名')
    client = get_client()
    try:
        # 两张表都预检通过后才开始写入。
        check_schema(client, settings.milvus_collection_chunk, 'chunk_id', chunk_rows)
        check_schema(client, settings.milvus_collection_item, 'pk', item_rows)
        for name, rows in [(settings.milvus_collection_chunk, chunk_rows),
                           (settings.milvus_collection_item, item_rows)]:
            for offset in range(0, len(rows), 100):
                batch = rows[offset:offset + 100]
                result = client.upsert(collection_name=name, data=batch)
                if result.get('upsert_count') != len(batch):
                    raise RuntimeError('写入数量不一致，可能部分写入；同一输入可重试')
        client.flush(collection_name=settings.milvus_collection_chunk)
        client.flush(collection_name=settings.milvus_collection_item)
    finally:
        client.close()
    state['import_result'] = {'document_id': document_id, 'chunks': len(chunk_rows), 'items': 1}
    set_task_result(state['task_id'], 'document_id', document_id)
    update_task_status(state['task_id'], TASK_STATUS_COMPLETED)
    return state
