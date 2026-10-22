"""剩余节点：真实本地处理、明确 Mock 的模型/数据库测试、显式启用的真实模型测试。"""
from pathlib import Path
from types import SimpleNamespace
import os
import pytest
from app.import_process.agent.state import create_default_state
from app.utils.task_utils import get_task_status, get_running_task_list
import app.import_process.agent.nodes.node_md_img as images
import app.import_process.agent.nodes.node_document_split as split
import app.import_process.agent.nodes.node_item_name_recognition as names
import app.import_process.agent.nodes.node_bge_embedding as embedding
import app.import_process.agent.nodes.node_import_milvus as storage


def state(**kw):
    return create_default_state(task_id='remaining', file_title='HAK180', **kw)


def test_markdown_reads_and_normalizes_images(tmp_path):
    image = tmp_path / 'a b.png'
    image.write_bytes(b'image')
    path = tmp_path / 'manual.md'
    path.write_text('# Manual\n![switch](<a b.png>)\n![remote](https://example.org/a.png)\n`![code](a.png)`', encoding='utf-8')
    result = images.node_md_img(state(md_path=str(path)))
    assert image.as_uri() in result['md_content']
    assert 'https://example.org/a.png' in result['md_content']
    assert '`![code](a.png)`' in result['md_content']


@pytest.mark.parametrize('content', ['', '   '])
def test_empty_markdown_fails(tmp_path, content):
    path = tmp_path / 'manual.md'
    path.write_text(content)
    with pytest.raises(RuntimeError):
        images.node_md_img(state(md_path=str(path)))
    assert get_task_status('remaining') == 'failed'
    assert not get_running_task_list('remaining')


def test_split_preserves_offsets_and_coverage(monkeypatch):
    monkeypatch.setattr(split, 'settings', SimpleNamespace(chunk_max_size=80, chunk_min_size=20, chunk_overlap_ratio=.15))
    text = '<!-- page 1 -->\n# HAK180\n' + '安全操作说明。' * 40 + '\n## Maintenance\n' + '维修。' * 50
    result = split.node_document_split(state(md_content=text, local_file_path='/source/manual.md'))
    covered = set()
    for i, chunk in enumerate(result['chunks']):
        meta = chunk['metadata']
        assert chunk['content'] == text[meta['start']:meta['end']]
        assert len(chunk['content']) <= 80
        assert meta['part'] == i
        assert meta['source'] == '/source/manual.md'
        covered.update(range(meta['start'], meta['end']))
    assert covered == set(range(len(text)))


@pytest.mark.parametrize('ratio', [-1, 1])
def test_split_rejects_parameters(monkeypatch, ratio):
    monkeypatch.setattr(split, 'settings', SimpleNamespace(chunk_max_size=80, chunk_min_size=20, chunk_overlap_ratio=ratio))
    with pytest.raises(RuntimeError):
        split.node_document_split(state(md_content='body'))


@pytest.mark.mock
def test_rules_do_not_call_cloud(monkeypatch):
    monkeypatch.setattr(names, 'settings', SimpleNamespace(has_llm_key=False))
    assert names.recognize('# HAK180 烫金机\n正文', 'manual') == 'HAK180 烫金机'
    assert names.recognize('设备型号 HAK180，操作方法', 'manual') == 'HAK180'


@pytest.mark.parametrize('dense,sparse', [([1.]*3, {1:1}), ([float('nan')]*1024,{1:1}),
    ([0.]*1024,{1:1}), ([1.]*1024,{}), ([1.]*1024,{-1:1}), ([1.]*1024,{1:float('inf')})])
def test_vector_validation_rejects_invalid(dense, sparse):
    with pytest.raises(ValueError):
        embedding.validate_vector(dense,sparse)


class Model:
    tokenizer = staticmethod(lambda text, **kw: {'input_ids': [1,2]})
    def encode(self, texts, **kw):
        return {'dense_vecs': [[1.]*1024 for t in texts], 'lexical_weights': [{'1':.5} for t in texts]}


@pytest.mark.mock
def test_embedding_includes_separate_item_vector(monkeypatch):
    monkeypatch.setattr(embedding, 'get_model', lambda *a: Model())
    result = embedding.node_bge_embedding(state(item_name='HAK180', chunks=[{'content':'body','metadata':{}}]))
    assert len(result['item_embedding']['dense']) == 1024
    assert len(result['embeddings_content']) == 1
    assert result['embeddings_content'][0]['sparse'] == {1:.5}


@pytest.mark.mock
def test_embedding_count_mismatch_does_not_publish_output(monkeypatch):
    model = Model()
    model.encode = lambda *a, **kw: {'dense_vecs': [], 'lexical_weights': []}
    monkeypatch.setattr(embedding, 'get_model', lambda *a: model)
    s = state(item_name='item', chunks=[{'content':'body'}])
    with pytest.raises(RuntimeError):
        embedding.node_bge_embedding(s)
    assert s['embeddings_content'] == [] and s['item_embedding'] == {}


class Client:
    def __init__(self): self.calls=[]; self.closed=False
    def has_collection(self, **kw): return True
    def describe_collection(self, collection_name):
        D = storage.get_sdk().DataType
        pk = 'pk' if collection_name.endswith('names') else 'chunk_id'
        fields = [{'name':pk,'type':D.VARCHAR,'is_primary':True,'params':{'max_length':64}},
                  {'name':'dense','type':D.FLOAT_VECTOR,'params':{'dim':1024}},
                  {'name':'sparse','type':D.SPARSE_FLOAT_VECTOR},
                  *[{'name':n,'type':D.VARCHAR,'params':{'max_length':65535}}
                    for n in ['file_title','item_name']]]
        if pk == 'chunk_id':
            fields += [{'name':n,'type':D.VARCHAR} for n in ['content','title','parent_title']]
            fields += [{'name':'part','type':D.INT64}]
        return {'auto_id':False,'fields':fields}
    def upsert(self, collection_name, data):
        self.calls.append((collection_name,data))
        return {'upsert_count':len(data)}
    def flush(self, **kw): pass
    def close(self): self.closed=True


def ready_state():
    s = state(item_name='HAK180', item_embedding={'dense':[1.]*1024,'sparse':{1:1.}})
    s['embeddings_content'] = [{'content':'body', 'metadata':{'document_id':'a'*64,'part':0,
        'title':'manual','parent_title':''},'dense':[1.]*1024,'sparse':{1:1.}}]
    return s


@pytest.mark.mock
def test_milvus_preflight_and_acknowledged_write(monkeypatch):
    client=Client()
    monkeypatch.setattr(storage, 'get_client', lambda:client)
    result=storage.node_import_milvus(ready_state())
    assert result['import_result']['chunks'] == 1
    assert len(client.calls)==2 and client.closed
    assert get_task_status('remaining') == 'completed'


@pytest.mark.mock
@pytest.mark.parametrize('failure', ['missing','schema','write'])
def test_milvus_failure_is_not_success(monkeypatch, failure):
    client=Client()
    if failure=='missing': client.has_collection=lambda **kw:False
    if failure=='schema': client.describe_collection=lambda **kw:{'fields':[]}
    if failure=='write': client.upsert=lambda **kw:{'upsert_count':0}
    monkeypatch.setattr(storage,'get_client',lambda:client)
    s=ready_state()
    with pytest.raises(RuntimeError): storage.node_import_milvus(s)
    assert client.closed and not s['import_result']
    assert get_task_status('remaining')=='failed'
    if failure!='write': assert not client.calls


@pytest.mark.integration
@pytest.mark.skipif(os.getenv('RUN_BGE_INTEGRATION')!='1', reason='显式设置 RUN_BGE_INTEGRATION=1 才加载真实权重')
def test_real_bge_embedding():
    result=embedding.node_bge_embedding(state(item_name='HAK180',chunks=[{'content':'更换色带前断开电源','metadata':{}}]))
    assert len(result['embeddings_content'][0]['dense'])==1024


@pytest.mark.mock
@pytest.mark.parametrize('answer,valid', [('HAK180', True), ('invented product', False)])
def test_cloud_name_must_exist_in_source(monkeypatch, answer, valid):
    import openai
    class FakeOpenAI:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        @property
        def chat(self): return self
        @property
        def completions(self): return self
        def create(self, **kw):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])
    monkeypatch.setattr(openai, 'OpenAI', FakeOpenAI)
    monkeypatch.setattr(names, 'settings', SimpleNamespace(has_llm_key=True,
        dashscope_api_key='test-only', dashscope_base_url='https://example.invalid', llm_model='test'))
    if valid:
        assert names.recognize('HAK180 操作说明', 'manual') == answer
    else:
        with pytest.raises(ValueError): names.recognize('HAK180 操作说明', 'manual')


@pytest.mark.mock
def test_overlong_embedding_is_not_silently_truncated(monkeypatch):
    model=Model()
    model.tokenizer=lambda *a, **kw:{'input_ids':[1]*9000}
    monkeypatch.setattr(embedding, 'get_model', lambda *a:model)
    with pytest.raises(RuntimeError):
        embedding.node_bge_embedding(state(item_name='item',chunks=[{'content':'body'}]))
