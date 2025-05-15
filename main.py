import json
from openai import OpenAI
import re
import os
from datetime import datetime

client = OpenAI(api_key="sk-pro", 
                timeout=60,
                #model="mistral-small3.1:latest",
                base_url="http://10.100.1.1:11434/v1") 
system_prompt = """
你是一个专业的信息提取专家，请从以下文本中提取所有的主体、关系和客体，并以JSON列表的形式返回三元组。
关系应该是描述主体和客体之间联系的动词或动词短语。
请确保提取尽可能完整和准确的信息。
你是一个知识图谱构建专家，请从以下任意类型的自然语言文本中提取知识图谱中的实体（entities）和它们之间的关系（relations）。

要求：
1. 能够处理陈述句、疑问句、命令句等多种语言形式。
2. 对于问句或不确定性表达，提取其中的已知实体和潜在的关系（如“想知道某实体的属性”也构成一种关系）。
3. 输出格式必须为 JSON，结构如下：

{
  "entities": [
    {"id": "E1", "name": "实体名称", "type": "类型"},
    {"id": "E2", "name": "实体名称", "type": "类型"},
    ...
  ],
  "relations": [
    {"from": "E1", "to": "E2", "relation": "关系名称"},
    ...
  ]
}

其中：
- `name` 是实体名（如“爱因斯坦”、“相对论”）。
- `type` 是实体类型（如“人物”、“理论”、“公司”、“产品”等）。
- `relation` 是两个实体之间的语义关系（如“提出”、“属于”、“创立了”、“想了解”、“影响了”等）。
- 对于没有足够上下文确定的类型或关系，可以合理推测，并标记为“推测”或“未知”。
- 所有实体必须具备唯一 ID，如 E1, E2, E3...

请开始处理以下文本：
```

接着你只需将文本内容接在最后即可。

---

### ✅ 示例输入（用户提供文本）：

> 爱因斯坦提出了相对论，这一理论改变了人类对时间和空间的理解。

---

### ✅ 示例输出（JSON）：

```json
{
  "entities": [
    {"id": "E1", "name": "爱因斯坦", "type": "人物"},
    {"id": "E2", "name": "相对论", "type": "理论"},
    {"id": "E3", "name": "时间", "type": "概念"},
    {"id": "E4", "name": "空间", "type": "概念"},
    {"id": "E5", "name": "人类", "type": "群体"}
  ],
  "relations": [
    {"from": "E1", "to": "E2", "relation": "提出"},
    {"from": "E2", "to": "E3", "relation": "影响"},
    {"from": "E2", "to": "E4", "relation": "影响"},
    {"from": "E2", "to": "E5", "relation": "改变理解"}
  ]
}
```

"""
def extract_json_from_string(s):
    # 去除 markdown 的 ```json 和 ``` 标记
    cleaned_string = re.sub(r"```json[^\r\n]*\r?\n(.*?)\r?\n```", r"\1", s, flags=re.DOTALL)
    # 尝试进一步清理，保证只有JSON内容
    try:
        # 找到第一个 { 和最后一个 } 之间的内容
        start_idx = cleaned_string.find('{')
        end_idx = cleaned_string.rfind('}') + 1
        if start_idx >= 0 and end_idx > start_idx:
            cleaned_string = cleaned_string[start_idx:end_idx]
    except:
        pass
    return cleaned_string
    
def extract_information(text):
    completion = client.chat.completions.create(
        stream=False,
        model="mistral-small3.1:latest",
        temperature=0.1,
        messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": text
        }
        ]
    )
    if completion.choices[0].message.content:
        response_text = completion.choices[0].message.content
        return extract_json_from_string(response_text)
    else:
        return None


def extract_entities_relations_with_llm(text_input):
    """
    使用 LLM (通过精心设计的提示) 提取实体和关系。
    """
    # prompt = f"""
    # 请从以下文本中提取所有的主体、关系和客体，并以JSON列表的形式返回三元组。
    # 每个三元组应包含 "subject", "relation", "object" 三个键。
    # 关系应该是描述主体和客体之间联系的动词或动词短语。
    # 请确保提取尽可能完整和准确的信息。

    # 文本：
    # "{text_input}"

    # 提取结果 (JSON格式)：
    # """
    #: K8S的核心功能是什么 --> K8S 核心功能 
    try:
        response_text = extract_information(text_input)
        # 解析 JSON 字符串
        extracted_triples = json.loads(response_text)
        return extracted_triples
    except json.JSONDecodeError:
        print("错误: LLM 未返回有效的 JSON 格式。")
        return []

import thulac
import spacy

from typing import List
import langdetect  # 用于语言识别
# 初始化分词器
thu = thulac.thulac(seg_only=True)
nlp = spacy.load("en_core_web_sm")


# 加载模型
thu = thulac.thulac(seg_only=True)  # 默认模式
nlp = spacy.load("en_core_web_sm")  # 或者其他合适的英文模型

def split_text(text, max_len=128, overlap_size=16):
    """
    中英文混合文本分句函数，保证句子完整性
    
    Args:
        text: 输入文本 (字符串)
        max_len: 最大切分长度 (整数，默认为 128)
        overlap_size: 重叠大小 (整数，默认为 16)
    
    Returns:
        sentences: 分句后的句子列表 (字符串列表)
    """
    # 首先将整个文本分成句子
    all_sentences = []
    
    # 尝试检测整个文本的主要语言
    try:
        lang = langdetect.detect(text)
    except langdetect.LangDetectException:
        # 默认处理为中文
        lang = 'zh'
    
    # 中文文档使用正则表达式进行初步分句
    if lang == 'zh' or re.search(r'[\u4e00-\u9fff]', text):
        # 匹配中文标点符号作为句子边界
        raw_sentences = re.split(r'(?<=[。！？；.!?;])', text)
        # 进一步精细处理
        for raw_sent in raw_sentences:
            if not raw_sent.strip():
                continue
                
            # 检测单个句子的语言
            try:
                sent_lang = langdetect.detect(raw_sent)
            except langdetect.LangDetectException:
                sent_lang = 'zh'
                
            if sent_lang == 'zh' or re.search(r'[\u4e00-\u9fff]', raw_sent):
                # 中文句子
                all_sentences.extend(split_chinese(raw_sent))
            else:
                # 英文句子
                all_sentences.extend(split_english(raw_sent))
    else:
        # 英文文档
        all_sentences = split_english(text)
    
    # 过滤空句子
    all_sentences = [sent for sent in all_sentences if sent.strip()]
    
    # 根据max_len和overlap_size组合句子
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in all_sentences:
        # 如果单个句子超过max_len，需要特殊处理
        if len(sentence) > max_len:
            # 如果当前chunk不为空，先添加到chunks
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            # 长句子单独作为一个chunk
            chunks.append(sentence)
            continue
        
        # 判断加上当前句子是否超过最大长度
        if current_len + len(sentence) <= max_len:
            current_chunk.append(sentence)
            current_len += len(sentence)
        else:
            # 将当前chunk添加到结果中
            if current_chunk:
                chunks.append("".join(current_chunk))
            
            # 处理重叠
            if overlap_size > 0 and len(current_chunk) > overlap_size:
                # 保留最后几个句子作为重叠部分
                current_chunk = current_chunk[-overlap_size:]
                current_len = sum(len(s) for s in current_chunk)
            else:
                current_chunk = []
                current_len = 0
            
            # 添加当前句子到新chunk
            current_chunk.append(sentence)
            current_len += len(sentence)
    
    # 处理最后剩余的chunk
    if current_chunk:
        chunks.append("".join(current_chunk))
    
    return chunks

def split_chinese(segment):
    """
    使用 thulac 进行中文分句
    """
    sentences = []
    words = thu.cut(segment, text=True)  # 使用 text=True 返回文本形式
    
    # 可以根据实际情况对 punctuations 进行调整
    punctuations = ['。', '？', '！', '；', '…', '\n']  # 常见的中文句尾标点
    current_sentence = ''
    for word in words:
        current_sentence += word
        if word in punctuations:
            sentences.append(current_sentence.strip())
            current_sentence = ''
    if current_sentence:  # 处理最后一句
        sentences.append(current_sentence.strip())
    return sentences


def split_english(segment):
    """
    使用 spaCy 进行英文分句
    """
    doc = nlp(segment)
    sentences = [sent.text.strip() for sent in doc.sents]
    return sentences


#创建图数据库保存数据{'subject': '某个指标', 'predicate': '调整', 'object': '迭代'}
from py2neo import Graph, Node, Relationship
graph = Graph("bolt://10.100.2.1:7687", auth=("xxx", "xx"))

def get_entity_name_by_id(id, triples):
    for entity in triples['entities']:
        if entity['id'] == id:
            return entity['name']
    return None
def save_to_neo4j(triples):
    try:
        # 'entities' =
        # [{'id': 'E1', 'name': 'K8s', 'type': '技术'}, {'id': 'E2', 'name': 'K8S官网文档', 'type': '网址'}, {'id': 'E3', 'name': 'kubernetes.io/zh/docs/home', 'type': 'URL'}]
        # 'relations' =
        # [{'from': 'E1', 'to': 'E2', 'relation': '有'}, {'from': 'E2', 'to': 'E3', 'relation': '位于网址'}]
        # 检查triples数据格式
        if not isinstance(triples, dict) or 'entities' not in triples or 'relations' not in triples:
            print("错误：数据格式不正确，应为包含'entities'和'relations'键的字典")
            return


        # 创建实体节点
        for entity in triples['entities']:
            # 创建或获取实体节点
            entity_node = Node("Entity", name=entity['name'])
            graph.merge(entity_node, "Entity", "name")
        # 创建关系
        for relation in triples['relations']:
            # 获取关系两端的实体节点， 根据id获取name
            from_name = get_entity_name_by_id(relation['from'], triples)
            to_name = get_entity_name_by_id(relation['to'], triples)
            from_node = graph.nodes.match("Entity", name=from_name).first()
            to_node = graph.nodes.match("Entity", name=to_name).first()
            # 创建关系
            relation_node = Relationship(from_node, relation['relation'], to_node)
            graph.create(relation_node)

        print(f"成功保存 {len(triples['entities'])} 个实体到Neo4j")
        print(f"成功保存 {len(triples['relations'])} 个关系到Neo4j")
    except Exception as e:
        print(f"保存到Neo4j时出错: {e}")

def get_relation_type(text):
    match = re.search(r'-\[:(\w+)\s*\{.*?\}\]->', text)  #  \s* 匹配冒号后可能存在的空格

    if match:
        attributes = match.group(1)  # 提取第一个捕获组（花括号内的内容）
        return attributes
    else:
        return None

def create_sentence_generic(entry):
  """
  通用方法，将图数据库返回的条目转换为自然语言句子。
  不枚举关系类型。
  """
  print(entry)
#   'e' =
# Node('Entity', name='Deployment')
# 'r' =
# 属性(Node('Entity', name='无需持久存储'), Node('Entity', name='Deployment'))
# 'e2' =
# Node('Entity', name='无需持久存储')
  if entry:
    e1_node = entry['e']
    e1_name = e1_node['name']
    relation_type = get_relation_type(str(entry['r']))
    e2_node = entry['e2']
    e2_name = e2_node['name']
    # 通用句子结构：[实体1] [关系] [实体2]。可以根据需要调整。
    sentence = f"{e1_name} {relation_type} {e2_name}。"
    return sentence


def query_from_neo4j(query: str):
    try:
        print(f"查询Neo4j: {query}")
        # 把查询文本拆分为三元组
        triples = extract_entities_relations_with_llm(query)
        if not isinstance(triples, dict) or 'entities' not in triples or 'relations' not in triples:
            print("错误：数据格式不正确，应为包含'entities'和'relations'键的字典")
            return []
        
        all_results = []
        # 'entities' =
        # [{'id': 'E1', 'name': 'K8s', 'type': '技术'}, {'id': 'E2', 'name': 'K8S官网文档', 'type': '网址'}, {'id': 'E3', 'name': 'kubernetes.io/zh/docs/home', 'type': 'URL'}]
        # 'relations' =
        # [{'from': 'E1', 'to': 'E2', 'relation': '有'}, {'from': 'E2', 'to': 'E3', 'relation': '位于网址'}]
        # 根据问题 拆分元素， 生成cypher查询语句
        for triple in triples['entities']:
            # 查找实体节点的所有关联节点
            cypher_query = f"MATCH (e:Entity {{name: '{triple['name']}'}})-[r]-(e2:Entity) RETURN e,r, e2"
            print(f"执行查询: {cypher_query}")
            try:
                result = graph.run(cypher_query).data()
                if result:
                    all_results.extend(result)
            except Exception as e:
                print(f"执行单个查询时出错: {e}")

        # for relation in triples['relations']:
        #     from_name = get_entity_name_by_id(relation['from'], triples)
        #     to_name = get_entity_name_by_id(relation['to'], triples)
        #     relation_name = relation['relation']
        #     cypher_query = f"MATCH (e1:Entity {{name: '{from_name}'}})-[r:{relation_name}]->(e2:Entity {{name: '{to_name}'}}) RETURN e1,r, e2"
        #     print(f"执行查询: {cypher_query}")
        #     try:
        #         result = graph.run(cypher_query).data()
        #         if result:
        #             all_results.extend(result)
        #     except Exception as e:
        #         print(f"执行单个查询时出错: {e}")


        print(f"查询结果: {all_results}")
        return all_results
    except Exception as e:
        print(f"查询Neo4j时出错: {e}")
        return []


#支持打开文本和markdown文件
def load_text_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# --- 使用示例 ---
if __name__ == "__main__":
    text = load_text_from_file(r"C:\workspaces\python-projects\ai-kg-test\k8s-info.txt")
    
    max_len = 128
    overlap_size = 64
    sentences = split_text(text, max_len, overlap_size)
    all_triples = []
    
    for i, sentence in enumerate(sentences):
        print(f"Sentence {i+1}: {sentence}")
        triples1 = extract_entities_relations_with_llm(sentence)
        print(triples1)
        print("--------------------------------")
        save_to_neo4j(triples1)
    
    # 导出到JSON文件
    # 生成带时间戳的文件名
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"kg_triples_{timestamp}.json")
 
    # 写入JSON文件
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_triples, f, ensure_ascii=False, indent=2)
        print(f"\n已将三元组导出至: {output_file}")
    except Exception as e:
        print(f"导出JSON文件时出错: {e}")
    # 查询
    # data = query_from_neo4j("Deployment和Statefulset有什么区别")
    # sentences = [create_sentence_generic(entry) for entry in data]

    # # 打印生成的句子
    # for sentence in sentences:
    #     print(sentence)