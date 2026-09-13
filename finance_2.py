# -*- coding: utf-8 -*-
"""
💰 家庭理财 · AI 智能问答系统
基于您提供的理财学术论文（PDF）构建本地知识库
支持 BM25 / Elasticsearch 双检索，Qwen 大模型生成回答
"""

import os
import re
import math
from typing import List, Dict, Any
from collections import defaultdict

import gradio as gr
from pypdf import PdfReader
from qwen_agent.llm import get_chat_model

# 尝试导入 Elasticsearch
try:
    from elasticsearch import Elasticsearch
    from elasticsearch.exceptions import ConnectionError as ESConnectionError

    ES_AVAILABLE = True
except ImportError:
    ES_AVAILABLE = False
    print("⚠️ Elasticsearch 未安装，请运行: pip install elasticsearch")

# ======================== 配置区域 ========================
LLM_CONFIG = {
    'model': 'qwen-turbo',# ← 使用Qwen Turbo模型
    'model_type': 'qwen_dashscope',
    'api_key': os.environ.get('DASHSCOPE_API_KEY', ''),
    'generate_cfg': {'top_p': 0.8, 'max_tokens': 2048}
}

# Elasticsearch 配置
ES_CONFIG = {
    'hosts': ['http://localhost:9200'],  # 使用 HTTP（Docker 启动时禁用了 SSL）
    # 如果启用了安全认证，取消注释下面两行
    # 'username': 'elastic',
    # 'password': '123456',
}

# 检索参数
CHUNK_SIZE = 600
TOP_K = 8
DOCS_FOLDER = 'finance_docs'

# ======================== 检索模式选择 ========================
# True: 使用 Elasticsearch（适合 >1万份文档）
# False: 使用 BM25 内存检索（适合 <1000份文档）
USE_ES = False  # 👈 改为 True 启用 ES


# ======================== BM25 检索器 ========================
class FinanceBM25Retriever:
    """基于内存的 BM25 检索，适合几百份文档"""

    def __init__(self, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.documents = []
        self.metadata = []
        self.corpus = []
        self.idf = {}
        self.avgdl = 0
        self.k1, self.b = 1.5, 0.75
        self.is_ready = False

    def load_from_folder(self, folder):
        self.documents, self.metadata, self.corpus = [], [], []
        if not os.path.exists(folder) or not os.listdir(folder):
            print("📝 未检测到文档，正在生成理财示例文档...")
            os.makedirs(folder, exist_ok=True)
            self._generate_demo_docs(folder)

        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            content = ""
            if fname.lower().endswith('.pdf'):
                try:
                    reader = PdfReader(fpath)
                    for page in reader.pages:
                        content += page.extract_text() or ""
                except Exception as e:
                    print(f"⚠️ PDF解析失败 {fname}: {e}")
                    continue
            elif fname.lower().endswith('.txt'):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"⚠️ TXT读取失败 {fname}: {e}")
                    continue
            else:
                continue
            if not content.strip():
                continue
            for i, chunk in enumerate(self._chunk_text(content)):
                self.documents.append(chunk)
                self.metadata.append({'file_name': fname, 'chunk_id': i})

        self._build_bm25()
        self.is_ready = True
        print(f"✅ BM25 知识库加载完成！共 {len(self.documents)} 个知识片段")

    def _generate_demo_docs(self, folder):
        demos = {
            "基金定投策略.txt": """
            基金定投的核心逻辑：
            1. 定期定额投入，平摊成本，降低择时风险。
            2. 止盈不止损：设定目标收益率（如15%），达到后分批赎回。
            3. 定投频率：月投或周投均可，长期看差异不大。
            4. 适合人群：工薪阶层，无暇频繁操作。
            5. 参考标的：宽基指数基金（沪深300、中证500）。
            """,
            "个人所得税专项附加扣除.txt": """
            2025年个人所得税专项附加扣除标准（摘要）：
            1. 子女教育：每个子女每月2000元。
            2. 继续教育：学历教育每月400元，职业资格抵扣3600元/年。
            3. 大病医疗：自付超过15000元部分，上限80000元/年。
            4. 住房贷款利息：首套房贷，每月1000元（最长240个月）。
            5. 住房租金：直辖市/省会每月1500元，其他城市800-1100元。
            6. 赡养老人：独生子女每月3000元。
            """,
            "家庭保险配置指南.txt": """
            家庭保险的"双十原则"和"四大险种"：
            - 双十原则：年保费占年收入10%，保额达到年收入10倍。
            四大险种：
            1. 重疾险（给付型）：保额30-50万。
            2. 医疗险（报销型）：百万医疗，免赔额1万。
            3. 意外险：保额100万。
            4. 定期寿险：家庭支柱必备，覆盖房贷+子女教育+5年生活开支。
            """,
            "房贷LPR与还款策略.txt": """
            房贷利率（LPR）：
            1. LPR每月20日报价，房贷利率=LPR+加点。
            2. 加点固定，LPR浮动，重定价周期通常1年。
            3. 提前还款：若投资收益率低于房贷利率（如4%），建议提前还。
            4. 等额本息 vs 等额本金：前者每月固定，前期利息高；后者总利息少，前期压力大。
            """,
            "股票投资基础术语.txt": """
            股票常见术语：
            - PE（市盈率）= 股价/每股收益，估值指标。
            - PB（市净率）= 股价/每股净资产。
            - ROE（净资产收益率）= 净利润/净资产，长期>15%为优质。
            - 股息率 = 每股分红/股价。
            - 换手率：反映活跃度。
            - 北向资金：外资买入A股的风向标。
            """
        }
        for fname, content in demos.items():
            with open(os.path.join(folder, fname), 'w', encoding='utf-8') as f:
                f.write(content)
        print("✅ 已生成5个理财示例文档")

    def _chunk_text(self, text):
        paras = re.split(r'\n\s*\n', text)   #将长文档切分成多个语义完整的小块
        chunks, cur = [], ""
        for p in paras:
            p = p.strip()
            if not p:
                continue
            if len(cur) + len(p) <= self.chunk_size:
                cur += p + "\n"
            else:
                if cur:
                    chunks.append(cur.strip())
                cur = p + "\n"
        if cur:
            chunks.append(cur.strip())
        return chunks

    def _tokenize(self, text):  #将文本拆分成最小语义单位（token），便于后续统计。
        tokens = []
        for word in text.split():
            if any('\u4e00' <= c <= '\u9fff' for c in word):
                tokens.extend(list(word))
            else:
                tokens.append(word.lower())
        return [t for t in tokens if t.strip() and not re.match(r'^[\W_]+$', t)]

    def _build_bm25(self):  #构建BM25索引,统计所有文档的词频信息
        if not self.documents:
            return
        self.corpus = [self._tokenize(d) for d in self.documents]
        doc_freq = defaultdict(int)
        for tokens in self.corpus:
            for t in set(tokens):
                doc_freq[t] += 1
        N = len(self.corpus)
        self.idf = {t: math.log((N - f + 0.5) / (f + 0.5) + 1) for t, f in doc_freq.items()}
        self.avgdl = sum(len(d) for d in self.corpus) / N if N else 0

    def search(self, query, top_k=TOP_K):  #BM25 检索，返回最相关的几个文档块
        if not self.is_ready:
            return []
        q_tokens = self._tokenize(query)
        scores = []
        for idx, doc_tokens in enumerate(self.corpus):
            score, dl = 0, len(doc_tokens)
            for t in q_tokens:
                if t not in self.idf:
                    continue
                tf = doc_tokens.count(t)
                if tf == 0:
                    continue
                idf = self.idf[t]
                #置信度score
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{'content': self.documents[idx], 'score': s, 'file_name': self.metadata[idx]['file_name']}
                for idx, s in scores[:top_k] if s > 0]


# ======================== Elasticsearch 检索器 ========================
class FinanceESRetriever:
    """基于 Elasticsearch 的检索器，适合大规模文档"""

    def __init__(self, es_config, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.es_config = es_config
        self.es = None
        self.index_name = 'finance_docs'
        self.is_ready = False

        if not ES_AVAILABLE:
            print("❌ Elasticsearch 客户端未安装")
            return

        try:
            # 连接 ES
            self.es = Elasticsearch(
                hosts=es_config['hosts'],
                request_timeout=30
            )
            # 测试连接
            if self.es.ping():
                print("✅ Elasticsearch 连接成功！")
                self.is_ready = True
                self._create_index()
            else:
                print("❌ Elasticsearch 连接失败，请检查服务是否启动")
                print(
                    "   启动命令: docker run -d --name elasticsearch -p 9200:9200 -p 9300:9300 -e 'discovery.type=single-node' -e 'xpack.security.enabled=false' -e 'ES_JAVA_OPTS=-Xms1g -Xmx1g' elasticsearch:8.15.0")
        except Exception as e:
            print(f"❌ Elasticsearch 连接异常: {e}")

    def _create_index(self):
        """创建索引映射进行分类"""
        if not self.es.indices.exists(index=self.index_name):
            mapping = {
                'mappings': {
                    'properties': {
                        'content': {
                            'type': 'text',  # 全文搜索
                            'analyzer': 'standard', # 索引时用的分词器
                            'search_analyzer': 'standard' # 搜索时用的分词器
                        },
                        'file_name': {'type': 'keyword'},
                        'chunk_id': {'type': 'integer'}
                    }
                }
            }
            try:
                self.es.indices.create(index=self.index_name, body=mapping)
                print(f"✅ 创建索引: {self.index_name}")
            except Exception as e:
                print(f"⚠️ 创建索引失败: {e}")

    def load_from_folder(self, folder):
        """将文档索引到 ES"""
        if not self.is_ready:
            print("❌ ES 未就绪，无法索引文档")
            return

        # 生成示例文档（如果文件夹为空）
        if not os.path.exists(folder) or not os.listdir(folder):
            print("📝 未检测到文档，正在生成理财示例文档...")
            os.makedirs(folder, exist_ok=True)
            bm25 = FinanceBM25Retriever()
            bm25._generate_demo_docs(folder)

        # 读取并索引文档
        doc_count = 0
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            content = ""
            if fname.lower().endswith('.pdf'):
                try:
                    reader = PdfReader(fpath)
                    for page in reader.pages:
                        content += page.extract_text() or ""
                except Exception as e:
                    print(f"⚠️ PDF解析失败 {fname}: {e}")
                    continue
            elif fname.lower().endswith('.txt'):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    print(f"⚠️ TXT读取失败 {fname}: {e}")
                    continue
            else:
                continue
            if not content.strip():
                continue

            # 分块并索引
            bm25 = FinanceBM25Retriever()
            chunks = bm25._chunk_text(content)
            for i, chunk in enumerate(chunks):
                doc = {
                    'content': chunk,
                    'file_name': fname,
                    'chunk_id': i
                }
                try:
                    self.es.index(index=self.index_name, body=doc)
                    doc_count += 1
                except Exception as e:
                    print(f"⚠️ 索引失败: {e}")

        # 刷新索引
        self.es.indices.refresh(index=self.index_name)
        print(f"✅ ES 索引完成！共索引 {doc_count} 个文档片段")

    def search(self, query, top_k=TOP_K):
        """使用 ES 检索"""  #向Elasticsearch发送搜索请求，获取与用户问题最相关的文档片段
        if not self.is_ready:
            return []

        try:
            response = self.es.search(
                index=self.index_name,
                body={
                    'query': {
                        'match': {
                            'content': {
                                'query': query,
                                'operator': 'or'
                            }
                        }
                    },
                    'size': top_k,
                    'highlight': {
                        'fields': {
                            'content': {
                                'fragment_size': 200,
                                'number_of_fragments': 1
                            }
                        }
                    }
                }
            )

            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'content': hit['_source']['content'],
                    'score': hit['_score'],
                    'file_name': hit['_source'].get('file_name', '未知来源')
                })
            return results
        except Exception as e:
            print(f"⚠️ ES 检索失败: {e}")
            return []


# ======================== Qwen 智能问答 ========================
class FinanceQwenAgent:
    def __init__(self, use_es=USE_ES):
        self.use_es = use_es
        self.llm = get_chat_model(LLM_CONFIG)
        self.retriever = None
        self.is_ready = False

    def load_knowledge(self, folder=DOCS_FOLDER):
        """加载知识库（根据配置选择 BM25 或 ES）"""
        if self.use_es and ES_AVAILABLE:
            print("🚀 使用 Elasticsearch 检索模式（适合大规模文档）")
            self.retriever = FinanceESRetriever(ES_CONFIG)
            self.retriever.load_from_folder(folder)
        else:
            if self.use_es and not ES_AVAILABLE:
                print("⚠️ Elasticsearch 不可用，降级使用 BM25 模式")
            print("📚 使用 BM25 内存检索模式（适合小规模文档）")
            self.retriever = FinanceBM25Retriever()
            self.retriever.load_from_folder(folder)

        self.is_ready = self.retriever.is_ready if hasattr(self.retriever, 'is_ready') else True
        if self.is_ready:
            print("✅ 知识库加载完成！")

    def ask(self, question):
        if not self.is_ready or not self.retriever:
            return "❌ 知识库未加载，请检查文档目录。", []

        chunks = self.retriever.search(question, top_k=TOP_K)
        if not chunks:
            return "🤔 您的理财资料中未找到相关信息，请尝试换个问法。", []

        context = "\n\n".join([f"[📄 来源：{c['file_name']}]\n{c['content']}" for c in chunks])
        prompt = f"""你是一位专业的家庭理财顾问，请基于用户提供的理财文献回答以下问题。
如果文献中没有明确提及，请如实告知"您的资料中未涉及此内容"，不要编造信息。

【参考资料】
{context}

【用户问题】
{question}

【专业回答】
"""
        try:#调用Qwen生成回答
            messages = [{'role': 'user', 'content': prompt}]
            response_gen = self.llm.chat(messages)
            final_resp = None
            for resp in response_gen:
                final_resp = resp
            if final_resp and isinstance(final_resp, list) and len(final_resp) > 0:
                answer = final_resp[0].get('content', '')
            else:
                answer = "生成回答失败：未获得有效响应"
        except Exception as e:
            answer = f"模型调用出错: {e}"
            print(f"详细错误: {e}")

        sources = [f"📎 {c['file_name']} (相关度: {c['score']:.3f})" for c in chunks]
        return answer, sources


# ======================== Gradio 界面 ========================
def create_finance_ui():
    agent = FinanceQwenAgent(use_es=USE_ES)
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER, exist_ok=True)
    agent.load_knowledge(DOCS_FOLDER) # 加载知识库

    def ask_question(query, history):
        if not query.strip():
            return history, ""
        answer, sources = agent.ask(query) # 调用RAG系统
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        return history, "\n".join(sources) if sources else "无来源"

    def clear_all():
        return [], ""

    with gr.Blocks(theme=gr.themes.Soft(), title="💰 家庭理财 · AI 问答") as demo:
        gr.Markdown(
            """
            # 💰 家庭理财 · AI 智能问答
            ### 基于您的理财文献库（PDF/TXT）提供专业解答
            已加载相关学术论文及政策文件，您可以询问：
            - 家庭资产配置、生命周期理财
            - 保险规划、养老金、子女教育金
            - 房贷、LPR、税务筹划
            - 银行理财产品选择、投资风险
            """
        )
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="对话历史", height=500, type="messages")
                with gr.Row():
                    msg = gr.Textbox(placeholder="例如：中等收入家庭如何规划子女教育金？", label="输入问题", scale=4)
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空对话", variant="secondary", scale=1)
            with gr.Column(scale=1):
                gr.Markdown("### 📎 答案来源")
                sources_display = gr.Textbox(label="引用文档及片段", lines=10, interactive=False)
                gr.Markdown(
                    """
                    **💡 使用提示**：
                    1. 将您的 PDF/TXT 文档放入 `finance_docs/` 文件夹
                    2. 首次运行会自动生成示例文档
                    3. 提问越具体，回答越精准
                    4. 所有数据均在本地，保护隐私
                    """
                )

        send_btn.click(ask_question, [msg, chatbot], [chatbot, sources_display]).then(
            lambda: "", None, [msg]
        )
        msg.submit(ask_question, [msg, chatbot], [chatbot, sources_display]).then(
            lambda: "", None, [msg]
        )
        clear_btn.click(clear_all, None, [chatbot, sources_display])

    return demo


if __name__ == "__main__":
    demo = create_finance_ui()
    demo.launch(
        server_name="127.0.0.1",  #只允许本机访问
        server_port=7860,
        share=False, #不生成公共链接
        inbrowser=True  #自动打开浏览器
    )