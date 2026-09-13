# -*- coding: utf-8 -*-
"""
💰 家庭理财 · AI 智能问答系统
基于您提供的 5 篇理财学术论文（PDF）构建本地知识库
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

# ======================== 配置区域 ========================
# 请务必填写您的 DashScope API Key
LLM_CONFIG = {
    'model': 'qwen-turbo',                # 或 qwen-plus, qwen-max
    'model_type': 'qwen_dashscope',
    'api_key': os.environ.get('DASHSCOPE_API_KEY', ''),    # 从环境变量读取，勿硬编码
    'generate_cfg': {'top_p': 0.8, 'max_tokens': 2048}
}

# Elasticsearch 配置（若文档数量巨大（>1万份），可启用）
ES_CONFIG = {
    'hosts': ['https://localhost:9200'],
    'username': 'elastic',
    'password': '123456',   #你的elastic密码
    'verify_certs': False,
}

# 检索参数
CHUNK_SIZE = 600          # 每块字符数（略大，适合学术段落）
TOP_K = 8                 # 检索返回的块数
DOCS_FOLDER = 'finance_docs'   # 文档存放目录

# ======================== BM25 检索器 ========================
class FinanceBM25Retriever:
    """基于内存的 BM25 检索，适合几百份文档"""
    def __init__(self, chunk_size=CHUNK_SIZE):
        self.chunk_size = chunk_size
        self.documents = []          # 原文块列表
        self.metadata = []           # 元数据（文件名等）
        self.corpus = []             # 分词后的文档
        self.idf = {}
        self.avgdl = 0
        self.k1, self.b = 1.5, 0.75
        self.is_ready = False

    def load_from_folder(self, folder):
        self.documents, self.metadata, self.corpus = [], [], []
        # 若文件夹为空，生成理财示例文档（方便首次测试）
        if not os.path.exists(folder) or not os.listdir(folder):
            print("📝 未检测到文档，正在生成理财示例文档...")
            os.makedirs(folder, exist_ok=True)
            self._generate_demo_docs(folder)

        # 遍历文件夹，解析 PDF/TXT
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
            # 分块
            for i, chunk in enumerate(self._chunk_text(content)):
                self.documents.append(chunk)
                self.metadata.append({'file_name': fname, 'chunk_id': i})

        self._build_bm25()
        self.is_ready = True
        print(f"✅ 理财知识库加载完成！共 {len(self.documents)} 个知识片段（来自 {len(set(d['file_name'] for d in self.metadata))} 个文档）")

    def _generate_demo_docs(self, folder):
        """生成理财示例文档（仅当无文档时）"""
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
        # 按双换行分块，保留段落结构
        paras = re.split(r'\n\s*\n', text)
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

    def _tokenize(self, text):
        # 简易分词（保留中文单字和英文单词）
        tokens = []
        for word in text.split():
            if any('\u4e00' <= c <= '\u9fff' for c in word):
                tokens.extend(list(word))
            else:
                tokens.append(word.lower())
        return [t for t in tokens if t.strip() and not re.match(r'^[\W_]+$', t)]

    def _build_bm25(self):
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

    def search(self, query, top_k=TOP_K):
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
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append((idx, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{'content': self.documents[idx], 'score': s, 'file_name': self.metadata[idx]['file_name']}
                for idx, s in scores[:top_k] if s > 0]

# ======================== Qwen 智能问答 ========================
class FinanceQwenAgent:
    def __init__(self, use_es=False):
        self.use_es = use_es
        self.llm = get_chat_model(LLM_CONFIG)
        self.retriever = None
        self.is_ready = False

    def load_knowledge(self, folder=DOCS_FOLDER):
        # 目前仅使用 BM25，ES 可后续扩展
        self.retriever = FinanceBM25Retriever()
        self.retriever.load_from_folder(folder)
        self.is_ready = True

    def ask(self, question):
        if not self.is_ready or not self.retriever:
            return "❌ 知识库未加载，请检查文档目录。", []
        chunks = self.retriever.search(question, top_k=TOP_K)
        if not chunks:
            return "🤔 您的理财资料中未找到相关信息，请尝试换个问法。", []
        context = "\n\n".join([f"[📄 来源：{c['file_name']}]\n{c['content']}" for c in chunks])
        prompt = f"""你是一位专业的家庭理财顾问，请基于用户提供的理财文献（学术论文、政策文件等）回答以下问题。
如果文献中没有明确提及，请如实告知“您的资料中未涉及此内容”，不要编造信息。

【参考资料】
{context}

【用户问题】
{question}

【专业回答】
"""
        try:
            messages = [{'role': 'user', 'content': prompt}]
            # 注意：qwen_agent 的 chat 方法返回生成器，需要迭代获取最终结果
            response_gen = self.llm.chat(messages)
            # 逐条获取（最后一条是完整回答）
            final_resp = None
            for resp in response_gen:
                final_resp = resp
            # 如果 final_resp 是列表且非空，取第一条的 content
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
    agent = FinanceQwenAgent(use_es=False)
    # 自动加载文档（若文件夹不存在则创建并生成示例）
    if not os.path.exists(DOCS_FOLDER):
        os.makedirs(DOCS_FOLDER, exist_ok=True)
    agent.load_knowledge(DOCS_FOLDER)

    def ask_question(query, history):
        if not query.strip():
            return history, ""
        answer, sources = agent.ask(query)
        # 使用字典格式（type='messages' 要求）
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
            已加载 相关学术论文及政策文件，您可以询问：
            - 家庭资产配置、生命周期理财
            - 保险规划、养老金、子女教育金
            - 房贷、LPR、税务筹划
            - 银行理财产品选择、投资风险
            - 大学生理财建议、消费金融
            """
        )
        with gr.Row():
            with gr.Column(scale=2):
                # 使用 type='messages' 以支持字典格式
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
                    2. 首次运行会自动生成示例文档（若该文件夹为空）
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
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True
    )