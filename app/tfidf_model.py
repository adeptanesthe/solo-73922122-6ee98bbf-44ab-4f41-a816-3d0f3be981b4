# -*- coding: utf-8 -*-
"""字符 2~3 元 TF-IDF 向量化。

关键约束：分析器只在文章正文（语料）上 fit，兴趣短文只做 transform，
保证特征空间与 IDF 权重完全由文章拟合得到。
"""
from __future__ import annotations

import threading

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .articles_data import ARTICLES
from .outline_matcher import split_sentences


class ArticleTfidfModel:
    """对文章语料拟合的字符 2~3 元 TF-IDF 模型（单例、线程安全初始化）。"""

    def __init__(self) -> None:
        # 纯字符级分析器，ngram 范围严格为 2~3；中文无空格，按字符滑窗即可，
        # 不使用 char_wb（它会在词边界补空格并产生单字特征）。
        # 不用停用词，保留全部字符二元、三元组。
        self.vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )
        bodies = [a["body"] for a in ARTICLES]
        # 仅用文章拟合（fit），兴趣/提纲文本之后只能 transform。
        self.article_matrix = self.vectorizer.fit_transform(bodies)
        self.article_ids = [a["id"] for a in ARTICLES]
        self.feature_count = len(self.vectorizer.vocabulary_)

        # 把 30 篇正文按 。！？ 切成句子，并在同一（文章拟合出的）特征空间内
        # 只做 transform，绝不重新 fit。每条记录关联其来源文章 id。
        self.sentence_records: list[dict] = []
        for a in ARTICLES:
            for idx, sentence in enumerate(split_sentences(a["body"])):
                self.sentence_records.append(
                    {"article_id": a["id"], "sentence_index": idx, "text": sentence}
                )
        self.sentence_matrix = self.vectorizer.transform(
            [r["text"] for r in self.sentence_records]
        )

    def transform_interest(self, text: str):
        """把兴趣短文变换到文章拟合出的 TF-IDF 空间（不重新拟合）。"""
        return self.vectorizer.transform([text or ""])

    def transform_texts(self, texts: list[str]):
        """把若干短句变换到文章拟合出的 TF-IDF 空间（不重新拟合）。"""
        return self.vectorizer.transform([t or "" for t in texts])

    def cosine_against_sentences(self, query_matrix) -> list[list[float]]:
        """查询矩阵（多行短句）与全部正文句子的余弦相似度。

        返回 rows × 句子数 的二维列表，元素截断到 [0, 1]。
        """
        sims = cosine_similarity(query_matrix, self.sentence_matrix)
        return [
            [float(max(0.0, min(1.0, v))) for v in row]
            for row in sims
        ]

    def relevance_vector(self, interest_text: str) -> dict[int, float]:
        """返回 {文章id: 余弦相关度}，相关度截断到 [0, 1]。"""
        q = self.transform_interest(interest_text)
        sims = cosine_similarity(q, self.article_matrix)[0]
        # 余弦理论范围 [-1,1]；TF-IDF 非负，实际落在 [0,1]，这里显式截断。
        return {aid: float(max(0.0, min(1.0, s))) for aid, s in zip(self.article_ids, sims)}


_model: ArticleTfidfModel | None = None
_model_lock = threading.Lock()


def get_model() -> ArticleTfidfModel:
    """进程内单例，避免重复拟合。"""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = ArticleTfidfModel()
    return _model
