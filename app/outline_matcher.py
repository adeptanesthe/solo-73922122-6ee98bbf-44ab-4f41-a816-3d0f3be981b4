# -*- coding: utf-8 -*-
"""提纲素材核查：句子切分、相似度计算与“一文章至多一点”的全局分配。

问题定义
========
专题编辑给出 3~6 个提纲点（短句）及各自的相似度下限，系统从 30 篇文章
正文按 ``。！？`` 切出的句子中为每个提纲点寻找相近论述：

1. TF-IDF 仍只在 30 篇文章正文上拟合（见 :mod:`app.tfidf_model`），
   提纲短句与句子都只做 ``transform``；
2. 相似度为字符 2~3 元 TF-IDF 向量的余弦相似度，截断到 [0, 1]；
3. **每篇文章至多分给一个提纲点**（文章内取与该点最相似的一句作为候选边）；
4. 优化目标按字典序：
   先让尽可能多的提纲点取得达标的来源文章，
   再使所选（点, 文章）句子的总相似度最高；
5. 未分配到达标来源的提纲点标记为“待补材料”。

实现
====
规模很小（点 ≤ 6，文章 30），用矩形二分图最大权匹配求精确全局最优：
``scipy.optimize.linear_sum_assignment``（scipy 随 scikit-learn 一并安装）。
为表达“先数量后总分”的字典序目标，边代价编码为

    达标边（相似度 >= 下限）:  1 - 相似度
    不达标边:                  1 + BIG

其中 BIG（如 1e6）远大于相似度区间，使算法优先选择全部达标边；
达标边内部再按相似度最大化（代价最小化）。行数（点数 <=6）小于列数（30 篇文章），
矩形匹配天然允许部分点不与任何文章配对，这些点即判定为待补材料。
同相似度的平局由文章 id 与句子序号确定性打破，重复运行结果稳定。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from scipy.optimize import linear_sum_assignment

from .articles_data import ARTICLES

# 句末标点：中文句号、问号、叹号（含半角 . ! ? 兜底，正文目前只用中文标点）
_SENTENCE_ENDINGS = "。！？!?…"
_SENT_SPLIT_RE = re.compile(rf"(.*?[{_SENTENCE_ENDINGS}])", re.DOTALL)

# 不达标边的额外代价：远大于任何相似度差异，保证“达标点数量”优先
BIG_PENALTY = 1e6


def split_sentences(text: str) -> list[str]:
    """按 。！？ 切句，保留句末标点；无标点的尾部文本作为最后一句。

    纯空白或仅含句末标点的片段被丢弃，其余不做额外清洗
    （与字符级 TF-IDF 保持一致）。
    """
    if not text:
        return []
    sentences = [m.group(1) for m in _SENT_SPLIT_RE.finditer(text)]
    tail = _SENT_SPLIT_RE.sub("", text).strip()
    if tail:
        sentences.append(tail)
    return [s for s in sentences if s.strip().strip(_SENTENCE_ENDINGS).strip()]


@dataclass
class OutlinePointResult:
    """单个提纲点的核查结果。"""

    index: int                      # 输入顺序（0 起）
    text: str                       # 提纲短句
    threshold: float                # 相似度下限
    assigned: bool                  # 是否分配到达标来源
    article_id: int | None = None
    article_title: str | None = None
    topic: str | None = None
    author: str | None = None
    sentence: str | None = None     # 来源原句
    score: float | None = None      # 所选句子的相似度
    # 未达标时，给出全库最接近的一句，便于编辑判断是否调低下限
    best_unsatisfied: dict | None = None


@dataclass
class OutlineCheckResult:
    matched_count: int
    point_count: int
    total_score: float
    results: list[OutlinePointResult] = field(default_factory=list)


def _article_meta(article_id: int) -> dict:
    a = next(x for x in ARTICLES if x["id"] == article_id)
    return {"id": a["id"], "title": a["title"], "topic": a["topic"], "author": a["author"]}


def match_outline(
    points: list[dict],
    sentence_similarities: list[list[dict]],
) -> OutlineCheckResult:
    """执行提纲点与文章句子的全局最优分配。

    参数
    ----
    points:
        每项含 ``text`` 与 ``threshold``（已校验在 0.05~0.95）。
    sentence_similarities:
        与 points 等长、按点排列；每个元素是该点对全部句子的评分记录列表，
        每条记录含 ``article_id``、``sentence_index``、``text``、``score``，
        句子顺序与 :class:`~app.tfidf_model.ArticleTfidfModel.sentence_records` 一致。

    约束：每篇文章至多分配给一个点；只有相似度不低于该点下限的边才算达标。
    """
    n_points = len(points)
    article_ids = [a["id"] for a in ARTICLES]
    n_articles = len(article_ids)
    article_pos = {aid: j for j, aid in enumerate(article_ids)}

    # ---- 1. 聚合为（点 × 文章）边：文章内取最相似的一句 ----
    # edge_score[i][j]: 点 i 与文章 j 最佳句子的相似度
    edge_score: list[list[float]] = [[0.0] * n_articles for _ in range(n_points)]
    # edge_sent[i][j]: 对应最佳句子的记录（article_id/sentence_index/text/score）
    edge_sent: list[list[dict | None]] = [[None] * n_articles for _ in range(n_points)]
    for i, point in enumerate(points):
        for rec in sentence_similarities[i]:
            j = article_pos[rec["article_id"]]
            score = float(rec["score"])
            prev = edge_sent[i][j]
            # 文章内多句时取最高分；平分保留先出现的句子（句序确定）
            if prev is None or score > float(prev["score"]):
                edge_score[i][j] = score
                edge_sent[i][j] = rec

    # ---- 2. 构造代价矩阵：达标数量优先，其次总相似度最大 ----
    # 线性分配要求行数 <= 列数；点（<=6）少于文章（30），天然满足。
    cost = [[0.0] * n_articles for _ in range(n_points)]
    for i, point in enumerate(points):
        threshold = float(point["threshold"])
        for j, aid in enumerate(article_ids):
            score = edge_score[i][j]
            if score + 1e-12 >= threshold:
                cost[i][j] = 1.0 - score          # 相似度越高代价越小
            else:
                cost[i][j] = 1.0 + BIG_PENALTY    # 不达标边代价极高

    row_ind, col_ind = linear_sum_assignment(cost)

    # ---- 3. 组装结果；未拿到达标边的点标记待补材料 ----
    results: list[OutlinePointResult] = []
    matched_count = 0
    total_score = 0.0
    assignment = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    for i, point in enumerate(points):
        threshold = float(point["threshold"])
        # 全库最接近的一句（不要求达标），用于未达标提示；平分时取记录顺序
        # 最靠前的一条（模型按文章 id、句序生成记录，结果确定）。
        best_rec = None
        best_key: tuple[float, int] | None = None
        for pos, rec in enumerate(sentence_similarities[i]):
            key = (float(rec["score"]), -pos)
            if best_key is None or key > best_key:
                best_key = key
                best_rec = rec
        best_unsatisfied = None
        if best_rec is not None and float(best_rec["score"]) + 1e-12 < threshold:
            meta = _article_meta(best_rec["article_id"])
            best_unsatisfied = {
                "article_id": meta["id"],
                "article_title": meta["title"],
                "topic": meta["topic"],
                "author": meta["author"],
                "sentence": best_rec["text"],
                "score": float(best_rec["score"]),
            }

        j = assignment.get(i)
        rec = edge_sent[i][j] if j is not None else None
        score = edge_score[i][j] if j is not None else 0.0
        is_qualified = rec is not None and score + 1e-12 >= threshold

        if is_qualified:
            meta = _article_meta(rec["article_id"])
            results.append(
                OutlinePointResult(
                    index=i,
                    text=point["text"],
                    threshold=threshold,
                    assigned=True,
                    article_id=meta["id"],
                    article_title=meta["title"],
                    topic=meta["topic"],
                    author=meta["author"],
                    sentence=rec["text"],
                    score=score,
                    best_unsatisfied=best_unsatisfied,
                )
            )
            matched_count += 1
            total_score += score
        else:
            results.append(
                OutlinePointResult(
                    index=i,
                    text=point["text"],
                    threshold=threshold,
                    assigned=False,
                    best_unsatisfied=best_unsatisfied,
                )
            )

    return OutlineCheckResult(
        matched_count=matched_count,
        point_count=n_points,
        total_score=float(total_score),
        results=results,
    )
