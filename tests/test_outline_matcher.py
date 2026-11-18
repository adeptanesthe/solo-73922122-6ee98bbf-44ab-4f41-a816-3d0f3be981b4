# -*- coding: utf-8 -*-
"""提纲核查单元测试：句子切分、阈值门槛、一文章一点、字典序目标与暴力对拍。"""
from __future__ import annotations

import itertools

import pytest

from app.articles_data import ARTICLES
from app.outline_matcher import BIG_PENALTY, match_outline, split_sentences
from app.tfidf_model import get_model


def _similarity_records(points: list[dict]) -> list[list[dict]]:
    """用真实 TF-IDF 模型生成每个点对全部句子的评分记录。"""
    model = get_model()
    q = model.transform_texts([p["text"] for p in points])
    rows = model.cosine_against_sentences(q)
    return [
        [dict(rec, score=score) for rec, score in zip(model.sentence_records, row)]
        for row in rows
    ]


# ---------------- 句子切分 ----------------


def test_split_sentences_keeps_endings():
    text = "第一句。第二句！第三句？没有标点的结尾"
    assert split_sentences(text) == ["第一句。", "第二句！", "第三句？", "没有标点的结尾"]


def test_split_sentences_handles_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   ") == []
    # 纯标点/空白不产生句子
    assert split_sentences("。。！") == []


def test_model_sentences_cover_all_articles():
    model = get_model()
    ids = {rec["article_id"] for rec in model.sentence_records}
    assert ids == {a["id"] for a in ARTICLES}
    # 每条句子记录都带来源文章 id、句序与非空文本
    for rec in model.sentence_records:
        assert rec["text"].strip()
        assert rec["sentence_index"] >= 0
    # 句子矩阵与记录行数一致，且与文章矩阵共享特征空间（只 transform 不 refit）
    assert model.sentence_matrix.shape[1] == model.article_matrix.shape[1]


# ---------------- 真实模型端到端分配 ----------------


def test_real_model_assignment_and_unique_article():
    points = [
        {"text": "大语言模型微调冻结参数并用小规模标注数据更新", "threshold": 0.15},
        {"text": "反向传播计算梯度并用过拟合手段提升泛化", "threshold": 0.15},
        {"text": "混合精度训练降低显存占用提升算力利用率", "threshold": 0.15},
    ]
    result = match_outline(points, _similarity_records(points))
    assert result.point_count == 3
    assert result.matched_count == 3
    assigned = [r for r in result.results if r.assigned]
    # 每篇文章至多分给一个点
    article_ids = [r.article_id for r in assigned]
    assert len(article_ids) == len(set(article_ids))
    # 来源句子确实达到该点下限
    for r in assigned:
        assert r.score >= r.threshold - 1e-12
        assert r.sentence and r.article_title
    # total_score 与各点得分一致
    assert result.total_score == pytest.approx(sum(r.score for r in assigned))


def test_unmatched_point_marked_need_material():
    # 与语料几乎无关的提纲点 + 高门槛，必然待补材料
    points = [
        {"text": "大语言模型微调与参数高效适配层", "threshold": 0.15},
        {"text": "神经网络梯度下降与过拟合缓解", "threshold": 0.15},
        {"text": "火星地下溶洞外星水母演奏量子爵士奏鸣曲", "threshold": 0.6},
    ]
    result = match_outline(points, _similarity_records(points))
    todo = [r for r in result.results if not r.assigned]
    assert len(todo) == 1
    r = todo[0]
    assert r.article_id is None and r.sentence is None
    # 仍给出全库最接近的一句，且其分数确实低于门槛
    assert r.best_unsatisfied is not None
    assert r.best_unsatisfied["score"] < r.threshold


def test_deterministic_repeated_matching():
    points = [
        {"text": "深度学习模型训练与数据清洗", "threshold": 0.1},
        {"text": "推荐系统嵌入向量与近邻召回", "threshold": 0.1},
        {"text": "计算机视觉姿态估计边缘部署", "threshold": 0.1},
    ]
    recs = _similarity_records(points)
    r1 = match_outline(points, recs)
    r2 = match_outline(points, recs)
    key1 = [(r.assigned, r.article_id, r.sentence, r.score) for r in r1.results]
    key2 = [(r.assigned, r.article_id, r.sentence, r.score) for r in r2.results]
    assert key1 == key2


# ---------------- 构造评分的精确性测试（含暴力对拍） ----------------


def _synthetic_records(points: list[dict], edge_scores: dict[int, dict[int, float]]):
    """按 {点序号: {文章id: 分数}} 构造单句/文章的评分记录。"""
    records = []
    for i, _ in enumerate(points):
        row = []
        for aid, score in edge_scores.get(i, {}).items():
            row.append({
                "article_id": aid,
                "sentence_index": 0,
                "text": f"article-{aid}-sentence",
                "score": float(score),
            })
        records.append(row)
    return records


def test_threshold_gating_rejects_below_floor():
    points = [
        {"text": "p1", "threshold": 0.5},
        {"text": "p2", "threshold": 0.5},
        {"text": "p3", "threshold": 0.5},
    ]
    # 只有文章 1、2 达标，文章 3 差一点 -> 第三个点必须待补
    edges = {
        0: {1: 0.9, 2: 0.2, 3: 0.2},
        1: {1: 0.2, 2: 0.8, 3: 0.2},
        2: {1: 0.2, 2: 0.2, 3: 0.49},
    }
    result = match_outline(points, _synthetic_records(points, edges))
    assert result.matched_count == 2
    assigned_articles = {r.article_id for r in result.results if r.assigned}
    assert assigned_articles == {1, 2}
    assert not result.results[2].assigned


def test_cardinality_beats_total_similarity():
    # 字典序第一目标：达标点数量优先，即使牺牲总相似度
    points = [
        {"text": "p1", "threshold": 0.5},
        {"text": "p2", "threshold": 0.5},
        {"text": "p3", "threshold": 0.5},
    ]
    edges = {
        # 点 0 独占高分文章 1；若点 0 改取文章 2，点 2 就能用文章 1 达标
        0: {1: 0.99, 2: 0.55},
        1: {3: 0.9},
        2: {1: 0.6, 2: 0.2},
    }
    result = match_outline(points, _synthetic_records(points, edges))
    # 必须 3 个点全部达标（点0 让文章1 给点2，自己取文章2）
    assert result.matched_count == 3
    by_index = {r.index: r for r in result.results}
    assert by_index[0].article_id == 2
    assert by_index[2].article_id == 1
    assert by_index[1].article_id == 3


def test_bruteforce_small_pool():
    """在 3 点 × 4 文章的小池子上与全暴力枚举对拍。"""
    import random

    rng = random.Random(123)
    pool = [a["id"] for a in ARTICLES[:4]]
    points = [
        {"text": f"p{i}", "threshold": round(0.1 + i * 0.1, 2)}
        for i in range(3)
    ]
    edges = {
        i: {aid: round(rng.random(), 4) for aid in pool}
        for i in range(3)
    }
    result = match_outline(points, _synthetic_records(points, edges))

    # 暴力：枚举“点 -> 不同文章/空”的所有注入，按 (达标数, 总分) 取字典序最优
    best_key = (-1, -1.0)
    best_pick: dict | None = None
    choices = [None] + pool
    for pick in itertools.product(choices, repeat=3):
        used = [a for a in pick if a is not None]
        if len(used) != len(set(used)):
            continue
        count = 0
        total = 0.0
        for i, aid in enumerate(pick):
            if aid is not None and edges[i][aid] >= points[i]["threshold"] - 1e-12:
                count += 1
                total += edges[i][aid]
        key = (count, round(total, 10))
        if key > best_key:
            best_key = key
            best_pick = pick

    assert result.matched_count == best_key[0]
    assert result.total_score == pytest.approx(best_key[1], abs=1e-9)
    for r in result.results:
        assert (r.article_id if r.assigned else None) == best_pick[r.index]


def test_big_penalty_dominates_similarity_range():
    # 防御性：不达标边的额外代价必须压过任何相似度差异（相似度 ∈ [0,1]）
    assert BIG_PENALTY > 2
