# -*- coding: utf-8 -*-
"""FastAPI 应用：推荐队列的条件管理、异步重查、分组揭示与演示页面。"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import database
from .articles_data import ARTICLES, TOPICS
from .tfidf_model import get_model
from .worker import get_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("recommender")

DEFAULT_INTEREST = "我想了解人工智能模型训练与数据驱动的分析方法"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class ConditionIn(BaseModel):
    interest: str | None = Field(default=None, description="兴趣短文")
    quotas: dict[str, int] | None = Field(default=None, description="各主题精确篇数")
    compute_delay_ms: int = Field(default=0, ge=0, le=30000)


class ExclusionIn(BaseModel):
    article_id: int = Field(ge=1)
    compute_delay_ms: int = Field(default=0, ge=0, le=30000)


class OutlinePointIn(BaseModel):
    text: str = Field(description="提纲点短句")
    threshold: float = Field(description="相似度下限，范围 0.05~0.95")


class OutlineCheckIn(BaseModel):
    points: list[OutlinePointIn] = Field(description="3~6 个提纲点")
    compute_delay_ms: int = Field(default=0, ge=0, le=30000)

def _validate_quotas(quotas: dict[str, int]) -> None:
    unknown = set(quotas) - set(TOPICS)
    if unknown:
        raise HTTPException(422, f"未知主题: {sorted(unknown)}")
    missing = set(TOPICS) - set(quotas)
    if missing:
        raise HTTPException(422, f"缺少主题配额: {sorted(missing)}")
    if any(v < 0 for v in quotas.values()):
        raise HTTPException(422, "配额不能为负数")
    if sum(quotas.values()) != 6:
        raise HTTPException(422, f"六个主题配额之和必须为 6，当前为 {sum(quotas.values())}")


def _validate_interest(interest: str) -> None:
    if not interest or not interest.strip():
        raise HTTPException(422, "兴趣短文不能为空")
    if len(interest) > 2000:
        raise HTTPException(422, "兴趣短文过长（上限 2000 字）")


def _validate_outline(body: OutlineCheckIn) -> list[dict]:
    """校验提纲核查请求并返回标准化的 points 快照。"""
    if not 3 <= len(body.points) <= 6:
        raise HTTPException(422, "提纲点数量必须在 3~6 个之间")
    points: list[dict] = []
    for i, p in enumerate(body.points):
        text = (p.text or "").strip()
        if not text:
            raise HTTPException(422, f"第 {i + 1} 个提纲点短句不能为空")
        if len(text) > 500:
            raise HTTPException(422, f"第 {i + 1} 个提纲点过长（上限 500 字）")
        if not 0.05 <= p.threshold <= 0.95:
            raise HTTPException(
                422, f"第 {i + 1} 个提纲点的相似度下限必须在 0.05~0.95 之间"
            )
        points.append({"text": text, "threshold": float(p.threshold)})
    return points


def _article_brief(a: dict) -> dict:
    return {"id": a["id"], "title": a["title"], "topic": a["topic"], "author": a["author"]}


def _status_payload() -> dict:
    state = database.get_state()
    job = database.get_latest_job()
    plan = None
    if job and job["status"] == "done" and job["generation"] == state["generation"]:
        plan = job["result"]
    return {
        "state": state,
        "job": job,
        "plan": plan,
        "topics": TOPICS,
        "articles": [_article_brief(a) for a in ARTICLES],
        "outline_job": database.get_latest_outline_job(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    # 预热 TF-IDF（仅文章拟合），失败不影响启动，worker 内会再试
    try:
        get_model()
    except Exception:  # pragma: no cover
        log.exception("TF-IDF 预热失败")
    get_worker().start()
    try:
        yield
    finally:
        get_worker().stop()


app = FastAPI(title="技术编辑选题推荐系统", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def status() -> dict:
    return _status_payload()


@app.get("/api/articles/{article_id}")
def article_detail(article_id: int) -> dict:
    for a in ARTICLES:
        if a["id"] == article_id:
            return a
    raise HTTPException(404, "文章不存在")


@app.put("/api/condition")
def update_condition(body: ConditionIn) -> dict:
    """修改兴趣和/或配额，开启新世代并触发异步重算。"""
    current = database.get_state()
    new_quotas = body.quotas if body.quotas is not None else current["quotas"]
    _validate_quotas(new_quotas)
    new_interest = body.interest if body.interest is not None else current["interest"]
    _validate_interest(new_interest)

    # 与已揭示前缀的兼容性前置检查（详细不可行原因由重算任务给出）
    from .optimizer import solve_queue  # 局部导入避免循环
    model = get_model()
    check = solve_queue(
        relevance=model.relevance_vector(new_interest),
        quotas=new_quotas,
        prefix=current["revealed"],
        excluded_ids=set(current["excluded_ids"]),
    )
    if not check.feasible:
        raise HTTPException(409, f"新条件与已揭示前缀不兼容: {check.reason}")

    database.begin_new_generation(
        interest=new_interest,
        quotas=new_quotas,
        compute_delay_ms=body.compute_delay_ms,
    )
    return _status_payload()


@app.post("/api/exclusions")
def add_exclusion(body: ExclusionIn) -> dict:
    if not any(a["id"] == body.article_id for a in ARTICLES):
        raise HTTPException(404, "文章不存在")
    try:
        database.add_exclusion(body.article_id, body.compute_delay_ms)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _status_payload()


@app.delete("/api/exclusions/{article_id}")
def remove_exclusion(article_id: int, compute_delay_ms: int = 0) -> dict:
    try:
        database.remove_exclusion(article_id, compute_delay_ms)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    return _status_payload()


@app.post("/api/reveal")
def reveal() -> dict:
    """揭示下一组两篇；同条件同进度下并发只推进一次。"""
    try:
        outcome = database.reveal_next_two()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))
    payload = _status_payload()
    payload["reveal_outcome"] = outcome
    return payload


@app.post("/api/outline-checks")
def create_outline_check(body: OutlineCheckIn) -> dict:
    """提交一次提纲素材核查，开启新的核查编号并由后台异步计算。

    每次提交都会使此前未完成的核查作废（stale），迟到结果无法覆盖新核查。
    """
    points = _validate_outline(body)
    snapshot = {"points": points, "compute_delay_ms": body.compute_delay_ms}
    database.create_outline_job(snapshot)
    return _status_payload()


@app.get("/api/outline-checks/{check_id}")
def get_outline_check(check_id: int) -> dict:
    job = database.get_outline_job(check_id)
    if job is None:
        raise HTTPException(404, "核查任务不存在")
    return job


@app.post("/api/reset")
def reset_demo() -> dict:
    """重置演示：清空揭示、排除与任务，世代回到 1。"""
    database.reset_all(DEFAULT_INTEREST, {t: 1 for t in TOPICS})
    return _status_payload()
