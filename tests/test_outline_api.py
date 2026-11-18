# -*- coding: utf-8 -*-
"""提纲素材核查 API 测试：校验、异步核查、迟到结果保护、重置与重启隔离。"""
from __future__ import annotations

import time

import pytest


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """每个用例独立数据库；worker 单例，提纲任务与推荐任务共用后台线程。"""
    from app import database, main

    main.get_worker().stop()
    db_path = tmp_path / "outline.db"
    monkeypatch.setenv("RECOMMENDER_DB", str(db_path))
    database._DB_PATH = str(db_path)
    database._reset_reveal_batch()
    database.init_db()
    main.get_worker().start()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c, database, main.get_worker()
    main.get_worker().stop()


def _wait_outline(c, check_id, timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        job = c.get(f"/api/outline-checks/{check_id}").json()
        if job["status"] in ("done", "failed", "stale"):
            return job
        time.sleep(0.05)
    raise AssertionError("提纲核查未在预期时间内结束")


def _good_points():
    return [
        {"text": "大语言模型微调时冻结网络参数并注入适配层", "threshold": 0.15},
        {"text": "神经网络反向传播梯度与过拟合泛化", "threshold": 0.15},
        {"text": "混合精度训练降低显存占用与算力调度", "threshold": 0.15},
    ]


def test_outline_check_happy_path(client):
    c, _, _ = client
    r = c.post("/api/outline-checks", json={"points": _good_points()})
    assert r.status_code == 200
    job_brief = r.json()["outline_job"]
    check_id = job_brief["check_id"]
    assert job_brief["status"] == "pending"

    job = _wait_outline(c, check_id)
    assert job["status"] == "done"
    result = job["result"]
    assert result["check_id"] == check_id
    assert result["point_count"] == 3
    assert result["matched_count"] >= 1
    # 结果顺序与输入一致，字段完整
    assert [p["index"] for p in result["results"]] == [0, 1, 2]
    assigned_articles = []
    for p in result["results"]:
        assert p["text"] and 0.05 <= p["threshold"] <= 0.95
        if p["assigned"]:
            assert p["article_id"] and p["sentence"] and p["score"] >= p["threshold"] - 1e-12
            assigned_articles.append(p["article_id"])
        else:
            assert p["article_id"] is None
    # 每篇文章至多分给一个点
    assert len(assigned_articles) == len(set(assigned_articles))


def test_outline_validation(client):
    c, _, _ = client
    base = _good_points()

    # 点数不足 3
    r = c.post("/api/outline-checks", json={"points": base[:2]})
    assert r.status_code == 422

    # 点数超过 6
    too_many = base + [
        {"text": f"额外提纲点 {i}", "threshold": 0.2} for i in range(4)
    ]
    r = c.post("/api/outline-checks", json={"points": too_many})
    assert r.status_code == 422

    # 阈值越界
    bad = [dict(p) for p in base]
    bad[0]["threshold"] = 0.04
    r = c.post("/api/outline-checks", json={"points": bad})
    assert r.status_code == 422
    bad[0]["threshold"] = 0.96
    r = c.post("/api/outline-checks", json={"points": bad})
    assert r.status_code == 422

    # 空短句
    empty = [dict(p) for p in base]
    empty[1]["text"] = "   "
    r = c.post("/api/outline-checks", json={"points": empty})
    assert r.status_code == 422

    # 非法延迟
    r = c.post(
        "/api/outline-checks",
        json={"points": base, "compute_delay_ms": 30001},
    )
    assert r.status_code == 422


def test_high_threshold_marks_points_unassigned(client):
    c, _, _ = client
    points = [
        {"text": "深度学习训练流程与数据工程", "threshold": 0.9},
        {"text": "推荐系统召回排序与嵌入向量", "threshold": 0.9},
        {"text": "高原耐力训练与运动损伤康复", "threshold": 0.9},
    ]
    r = c.post("/api/outline-checks", json={"points": points}).json()
    job = _wait_outline(c, r["outline_job"]["check_id"])
    result = job["result"]
    # 0.9 的高门槛下语料句子很难达标：至少存在待补材料点，且这些点字段安全
    assert result["matched_count"] < 3
    for p in result["results"]:
        if not p["assigned"]:
            assert p["article_id"] is None and p["sentence"] is None
            assert p["best_unsatisfied"] is not None
            assert p["best_unsatisfied"]["score"] < p["threshold"]


def test_stale_outline_result_never_overwrites_new_check(client):
    c, database, worker = client
    # 第一次核查带 2 秒延迟
    r1 = c.post(
        "/api/outline-checks",
        json={"points": _good_points(), "compute_delay_ms": 2000},
    ).json()
    id_a = r1["outline_job"]["check_id"]

    # 延迟窗口内立刻提交第二次核查（无延迟）
    points_b = _good_points()
    points_b[0] = {"text": "家庭烘焙低油低盐与炖煮技法", "threshold": 0.15}
    r2 = c.post("/api/outline-checks", json={"points": points_b}).json()
    id_b = r2["outline_job"]["check_id"]
    assert id_b == id_a + 1

    job_b = _wait_outline(c, id_b)
    assert job_b["status"] == "done"

    # 等第一次核查结算：必须是 stale，且没有 result 落库
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job_a = database.get_outline_job(id_a)
        if job_a["status"] in ("stale", "done", "failed"):
            break
        time.sleep(0.05)
    job_a = database.get_outline_job(id_a)
    assert job_a["status"] == "stale"
    assert job_a["result"] is None

    # /api/status 只暴露最新核查，前端不会拿到被作废的旧结果
    status = c.get("/api/status").json()
    assert status["outline_job"]["check_id"] == id_b
    assert status["outline_job"]["status"] == "done"
    # 新结果反映的是第二次提纲的文本
    texts = [p["text"] for p in status["outline_job"]["result"]["results"]]
    assert texts == [p["text"] for p in points_b]


def test_outline_check_unknown_id_404(client):
    c, _, _ = client
    assert c.get("/api/outline-checks/99999").status_code == 404


def test_reset_clears_outline_jobs(client):
    c, database, _ = client
    r = c.post("/api/outline-checks", json={"points": _good_points()}).json()
    _wait_outline(c, r["outline_job"]["check_id"])
    assert database.get_latest_outline_job() is not None

    c.post("/api/reset")
    assert database.get_latest_outline_job() is None
    assert c.get("/api/status").json()["outline_job"] is None


def test_outline_failure_is_recorded_without_crashing_worker(client, monkeypatch):
    c, database, worker = client
    import app.worker as wmod

    def boom(*args, **kwargs):
        raise RuntimeError("simulated outline failure")

    monkeypatch.setattr(wmod, "match_outline", boom)
    r = c.post("/api/outline-checks", json={"points": _good_points()}).json()
    check_id = r["outline_job"]["check_id"]
    job = _wait_outline(c, check_id)
    assert job["status"] == "failed"
    assert "simulated outline failure" in job["error"]

    # worker 仍存活：撤销补丁后新的核查可以正常完成
    monkeypatch.undo()
    r2 = c.post("/api/outline-checks", json={"points": _good_points()}).json()
    job2 = _wait_outline(c, r2["outline_job"]["check_id"])
    assert job2["status"] == "done"
    assert worker._thread is not None and worker._thread.is_alive()


def test_recommender_still_works_alongside_outline(client):
    """新增提纲核查不得影响原有阅读推荐链路。"""
    c, _, _ = client
    # 推荐状态正常
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        st = c.get("/api/status").json()
        if st["job"] and st["job"]["status"] == "done":
            break
        time.sleep(0.05)
    st = c.get("/api/status").json()
    assert st["plan"] is not None and st["plan"]["feasible"]

    # 提交提纲核查后再揭示，推荐队列与揭示进度照常工作
    c.post("/api/outline-checks", json={"points": _good_points()})
    out = c.post("/api/reveal").json()
    assert out["reveal_outcome"]["advanced"] is True
    assert len(out["state"]["revealed"]) == 2
