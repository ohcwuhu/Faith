"""AI 对话留痕的可复核性测试：阶段/风险落库、授权存证、总结承接与统计出口。

这些用例对应技术报告里的三条主张：

* "每一轮都有阶段判定与风险分级" → 消息表字段 + 会话级最高风险；
* "语音与图像在授权后才处理" → ``consent_records`` 存证；
* "通话结束可生成日记并与原会话关联" → 总结确认接口的幂等行为。
"""

import asyncio
import time as time_mod
from datetime import timedelta

from sqlalchemy import func, select

from app.db.session import AsyncSessionLocal, SessionLocal
from app.models.ai_conversation import AiConversation, AiMessage, ConsentRecord
from app.models.growth import EmotionJournal
from app.models.user import User
from app.services import ai_conversation_service as conversations
from app.services.maintenance_service import sweep_stale_coaching_sessions


def unique_phone() -> str:
    return "128" + str(int(time_mod.time() * 1000) % 100000000).zfill(8)


def register(client) -> dict:
    phone = unique_phone()
    resp = client.post(
        "/api/v1/auth/register",
        json={
            "phone": phone,
            "password": "Test123456",
            "nickname": "留痕测试",
            "privacyAgreed": True,
            "serviceAgreed": True,
        },
    )
    assert resp.status_code == 201
    return {
        "phone": phone,
        "headers": {"Authorization": f"Bearer {resp.json()['data']['accessToken']}"},
    }


def cleanup(phone: str) -> None:
    db = SessionLocal()
    try:
        user = db.scalar(select(User).where(User.phone == phone))
        if user is not None:
            user_id = user.id
            db.query(AiMessage).filter(
                AiMessage.conversation_id.in_(
                    select(AiConversation.id).where(AiConversation.user_id == user_id)
                )
            ).delete(synchronize_session=False)
            db.query(ConsentRecord).filter(ConsentRecord.user_id == user_id).delete(
                synchronize_session=False
            )
            db.query(AiConversation).filter(AiConversation.user_id == user_id).delete(
                synchronize_session=False
            )
            db.query(EmotionJournal).filter(EmotionJournal.user_id == user_id).delete(
                synchronize_session=False
            )
            db.commit()
            db.delete(user)
            db.commit()
    finally:
        db.close()


# ============================================================
#  纯函数：情绪映射 / 风险合并 / 草稿模板 / 授权快照
# ============================================================
def test_suggest_mood_maps_unified_labels():
    """七类融合情绪映射到六类日记情绪。"""
    assert conversations.suggest_mood("sad") == "DOWN"
    assert conversations.suggest_mood("fearful") == "ANXIOUS"
    assert conversations.suggest_mood("angry") == "IRRITATED"
    assert conversations.suggest_mood("disgusted") == "IRRITATED"
    assert conversations.suggest_mood("happy") == "HAPPY"
    assert conversations.suggest_mood("surprised") == "HAPPY"
    assert conversations.suggest_mood("neutral") == "CALM"
    assert conversations.suggest_mood(None) == "OTHER"
    assert conversations.suggest_mood("SAD") == "DOWN"
    assert conversations.suggest_mood("confused") == "OTHER"


def test_max_risk_level_picks_highest_and_ignores_unknown():
    """会话级最高风险取等级序最大者，非法值不参与比较。"""
    assert conversations.max_risk_level(["NONE", "MEDIUM", "LOW"]) == "MEDIUM"
    assert conversations.max_risk_level(["LOW", "HIGH"]) == "HIGH"
    assert conversations.max_risk_level([]) is None
    assert conversations.max_risk_level([None, "", "  "]) is None
    assert conversations.max_risk_level([None, "unknown", "LOW"]) == "LOW"
    assert conversations.max_risk_level(["low", "medium"]) == "MEDIUM"


def test_summary_draft_prefers_closing_text_and_falls_back_to_template():
    """有收束原文时用原文，没有时用确定性模板，保证草稿始终可用。"""
    closing = conversations.build_summary_draft(
        theme="要不要考研",
        final_stage="closing",
        closing_text="我们今天主要聊到了考研这件事，你准备先了解三个方向的信息。",
    )
    assert closing.startswith("我们今天主要聊到了考研")

    templated = conversations.build_summary_draft(theme="要不要考研", final_stage="goal_setting")
    assert "要不要考研" in templated
    assert "更清楚自己想要什么" in templated
    assert conversations.build_summary_draft(theme="   ", final_stage="opening") == "先聊了聊最近的状态。"

    long_closing = "总结" * 600
    assert len(conversations.build_summary_draft(
        theme="x", final_stage="closing", closing_text=long_closing,
    )) == conversations.MAX_SUMMARY_LENGTH


def test_normalize_consent_defaults_to_all_false():
    """未提供授权时不臆造同意，并记录协议版本与授权时间。"""
    snapshot = conversations.normalize_consent(None)
    assert snapshot["mic"] is False
    assert snapshot["camera"] is False
    assert snapshot["multimodal"] is False
    assert snapshot["policyVersion"] == conversations.DEFAULT_POLICY_VERSION
    assert snapshot["grantedAt"]

    nested = conversations.normalize_consent({
        "scopes": {"mic": True, "camera": True, "multimodal": 1, "extra": "ignored"},
        "policyVersion": "2026-10",
        "basis": "CALL_START_DIALOG",
    })
    assert nested["mic"] is True and nested["camera"] is True and nested["multimodal"] is True
    assert nested["policyVersion"] == "2026-10"
    assert nested["basis"] == "CALL_START_DIALOG"
    assert "extra" not in nested


def test_build_draft_from_messages_uses_theme_and_closing_branch():
    """主题取首条用户表达；收束轮的 AI 原文优先，并给出情绪候选。"""
    messages = [
        {"role": "USER", "content": "我在纠结要不要考研", "fusion_emotion": None,
         "emotion": None, "should_summarize": None},
        {"role": "ASSISTANT", "content": "我们可以先看看你的顾虑", "should_summarize": None},
    ]
    draft = conversations.build_draft_from_messages(
        title="自我教练对话", final_stage="goal_setting", turn_count=1,
        messages=messages, already_confirmed=False,
    )
    assert "我在纠结要不要考研" in draft["draft"]
    assert draft["mood_type"] == "OTHER"
    assert draft["already_confirmed"] is False

    closing_messages = messages + [
        {"role": "USER", "content": "我打算先查三个方向", "fusion_emotion": "fearful",
         "emotion": None, "should_summarize": None},
        {"role": "ASSISTANT", "content": "今天主要聊到了考研的方向选择。",
         "should_summarize": True},
    ]
    closing = conversations.build_draft_from_messages(
        title="自我教练对话", final_stage="closing", turn_count=2,
        messages=closing_messages, already_confirmed=False,
    )
    assert closing["draft"] == "今天主要聊到了考研的方向选择。"
    assert closing["mood_type"] == "ANXIOUS"


# ============================================================
#  落库：授权存证、逐轮阶段/风险、异常会话收尾
# ============================================================
async def _seed_session(user_id: int, phone_sid: str = "sid-evidence"):
    session_id = await conversations.start_session_safely(
        user_id=user_id,
        client_session_id=phone_sid,
        consent={"mic": True, "camera": False, "multimodal": True, "policyVersion": "2026-09"},
    )
    assert session_id, "会话应当创建成功"
    await conversations.record_message_safely(
        session_id,
        role="USER",
        content="我最近压力很大，不知道自己想要什么",
        turn_index=1,
        emotion={"fusion_emotion_cn": "焦虑", "fusion_confidence": 0.72},
        snapshot=conversations.TurnSnapshot(
            turn_index=1,
            user_text="我最近压力很大，不知道自己想要什么",
            stage="exploration",
            goal_clear=False,
            action_ready=False,
            should_summarize=False,
            summary_reason="问题仍在探索阶段",
            risk_level="LOW",
            risk_score=20,
            fusion_emotion="fearful",
            fusion_confidence=0.72,
        ),
        timings={"asr_seconds": 0.8},
    )
    await conversations.record_message_safely(
        session_id,
        role="ASSISTANT",
        content="听起来最近确实很辛苦，我们先看看压力来自哪里。",
        turn_index=1,
        snapshot=conversations.TurnSnapshot(
            turn_index=1,
            user_text="我最近压力很大，不知道自己想要什么",
            stage="exploration",
            risk_level="LOW",
            risk_score=20,
            fusion_emotion="fearful",
            fusion_confidence=0.72,
        ),
        timings={"asr_seconds": 0.8, "llm_total_seconds": 2.4, "e2e_seconds": 4.1},
    )
    return session_id


def test_session_evidence_persists_consent_stage_and_risk(client):
    acc = register(client)
    try:
        db = SessionLocal()
        try:
            user_id = db.scalar(select(User.id).where(User.phone == acc["phone"]))
        finally:
            db.close()

        session_id = asyncio.run(_seed_session(user_id))

        db = SessionLocal()
        try:
            conversation = db.get(AiConversation, session_id)
            assert conversation.turn_count == 1
            assert conversation.final_stage == "exploration"
            assert conversation.max_risk_level == "LOW"
            assert conversation.consent["multimodal"] is True
            assert conversation.consent["camera"] is False

            consent = db.scalar(
                select(ConsentRecord).where(ConsentRecord.conversation_id == session_id)
            )
            assert consent is not None
            assert consent.scopes == {
                "mic": True,
                "camera": False,
                "multimodal": True,
                "basis": "VIDEO_CALL_START",
            }
            assert consent.policy_version == "2026-09"
            assert consent.source == "VIDEO_CALL"

            messages = list(
                db.scalars(
                    select(AiMessage)
                    .where(AiMessage.conversation_id == session_id)
                    .order_by(AiMessage.id.asc())
                )
            )
            assert [m.role for m in messages] == ["USER", "ASSISTANT"]
            assert all(m.turn_index == 1 for m in messages)
            assert messages[0].stage == "exploration"
            assert messages[0].risk_level == "LOW"
            assert messages[0].fusion_emotion == "fearful"
            assert messages[1].timings["llm_total_seconds"] == 2.4
            assert conversation.message_count == 2
        finally:
            db.close()
    finally:
        cleanup(acc["phone"])


def test_stale_active_conversation_is_marked_abandoned(client):
    """异常断开的会话由维护任务收尾，保证会话状态可统计。"""
    acc = register(client)
    try:
        db = SessionLocal()
        try:
            user_id = db.scalar(select(User.id).where(User.phone == acc["phone"]))
        finally:
            db.close()

        session_id = asyncio.run(_seed_session(user_id, phone_sid="sid-stale"))

        db = SessionLocal()
        try:
            conversation = db.get(AiConversation, session_id)
            conversation.created_at = db.scalar(select(func.now())) - timedelta(hours=8)
            db.commit()
        finally:
            db.close()

        async def _sweep() -> int:
            async with AsyncSessionLocal() as db:
                return await sweep_stale_coaching_sessions(db)

        assert asyncio.run(_sweep()) >= 1

        db = SessionLocal()
        try:
            conversation = db.get(AiConversation, session_id)
            assert conversation.status == "ABANDONED"
            assert conversation.ended_at is not None
        finally:
            db.close()
    finally:
        cleanup(acc["phone"])


def test_consent_change_records_grant_and_revocation(client):
    """通话中开关摄像头：每次变更都留痕，撤回时给被撤回的授权补 revoked_at。

    对应"用户可随时关闭摄像头"这条承诺——审核时能看到撤回发生在哪一刻，
    以及撤回之后仍然有效的范围是什么。
    """
    acc = register(client)
    sid = "sid-consent-change"
    try:
        db = SessionLocal()
        try:
            user_id = db.scalar(select(User.id).where(User.phone == acc["phone"]))
        finally:
            db.close()

        # 通话开始时未授权摄像头
        session_id = asyncio.run(_seed_session(user_id, phone_sid=sid))

        # 通话中用户主动开启摄像头
        granted_id = asyncio.run(conversations.record_consent_change_safely(
            user_id=user_id,
            client_session_id=sid,
            conversation_id=session_id,
            consent={
                "mic": True,
                "camera": True,
                "multimodal": True,
                "basis": "USER_ENABLE_CAMERA",
            },
        ))
        assert granted_id

        # 随后撤回摄像头
        revoked_id = asyncio.run(conversations.record_consent_change_safely(
            user_id=user_id,
            client_session_id=sid,
            conversation_id=session_id,
            consent={
                "mic": True,
                "camera": False,
                "multimodal": True,
                "basis": "USER_DISABLE_CAMERA",
            },
            revoked_scopes=["camera"],
        ))
        assert revoked_id and revoked_id != granted_id

        db = SessionLocal()
        try:
            records = list(
                db.scalars(
                    select(ConsentRecord)
                    .where(ConsentRecord.client_session_id == sid)
                    .order_by(ConsentRecord.id.asc())
                )
            )
            assert len(records) == 3, "通话开始 1 条 + 开启 1 条 + 撤回 1 条"
            assert records[0].scopes["camera"] is False
            assert records[0].scopes["basis"] == "VIDEO_CALL_START"

            assert records[1].scopes["camera"] is True
            assert records[1].scopes["basis"] == "USER_ENABLE_CAMERA"
            assert records[1].revoked_at is not None, "被撤回的授权应补上撤回时间"

            assert records[2].scopes["camera"] is False
            assert records[2].scopes["basis"] == "USER_DISABLE_CAMERA"
            assert records[2].revoked_at is None

            conversation = db.get(AiConversation, session_id)
            assert conversation.consent["camera"] is False, "会话快照跟随当前有效范围"
            assert conversation.consent["multimodal"] is True
        finally:
            db.close()
    finally:
        cleanup(acc["phone"])


# ============================================================
#  接口：会话详情字段、总结确认幂等、统计出口
# ============================================================
def test_conversation_detail_exposes_stage_and_risk(client):
    acc = register(client)
    try:
        db = SessionLocal()
        try:
            user_id = db.scalar(select(User.id).where(User.phone == acc["phone"]))
        finally:
            db.close()

        session_id = asyncio.run(_seed_session(user_id, phone_sid="sid-detail"))
        asyncio.run(conversations.end_session_safely(session_id))

        resp = client.get(f"/api/v1/ai-conversations/{session_id}", headers=acc["headers"])
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["conversation"]["turnCount"] == 1
        assert data["conversation"]["finalStage"] == "exploration"
        assert data["conversation"]["finalStageLabel"] == "问题探索"
        assert data["conversation"]["maxRiskLevel"] == "LOW"
        assert data["conversation"]["status"] == "ENDED"
        first = data["messages"][0]
        assert first["stage"] == "exploration"
        assert first["riskLevel"] == "LOW"
        assert first["emotion"]["fusion_emotion_cn"] == "焦虑"
    finally:
        cleanup(acc["phone"])


def test_summary_confirm_is_idempotent_and_links_journal(client):
    """确认总结后写入会话总结并生成日记；重复确认不产生第二篇日记。"""
    acc = register(client)
    try:
        db = SessionLocal()
        try:
            user_id = db.scalar(select(User.id).where(User.phone == acc["phone"]))
        finally:
            db.close()

        session_id = asyncio.run(_seed_session(user_id, phone_sid="sid-summary"))
        asyncio.run(conversations.end_session_safely(session_id))

        body = {"content": "今天聊了聊压力，我想先把下一步定下来。", "moodType": "ANXIOUS"}
        first = client.post(
            f"/api/v1/ai-conversations/{session_id}/summary/confirm",
            headers=acc["headers"],
            json=body,
        )
        assert first.status_code == 201
        payload = first.json()["data"]
        assert payload["moodType"] == "ANXIOUS"
        assert payload["journalId"]

        second = client.post(
            f"/api/v1/ai-conversations/{session_id}/summary/confirm",
            headers=acc["headers"],
            json=body,
        )
        assert second.status_code == 201
        assert second.json()["data"]["journalId"] == payload["journalId"]

        db = SessionLocal()
        try:
            journals = list(
                db.scalars(
                    select(EmotionJournal).where(
                        EmotionJournal.source_conversation_id == session_id
                    )
                )
            )
            assert len(journals) == 1
            assert journals[0].source == "SELF_COACHING"
            conversation = db.get(AiConversation, session_id)
            assert conversation.summary == body["content"]
            assert conversation.summary_confirmed_at is not None
            assert conversation.journal_id == journals[0].id
        finally:
            db.close()
    finally:
        cleanup(acc["phone"])


def test_admin_multimodal_stats_endpoint(client, admin_headers):
    """统计接口给出分析总量、状态分布、耗时口径与会话数。"""
    resp = client.get("/api/v1/admin/stats/multimodal?days=30", headers=admin_headers)
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "totals" in data and "status" in data["totals"]
    assert "latency" in data and "metrics" in data["latency"]
    assert "coaching" in data
    assert {"sessions", "messages", "distinctUsers"} <= set(data["coaching"])
    assert "daily" in data
    assert data["range"]["days"] == 30


def test_conversation_endpoints_require_authentication(client):
    """未登录不能读取或确认任何对话记录。"""
    assert client.get("/api/v1/ai-conversations").status_code == 401
    assert client.get("/api/v1/ai-conversations/1").status_code == 401
    assert client.delete("/api/v1/ai-conversations/1").status_code == 401
    assert client.post("/api/v1/ai-conversations/1/summary").status_code == 401


def test_missing_or_foreign_conversation_returns_404(client):
    """不存在的会话与他人会话统一返回 404，避免探测资源是否存在。"""
    owner = register(client)
    other = register(client)
    try:
        db = SessionLocal()
        try:
            owner_id = db.scalar(select(User.id).where(User.phone == owner["phone"]))
        finally:
            db.close()

        session_id = asyncio.run(_seed_session(owner_id, phone_sid="sid-ownership"))

        assert client.get(
            "/api/v1/ai-conversations/99999999", headers=owner["headers"]
        ).status_code == 404
        assert client.get(
            f"/api/v1/ai-conversations/{session_id}", headers=other["headers"]
        ).status_code == 404
        assert client.delete(
            f"/api/v1/ai-conversations/{session_id}", headers=other["headers"]
        ).status_code == 404
    finally:
        cleanup(owner["phone"])
        cleanup(other["phone"])


def test_summary_confirm_validates_input(client):
    """空内容或非法情绪类型由参数校验拦截（统一 400 + VALIDATION_ERROR），不写入日记。"""
    acc = register(client)
    try:
        db = SessionLocal()
        try:
            user_id = db.scalar(select(User.id).where(User.phone == acc["phone"]))
        finally:
            db.close()

        session_id = asyncio.run(_seed_session(user_id, phone_sid="sid-validate"))
        empty = client.post(
            f"/api/v1/ai-conversations/{session_id}/summary/confirm",
            headers=acc["headers"],
            json={"content": "", "moodType": "ANXIOUS"},
        )
        assert empty.status_code == 400
        assert empty.json()["code"] == "VALIDATION_ERROR"

        illegal_mood = client.post(
            f"/api/v1/ai-conversations/{session_id}/summary/confirm",
            headers=acc["headers"],
            json={"content": "总结", "moodType": "HAPPY_SAD"},
        )
        assert illegal_mood.status_code == 400
        assert illegal_mood.json()["code"] == "VALIDATION_ERROR"
    finally:
        cleanup(acc["phone"])


def test_delete_conversation_keeps_journal_and_removes_messages(client):
    """删除会话后消息级联删除，情绪日记保留但解除关联（成长记录不因删对话而丢失）。"""
    acc = register(client)
    try:
        db = SessionLocal()
        try:
            user_id = db.scalar(select(User.id).where(User.phone == acc["phone"]))
        finally:
            db.close()

        session_id = asyncio.run(_seed_session(user_id, phone_sid="sid-delete"))
        created = client.post(
            f"/api/v1/ai-conversations/{session_id}/summary/confirm",
            headers=acc["headers"],
            json={"content": "今天聊了聊压力。", "moodType": "ANXIOUS"},
        )
        journal_id = created.json()["data"]["journalId"]

        assert client.delete(
            f"/api/v1/ai-conversations/{session_id}", headers=acc["headers"]
        ).status_code == 204

        db = SessionLocal()
        try:
            assert db.get(AiConversation, session_id) is None
            assert list(
                db.scalars(select(AiMessage).where(AiMessage.conversation_id == session_id))
            ) == []
            journal = db.get(EmotionJournal, journal_id)
            assert journal is not None
            assert journal.source_conversation_id is None
        finally:
            db.close()
    finally:
        cleanup(acc["phone"])


def test_admin_stats_forbidden_for_normal_user(client, auth_headers):
    """管理端统计不对普通用户开放。"""
    assert client.get("/api/v1/admin/stats/multimodal", headers=auth_headers).status_code == 403
