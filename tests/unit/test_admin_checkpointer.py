"""
관리자 AI 에이전트 Checkpointer 팩토리 단위 테스트 (Step 7c, 2026-04-27).

대상: `monglepick.agents.admin_assistant.graph` 의 checkpointer 헬퍼
- `_is_redis_checkpointer_enabled` — 환경변수 토글
- `_make_admin_checkpointer` — Redis vs Memory 분기 + import/init 실패 시 폴백
- `setup_admin_assistant_checkpointer` — asetup 호출 / no-op / 실패 swallow

Redis 인스턴스는 mock 처리. 실제 Redis 연결은 통합 테스트 영역.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

from monglepick.agents.admin_assistant import graph as admin_graph_module
from monglepick.agents.admin_assistant.graph import (
    _is_redis_checkpointer_enabled,
    _make_admin_checkpointer,
    setup_admin_assistant_checkpointer,
)


# ============================================================
# 1) _is_redis_checkpointer_enabled — 환경변수 파싱
# ============================================================

class TestIsRedisEnabled:
    def test_default_false(self, monkeypatch):
        monkeypatch.delenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", raising=False)
        assert _is_redis_checkpointer_enabled() is False

    def test_explicit_false(self, monkeypatch):
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "false")
        assert _is_redis_checkpointer_enabled() is False

    def test_true_lowercase(self, monkeypatch):
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "true")
        assert _is_redis_checkpointer_enabled() is True

    def test_true_uppercase(self, monkeypatch):
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "TRUE")
        assert _is_redis_checkpointer_enabled() is True

    def test_one(self, monkeypatch):
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "1")
        assert _is_redis_checkpointer_enabled() is True

    def test_yes(self, monkeypatch):
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "yes")
        assert _is_redis_checkpointer_enabled() is True

    def test_garbage_falls_back_false(self, monkeypatch):
        """무의미한 값은 false 로 안전 폴백."""
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "maybe")
        assert _is_redis_checkpointer_enabled() is False


# ============================================================
# 2) _make_admin_checkpointer — saver 팩토리
# ============================================================

class TestMakeCheckpointer:
    def test_disabled_returns_memory_saver(self, monkeypatch):
        """RAG 비활성 → MemorySaver."""
        monkeypatch.delenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", raising=False)
        saver, kind = _make_admin_checkpointer()
        assert isinstance(saver, MemorySaver)
        assert kind == "memory"

    def test_enabled_returns_redis_saver(self, monkeypatch):
        """RAG 활성 → AsyncRedisSaver (mock)."""
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "true")

        fake_saver = MagicMock()
        with patch(
            "langgraph.checkpoint.redis.aio.AsyncRedisSaver",
            return_value=fake_saver,
        ) as mock_cls:
            saver, kind = _make_admin_checkpointer()

        assert kind == "redis"
        assert saver is fake_saver
        # 키 prefix 가 admin_assistant 네임스페이스로 격리됐는지
        kwargs = mock_cls.call_args.kwargs
        assert kwargs["checkpoint_prefix"] == "admin_assistant:checkpoint"
        assert kwargs["checkpoint_blob_prefix"] == "admin_assistant:cp_blob"
        assert kwargs["checkpoint_write_prefix"] == "admin_assistant:cp_write"
        assert "redis_url" in kwargs

    def test_enabled_but_import_fails_falls_back_memory(self, monkeypatch):
        """RAG 활성이지만 redis 패키지 import 실패 → MemorySaver 폴백 (앱 기동 차단 X)."""
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "true")

        # importlib 차단을 위해 sys.modules 우회 — `from langgraph.checkpoint.redis.aio import ...`
        # 가 ImportError 던지도록 monkeypatch.
        import builtins
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "langgraph.checkpoint.redis.aio":
                raise ImportError("simulated missing pkg")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=fake_import):
            saver, kind = _make_admin_checkpointer()

        assert isinstance(saver, MemorySaver)
        assert kind == "memory"

    def test_enabled_but_init_fails_falls_back_memory(self, monkeypatch):
        """AsyncRedisSaver 생성 자체가 예외 던지면 MemorySaver 폴백."""
        monkeypatch.setenv("ADMIN_REDIS_CHECKPOINTER_ENABLED", "true")

        with patch(
            "langgraph.checkpoint.redis.aio.AsyncRedisSaver",
            side_effect=Exception("redis url malformed"),
        ):
            saver, kind = _make_admin_checkpointer()

        assert isinstance(saver, MemorySaver)
        assert kind == "memory"


# ============================================================
# 3) setup_admin_assistant_checkpointer — lifespan hook
# ============================================================

@pytest.mark.asyncio
class TestSetupCheckpointer:
    async def test_memory_saver_is_noop(self):
        """MemorySaver 는 asetup 미보유 → no-op."""
        with patch.object(admin_graph_module, "_admin_assistant_saver", MemorySaver()):
            # 예외 없이 통과해야 함
            await setup_admin_assistant_checkpointer()

    async def test_redis_saver_calls_asetup(self):
        """asetup 보유 saver 는 호출 1회."""
        fake_saver = MagicMock()
        fake_saver.asetup = AsyncMock()

        with patch.object(admin_graph_module, "_admin_assistant_saver", fake_saver):
            await setup_admin_assistant_checkpointer()

        fake_saver.asetup.assert_awaited_once()

    async def test_asetup_failure_swallowed(self):
        """asetup 실패해도 앱 기동 차단 X (예외 전파 없음)."""
        fake_saver = MagicMock()
        fake_saver.asetup = AsyncMock(side_effect=Exception("redis search index err"))

        with patch.object(admin_graph_module, "_admin_assistant_saver", fake_saver):
            # 예외 전파 안 됨
            await setup_admin_assistant_checkpointer()

        fake_saver.asetup.assert_awaited_once()

    async def test_no_saver_warns_and_returns(self):
        """saver 가 None 이면 경고 후 조용히 반환."""
        with patch.object(admin_graph_module, "_admin_assistant_saver", None):
            await setup_admin_assistant_checkpointer()
        # 별도 어설션 없음 — 예외 없이 통과해야 함
