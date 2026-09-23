"""运行时防护测试：后台任务登记 + 上传文件标识格式。

这两块都属于"平时不报错、出事就很难查"的边界：

1. ``asyncio.create_task`` 不持引用时，任务可能被 GC 掉，留痕静默丢失；
2. ``vc_audio_end`` 用客户端传来的 ``file_id`` 拼路径，必须挡住 ``../``。
"""

import asyncio
import uuid

from app.core import tasks
from app.services.ai_lab.socket_events import UPLOAD_FILE_ID_PATTERN


# ============================================================
#  后台任务登记
# ============================================================
def test_spawn_task_keeps_reference_until_done():
    """任务执行期间必须有强引用，结束后自动释放。"""

    async def main() -> int:
        started = asyncio.Event()
        release = asyncio.Event()

        async def work() -> str:
            started.set()
            await release.wait()
            return "done"

        task = tasks.spawn_task(work(), name="test-work")
        assert task is not None
        await started.wait()
        assert tasks.pending_count() == 1, "任务未被登记（存在被 GC 回收的风险）"
        release.set()
        await task
        # done_callback 在下一个事件循环空转周期执行
        await asyncio.sleep(0)
        return tasks.pending_count()

    assert asyncio.run(main()) == 0


def test_spawn_task_survives_task_exception():
    """后台任务抛异常只记日志，不冒泡、不残留引用。"""

    async def main() -> int:
        async def boom() -> None:
            raise RuntimeError("模拟留痕失败")

        task = tasks.spawn_task(boom(), name="test-boom")
        assert task is not None
        await asyncio.sleep(0)
        assert task.done()
        await asyncio.sleep(0)
        return tasks.pending_count()

    assert asyncio.run(main()) == 0


def test_spawn_task_without_running_loop_returns_none():
    """没有事件循环时返回 None，而不是抛出 RuntimeError。"""

    async def work() -> None:  # pragma: no cover - 不会被 await
        pass

    coro = work()
    assert tasks.spawn_task(coro, name="test-no-loop") is None
    # 协程已被关闭，不会产生 "coroutine was never awaited" 警告
    assert coro.cr_frame is None


# ============================================================
#  上传文件标识
# ============================================================
def test_upload_file_id_accepts_generated_uuid_hex():
    assert UPLOAD_FILE_ID_PATTERN.match(uuid.uuid4().hex)


def test_upload_file_id_rejects_traversal_and_malformed():
    """`../` 越权路径、大小写/长度异常都要被挡在拼路径之前。"""
    for bad in (
        "../../../etc/passwd",
        "..%2f..%2fsecret",
        "abc",
        "",
        uuid.uuid4().hex.upper(),          # 大写不属于生成格式
        uuid.uuid4().hex + "0",            # 长度超标
        uuid.uuid4().hex[:-1] + "/",       # 混入分隔符
    ):
        assert UPLOAD_FILE_ID_PATTERN.match(bad) is None, bad
