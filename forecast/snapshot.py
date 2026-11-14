"""Cooperative snapshot locking and recoverable three-file publication.

The OS releases locks even after process termination. A pending journal is
rolled back before the next reader/writer can observe any of its files.
"""
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
import json
import os
import shutil
import threading

FILES = ("shandong_pmos_daily.csv", "weather_data_daily.csv", "weather_data_daily_predict.csv")
CURRENT = ".load-data-current.json"
JOURNAL = ".load-data-transaction.json"
_mutex = threading.RLock()
_local = threading.local()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def cleanup_temporary(directory, temporary):
    """Remove only our own direct child staging directory, never an output archive."""
    directory, temporary = Path(directory).resolve(), Path(temporary)
    if (temporary.is_symlink() or temporary.resolve().parent != directory
            or not temporary.name.startswith(".load-data-tmp-")):
        raise ValueError("临时目录不在允许的同步范围内")
    if temporary.exists():
        shutil.rmtree(temporary)


def _recover(directory):
    journal = directory / JOURNAL
    if not journal.exists():
        return
    state = json.loads(journal.read_text(encoding="utf-8"))
    if state["status"] != "committed":
        backups = Path(state["backups"])
        if set(state["existed"]) not in (set(FILES), set((*FILES, CURRENT))):
            raise ValueError("刷新恢复记录结构错误，禁止读取数据")
        # Backups are copied, not moved: recovery itself can be interrupted.
        for name, existed in state["existed"].items():
            target = directory / name
            if existed:
                temporary = directory / (".load-data-restore-" + name)
                shutil.copyfile(backups / name, temporary)
                os.replace(temporary, target)
            else:
                target.unlink(missing_ok=True)
        # Recovery may itself be interrupted during cleanup. Do not require the
        # backups again once every old file has already been restored.
        write_json(journal, {**state, "status": "committed"})
    if state.get("temporary_dir"):
        cleanup_temporary(directory, state["temporary_dir"])
    journal.unlink()


@contextmanager
def dataset_lock(directory):
    """Exclusive, reentrant in one thread; fail promptly if another run holds it."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if not _mutex.acquire(blocking=False):
        raise ValueError("数据正在刷新或读取，请稍后重试")
    held = getattr(_local, "held", set())
    stream = None
    locked = False
    nested = directory in held
    try:
        if not nested:
            stream = (directory / ".load-data-lock").open("a+b")
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise ValueError("数据正在刷新或读取，请稍后重试") from None
            locked = True
            _local.held = held | {directory}
            _recover(directory)
            for temporary in directory.glob(".load-data-tmp-*"):
                cleanup_temporary(directory, temporary)
        yield
    finally:
        if stream is not None:
            if locked:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_UN)
            stream.close()
            _local.held = held
        _mutex.release()


def guarded_run(function):
    """Keep market and weather inputs from the same generation during a run."""
    @wraps(function)
    def guarded(config, *args, **kwargs):
        if config["data"].get("market_input_mode", "daily_hourly") != "daily_only":
            return function(config, *args, **kwargs)
        from .data import resolve_path
        directory = resolve_path(config["data"]["daily"]).parent
        with dataset_lock(directory):
            return function(config, *args, **kwargs)
    return guarded


def publish(directory, staged):
    """Caller holds dataset_lock; publish three CSVs with temporary rollback data."""
    directory, staged = Path(directory), Path(staged)
    temporary = staged.parent.resolve()
    if temporary.parent != directory.resolve() or not temporary.name.startswith(".load-data-tmp-"):
        raise ValueError("同步暂存目录必须在数据目录内")
    backups = staged.parent / "backups"
    backups.mkdir()
    existed = {}
    for name in (*FILES, CURRENT):
        target = directory / name
        existed[name] = target.exists()
        if existed[name]:
            shutil.copyfile(target, backups / name)
    state = {"status": "pending", "backups": str(backups.resolve()), "existed": existed,
             "temporary_dir": str(temporary)}
    write_json(directory / JOURNAL, state)
    try:
        for name in FILES:
            os.replace(staged / name, directory / name)
        # Retire the old report pointer without creating another manifest.
        (directory / CURRENT).unlink(missing_ok=True)
        write_json(directory / JOURNAL, {**state, "status": "committed"})
    except BaseException:
        _recover(directory)
        raise
    # A committed journal keeps cleanup recoverable if the filesystem is busy.
    try:
        _recover(directory)
    except OSError:
        pass
