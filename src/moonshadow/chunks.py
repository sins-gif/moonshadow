"""冷存分帧容器：让「按 source_id 精确回取」在压缩后依然成立。

v1.0 只写了 ``chunks/2026-09-30.zst``，没有定义如何在压缩流里定位第 N 条消息——
整块解压是 O(全天)，不可接受。这里定成显式容器格式：

    chunks/<day>.<codec>          压缩帧按顺序拼接
    chunks/<day>.<codec>.idx.json 帧索引：行号区间、偏移、压缩长度、原文 sha256

每帧固定 ``frame_lines`` 行（默认 1000），因此回取只需解压 1 帧：
    frame = line_no // frame_lines
    offset = frame.offset, length = frame.comp_len

默认 codec 为 stdlib ``zlib``（本机无 zstd）。若环境提供 ``zstandard`` 或
CPython 3.14+ 的 ``compression.zstd``，注册后同名机制直接生效。
"""

from __future__ import annotations

import hashlib
import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

DEFAULT_FRAME_LINES = 1000


class ChunkError(RuntimeError):
    pass


@dataclass(frozen=True)
class Frame:
    index: int
    first_line: int
    line_count: int
    offset: int
    comp_len: int
    raw_len: int
    sha256: str


@dataclass(frozen=True)
class ChunkIndex:
    day: str
    codec: str
    frame_lines: int
    container: str
    total_lines: int
    total_bytes: int
    frames: tuple[Frame, ...]

    @property
    def index_path(self) -> Path:
        return Path(self.container + ".idx.json")

    def frame_of(self, line_no: int) -> Frame:
        if line_no < 0:
            raise ChunkError(f"行号不能为负：{line_no}")
        target = line_no // self.frame_lines
        for frame in self.frames:
            if frame.index == target:
                return frame
        raise ChunkError(f"第 {line_no} 行超出容器范围（{self.total_lines} 行）")


_COMPRESSORS: dict[str, tuple[Callable[[bytes], bytes], Callable[[bytes], bytes]]] = {
    "zlib": (lambda data: zlib.compress(data, 9), zlib.decompress),
}


def register_codec(
    name: str,
    compress: Callable[[bytes], bytes],
    decompress: Callable[[bytes], bytes],
) -> None:
    """注册额外 codec（例如 zstd 可用时调 register_zstd()）。"""
    _COMPRESSORS[name] = (compress, decompress)


def register_zstd() -> str | None:
    """尝试注册 zstd；成功返回 codec 名，失败返回 None（静默降级，不阻断流程）。"""
    try:  # CPython 3.14+
        from compression import zstd  # type: ignore
    except Exception:
        try:
            import zstandard as zstd  # type: ignore
        except Exception:
            return None

    if hasattr(zstd, "compress"):
        register_codec("zstd", zstd.compress, zstd.decompress)
    else:  # zstandard 的流式 API
        def _compress(data: bytes) -> bytes:
            return zstd.ZstdCompressor().compress(data)

        def _decompress(data: bytes) -> bytes:
            return zstd.ZstdDecompressor().decompress(data)

        register_codec("zstd", _compress, _decompress)
    return "zstd"


def available_codecs() -> tuple[str, ...]:
    return tuple(_COMPRESSORS)


def container_path(directory: str | Path, day: str, codec: str) -> Path:
    return Path(directory) / f"{day}.{codec}"


def write_container(
    container: str | Path,
    lines: Iterable[bytes],
    *,
    day: str,
    codec: str = "zlib",
    frame_lines: int = DEFAULT_FRAME_LINES,
) -> ChunkIndex:
    """把原始行按帧压缩写入容器，并落一份索引。行内容不含换行符。"""
    if codec not in _COMPRESSORS:
        raise ChunkError(f"未注册的 codec：{codec}（可用：{sorted(_COMPRESSORS)}）")
    if frame_lines <= 0:
        raise ChunkError("frame_lines 必须为正整数")

    compress = _COMPRESSORS[codec][0]
    path = Path(container)
    path.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Frame] = []
    total_lines = 0
    total_bytes = 0
    offset = 0

    with path.open("wb") as handle:
        batch: list[bytes] = []

        def flush(index: int, first_line: int) -> None:
            nonlocal offset, total_lines, total_bytes
            if not batch:
                return
            payload = b"\n".join(batch) + b"\n"
            blob = compress(payload)
            handle.write(blob)
            frames.append(
                Frame(
                    index=index,
                    first_line=first_line,
                    line_count=len(batch),
                    offset=offset,
                    comp_len=len(blob),
                    raw_len=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            )
            offset += len(blob)
            total_lines += len(batch)
            total_bytes += len(payload)
            batch.clear()

        frame_index = 0
        first = 0
        for line in lines:
            if b"\n" in line:
                raise ChunkError("单行内容不得包含换行符")
            if not batch:
                first = total_lines
            batch.append(line)
            if len(batch) == frame_lines:
                flush(frame_index, first)
                frame_index += 1
        flush(frame_index, first)

    index = ChunkIndex(
        day=day,
        codec=codec,
        frame_lines=frame_lines,
        container=str(path),
        total_lines=total_lines,
        total_bytes=total_bytes,
        frames=tuple(frames),
    )
    with index.index_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **{k: v for k, v in asdict(index).items() if k != "frames"},
                "frames": [asdict(f) for f in index.frames],
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return index


def read_index(container: str | Path) -> ChunkIndex:
    path = Path(str(container) + ".idx.json")
    if not path.exists():
        raise ChunkError(f"缺少容器索引：{path}")
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return ChunkIndex(
        day=raw["day"],
        codec=raw["codec"],
        frame_lines=raw["frame_lines"],
        container=raw["container"],
        total_lines=raw["total_lines"],
        total_bytes=raw["total_bytes"],
        frames=tuple(Frame(**frame) for frame in raw["frames"]),
    )


def _read_frame(index: ChunkIndex, frame: Frame) -> list[bytes]:
    if index.codec not in _COMPRESSORS:
        raise ChunkError(f"缺少 codec {index.codec}：请先 register_codec()")
    with Path(index.container).open("rb") as handle:
        handle.seek(frame.offset)
        blob = handle.read(frame.comp_len)
    if len(blob) != frame.comp_len:
        raise ChunkError(f"帧 {frame.index} 读取不完整")
    payload = _COMPRESSORS[index.codec][1](blob)
    if hashlib.sha256(payload).hexdigest() != frame.sha256:
        raise ChunkError(f"帧 {frame.index} 校验失败：容器已损坏")
    return payload.split(b"\n")[: frame.line_count]


def read_line(container: str | Path, line_no: int, *, index: ChunkIndex | None = None) -> bytes:
    """按行号精确回取单行原文（只解压所属帧）。"""
    idx = index if index is not None else read_index(container)
    frame = idx.frame_of(line_no)
    rows = _read_frame(idx, frame)
    local = line_no - frame.first_line
    if local >= len(rows):
        raise ChunkError(f"第 {line_no} 行不在帧 {frame.index} 内")
    return rows[local]
