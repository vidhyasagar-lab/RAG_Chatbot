"""Build small DOCX files in tests, so nothing depends on a checked-in binary."""

from __future__ import annotations

import io
import random
import struct
import zlib


def noise_png(seed: int, size: int = 120) -> bytes:
    """A valid PNG of random pixels. Random, so it compresses badly and stays
    above the 5,000-byte floor below which DOCX images are skipped as bullets."""
    rnd = random.Random(seed)
    raw = b"".join(
        b"\x00" + bytes(rnd.getrandbits(8) for _ in range(size * 3)) for _ in range(size)
    )

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def picture(seed: int) -> io.BytesIO:
    return io.BytesIO(noise_png(seed))
