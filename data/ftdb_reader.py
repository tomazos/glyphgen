from __future__ import annotations

import io
import os
import struct
import mmap
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Iterable, Dict

# ================================================================
# FTDB (Font Contours Pack) Reader
#   - Zero-copy where practical using mmap
#   - Lazy decoding of strings and geometry
#   - Designed for very large files (multi‑GB)
# ================================================================

MAGIC = b"FTDB"
TAG_PNTS = b"PNTS"
TAG_FNTS = b"FNTS"
TAG_GLFS = b"GLFS"
TAG_CNTS = b"CNTS"
TAG_CPGI = b"CPGI"
TAG_CDPS = b"CDPS"
TAG_STRS = b"STRS"

_u32 = struct.Struct("<I")
_u64 = struct.Struct("<Q")
_f32 = struct.Struct("<f")
_point = struct.Struct("<ff")  # x, y

# Record structs (no padding / packed)
# GLFS record: codepoint (u32), first_contour (u64), num_contours (u64)
_glfs_rec = struct.Struct("<IQQ")  # 4 + 8 + 8 = 20 bytes
# CNTS record: first_point (u64), num_points (u64)
_cnts_rec = struct.Struct("<QQ")    # 16 bytes
# FNTS record: 5 * u64 (3 string offs + first_glyph + num_glyphs)
_fnts_rec = struct.Struct("<QQQQQ") # 40 bytes
# CDPS record: codepoint (u32), start_in_cpgi (u64), count_in_cpgi (u64)
_cdps_rec = struct.Struct("<IQQ")   # 20 bytes


@dataclass(frozen=True)
class FontInfo:
    index: int
    family: str
    subfamily: str
    path: str
    first_glyph: int
    num_glyphs: int

    def glyph_range(self) -> range:
        return range(self.first_glyph, self.first_glyph + self.num_glyphs)


@dataclass(frozen=True)
class GlyphInfo:
    index: int
    codepoint: int
    first_contour: int
    num_contours: int

    def contour_range(self) -> range:
        return range(self.first_contour, self.first_contour + self.num_contours)


@dataclass(frozen=True)
class ContourInfo:
    index: int
    first_point: int
    num_points: int  # must be even (on, off, on, off, ...)

    def point_range(self) -> range:
        return range(self.first_point, self.first_point + self.num_points)


class FTDB:
    """Memory-mapped reader for the FTDB font contours pack.

    Usage:
        with FTDB(path) as db:
            print(db.point_count)
            print(len(db.fonts))
            glyph_idxs = db.glyphs_for_codepoint(0x41)  # 'A'
            g = db.glyph(glyph_idxs[0])
            for contour in db.iter_contour_points_of_glyph(g.index):
                # contour is an iterator of (x, y) floats
                for x, y in contour:
                    pass
    """

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "rb")
        self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        self.size = self._mm.size()

        # Section offsets and counts (absolute)
        self._off_magic = 0
        self._off_pnts_tag = 4
        self._off_pnts_count = 8
        self._off_pnts_data = 0x10

        self._off_fnts_tag: int = 0
        self._off_fnts_count: int = 0
        self._off_fnts_records: int = 0

        self._off_glfs_tag: int = 0
        self._off_glfs_count: int = 0
        self._off_glfs_records: int = 0

        self._off_cnts_tag: int = 0
        self._off_cnts_count: int = 0
        self._off_cnts_records: int = 0

        self._off_cpgi_tag: int = 0
        self._off_cpgi_count: int = 0
        self._off_cpgi_array: int = 0

        self._off_cdps_tag: int = 0
        self._off_cdps_count: int = 0
        self._off_cdps_records: int = 0

        self._off_strs_tag: int = 0
        self._off_strs_count: int = 0
        self._off_strs_blob: int = 0

        # Caches
        self._str_cache: Dict[int, str] = {}
        self._fonts_cache: Optional[List[FontInfo]] = None
        self._glyph_count: Optional[int] = None
        self._contour_count: Optional[int] = None
        self._codepoint_to_slice_cache: Dict[int, Tuple[int, int]] = {}

        self._parse_structure()

    # ------------------------------
    # Context manager & cleanup
    # ------------------------------
    def close(self):
        if getattr(self, "_mm", None) is not None:
            try:
                self._mm.close()
            finally:
                self._mm = None  # type: ignore
        if getattr(self, "_fh", None) is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None  # type: ignore

    def __enter__(self) -> "FTDB":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ------------------------------
    # Parsing helpers
    # ------------------------------
    def _read_exact(self, off: int, n: int) -> bytes:
        b = self._mm[off:off+n]
        if len(b) != n:
            raise EOFError("Unexpected EOF while reading")
        return b

    def _expect_tag(self, off: int, tag: bytes) -> None:
        actual = self._read_exact(off, 4)
        if actual != tag:
            raise ValueError(f"Expected tag {tag!r} at 0x{off:X}, found {actual!r}")

    def _u64_at(self, off: int) -> int:
        return _u64.unpack_from(self._mm, off)[0]

    def _u32_at(self, off: int) -> int:
        return _u32.unpack_from(self._mm, off)[0]

    def _parse_structure(self) -> None:
        # FTDB magic
        self._expect_tag(0x0000, MAGIC)
        # PNTS
        self._expect_tag(0x0004, TAG_PNTS)
        point_count = self._u64_at(0x0008)
        self._point_count = point_count
        pnts_bytes = point_count * _point.size  # 8 * point_count
        next_off = self._off_pnts_data + pnts_bytes

        # FNTS
        self._expect_tag(next_off + 0, TAG_FNTS)
        self._off_fnts_tag = next_off
        self._off_fnts_count = next_off + 4
        f_cnt = self._u64_at(self._off_fnts_count)
        self._off_fnts_records = self._off_fnts_count + 8
        next_off = self._off_fnts_records + f_cnt * _fnts_rec.size

        # GLFS
        self._expect_tag(next_off + 0, TAG_GLFS)
        self._off_glfs_tag = next_off
        self._off_glfs_count = next_off + 4
        g_cnt = self._u64_at(self._off_glfs_count)
        self._glyph_count = g_cnt
        self._off_glfs_records = self._off_glfs_count + 8
        next_off = self._off_glfs_records + g_cnt * _glfs_rec.size

        # CNTS
        self._expect_tag(next_off + 0, TAG_CNTS)
        self._off_cnts_tag = next_off
        self._off_cnts_count = next_off + 4
        c_cnt = self._u64_at(self._off_cnts_count)
        self._contour_count = c_cnt
        self._off_cnts_records = self._off_cnts_count + 8
        next_off = self._off_cnts_records + c_cnt * _cnts_rec.size

        # CPGI
        self._expect_tag(next_off + 0, TAG_CPGI)
        self._off_cpgi_tag = next_off
        self._off_cpgi_count = next_off + 4
        gi_cnt = self._u64_at(self._off_cpgi_count)
        self._off_cpgi_array = self._off_cpgi_count + 8
        next_off = self._off_cpgi_array + gi_cnt * _u64.size

        # CDPS
        self._expect_tag(next_off + 0, TAG_CDPS)
        self._off_cdps_tag = next_off
        self._off_cdps_count = next_off + 4
        cp_cnt = self._u64_at(self._off_cdps_count)
        self._off_cdps_records = self._off_cdps_count + 8
        next_off = self._off_cdps_records + cp_cnt * _cdps_rec.size

        # STRS
        self._expect_tag(next_off + 0, TAG_STRS)
        self._off_strs_tag = next_off
        self._off_strs_count = next_off + 4
        s_cnt = self._u64_at(self._off_strs_count)
        self._off_strs_blob = self._off_strs_count + 8
        # We do not need to parse STRS contents now. Offsets are absolute.

        # Basic sanity checks
        if any(off < 0 or off > self.size for off in [
            self._off_fnts_records,
            self._off_glfs_records,
            self._off_cnts_records,
            self._off_cpgi_array,
            self._off_cdps_records,
            self._off_strs_blob,
        ]):
            raise ValueError("Corrupt FTDB: computed offsets out of range")

    # ------------------------------
    # Top-level properties
    # ------------------------------
    @property
    def point_count(self) -> int:
        return self._point_count

    @property
    def glyph_count(self) -> int:
        assert self._glyph_count is not None
        return self._glyph_count

    @property
    def contour_count(self) -> int:
        assert self._contour_count is not None
        return self._contour_count

    # ------------------------------
    # String decoding
    # ------------------------------
    def _decode_c_string_at(self, abs_off: int) -> str:
        """Decode a zero-terminated UTF-8 string at absolute offset.
        Caches decoded results for reuse.
        """
        if abs_off in self._str_cache:
            return self._str_cache[abs_off]
        # Find NUL terminator without copying whole blob
        mv = memoryview(self._mm)
        start = abs_off
        end = start
        # Scan in chunks to avoid O(file) slicing
        CHUNK = 1 << 16
        while True:
            chunk = mv[end:end+CHUNK]
            if not chunk:
                raise ValueError("Unterminated string in STRS")
            idx = bytes(chunk).find(b"\x00")
            if idx != -1:
                end += idx
                break
            end += CHUNK
        raw = self._mm[start:end]
        s = raw.decode("utf-8", errors="strict")
        self._str_cache[abs_off] = s
        return s

    # ------------------------------
    # Fonts
    # ------------------------------
    @property
    def fonts(self) -> List[FontInfo]:
        if self._fonts_cache is not None:
            return self._fonts_cache
        f_cnt = self._u64_at(self._off_fnts_count)
        base = self._off_fnts_records
        fonts: List[FontInfo] = []
        for i in range(f_cnt):
            off = base + i * _fnts_rec.size
            (fam_off, subfam_off, path_off, first_glyph, num_glyphs) = _fnts_rec.unpack_from(self._mm, off)
            family = self._decode_c_string_at(fam_off) if fam_off else ""
            subfamily = self._decode_c_string_at(subfam_off) if subfam_off else ""
            path = self._decode_c_string_at(path_off) if path_off else ""
            fonts.append(FontInfo(i, family, subfamily, path, first_glyph, num_glyphs))
        self._fonts_cache = fonts
        return fonts

    def find_fonts(self, family_substr: str) -> List[FontInfo]:
        q = family_substr.casefold()
        return [f for f in self.fonts if q in f.family.casefold()]

    # ------------------------------
    # Glyphs
    # ------------------------------
    def glyph(self, index: int) -> GlyphInfo:
        g_cnt = self.glyph_count
        if not (0 <= index < g_cnt):
            raise IndexError("glyph index out of range")
        off = self._off_glfs_records + index * _glfs_rec.size
        codepoint, first_contour, num_contours = _glfs_rec.unpack_from(self._mm, off)
        return GlyphInfo(index, codepoint, first_contour, num_contours)

    def iter_glyphs(self, start: int = 0, count: Optional[int] = None) -> Iterator[GlyphInfo]:
        g_cnt = self.glyph_count
        if count is None:
            end = g_cnt
        else:
            end = min(g_cnt, start + count)
        for i in range(start, end):
            yield self.glyph(i)

    # ------------------------------
    # Contours
    # ------------------------------
    def contour(self, index: int) -> ContourInfo:
        c_cnt = self.contour_count
        if not (0 <= index < c_cnt):
            raise IndexError("contour index out of range")
        off = self._off_cnts_records + index * _cnts_rec.size
        first_point, num_points = _cnts_rec.unpack_from(self._mm, off)
        if num_points % 2 != 0:
            raise ValueError(f"Contour {index} has odd num_points={num_points}")
        return ContourInfo(index, first_point, num_points)

    # ------------------------------
    # Points access
    # ------------------------------
    def point(self, index: int) -> Tuple[float, float]:
        if not (0 <= index < self.point_count):
            raise IndexError("point index out of range")
        off = self._off_pnts_data + index * _point.size
        return _point.unpack_from(self._mm, off)

    def iter_points(self, start: int = 0, count: Optional[int] = None) -> Iterator[Tuple[float, float]]:
        end = self.point_count if count is None else min(self.point_count, start + count)
        base = self._off_pnts_data + start * _point.size
        for i in range(start, end):
            yield _point.unpack_from(self._mm, base + (i - start) * _point.size)

    # ------------------------------
    # High-level geometry helpers
    # ------------------------------
    def iter_contour_points(self, contour_index: int) -> Iterator[Tuple[float, float]]:
        ci = self.contour(contour_index)
        base = self._off_pnts_data + ci.first_point * _point.size
        for j in range(ci.num_points):
            yield _point.unpack_from(self._mm, base + j * _point.size)

    def iter_contour_points_of_glyph(self, glyph_index: int) -> Iterator[Iterator[Tuple[float, float]]]:
        g = self.glyph(glyph_index)
        for cidx in range(g.first_contour, g.first_contour + g.num_contours):
            yield self.iter_contour_points(cidx)

    def iter_quadratic_segments(self, contour_index: int) -> Iterator[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """Yield quadratic Bezier segments as (P0, C, P1) tuples for a contour.
        Encoding rule from spec: even i -> on-curve P, odd i -> off-curve C.
        segments are (P[i], C[i+1], P[(i+2) % N]).
        """
        ci = self.contour(contour_index)
        if ci.num_points == 0:
            return
        base = self._off_pnts_data + ci.first_point * _point.size
        N = ci.num_points
        # prefetch first two points to avoid extra modulo ops in tight loop
        def read_point(k: int) -> Tuple[float, float]:
            return _point.unpack_from(self._mm, base + k * _point.size)
        for i in range(0, N, 2):
            p0 = read_point(i % N)
            c1 = read_point((i + 1) % N)
            p1 = read_point((i + 2) % N)
            yield (p0, c1, p1)

    def glyph_quadratic_segments(self, glyph_index: int) -> Iterator[Iterator[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]]:
        g = self.glyph(glyph_index)
        for cidx in range(g.first_contour, g.first_contour + g.num_contours):
            yield self.iter_quadratic_segments(cidx)

    # ------------------------------
    # Codepoint → glyph mapping (CDPS / CPGI)
    # ------------------------------
    def _lookup_cp_slice(self, codepoint: int) -> Optional[Tuple[int, int]]:
        """Return (start, count) into CPGI for a codepoint, or None if not mapped."""
        if codepoint in self._codepoint_to_slice_cache:
            return self._codepoint_to_slice_cache[codepoint]
        cp_cnt = self._u64_at(self._off_cdps_count)
        base = self._off_cdps_records
        # Binary search over ascending codepoints
        lo, hi = 0, cp_cnt
        while lo < hi:
            mid = (lo + hi) // 2
            off = base + mid * _cdps_rec.size
            cp_mid = _u32.unpack_from(self._mm, off)[0]
            if cp_mid < codepoint:
                lo = mid + 1
            else:
                hi = mid
        if lo >= cp_cnt:
            self._codepoint_to_slice_cache[codepoint] = None  # type: ignore
            return None
        off = base + lo * _cdps_rec.size
        cp_val, start, count = _cdps_rec.unpack_from(self._mm, off)
        if cp_val != codepoint:
            self._codepoint_to_slice_cache[codepoint] = None  # type: ignore
            return None
        self._codepoint_to_slice_cache[codepoint] = (start, count)
        return (start, count)

    def glyphs_for_codepoint(self, codepoint: int) -> List[int]:
        """Return list of glyph indices (into GLFS) for the given Unicode codepoint."""
        sl = self._lookup_cp_slice(codepoint)
        if sl is None:
            return []
        start, count = sl
        if count == 0:
            return []
        base = self._off_cpgi_array + start * _u64.size
        # Read without building a giant slice (large files)
        return [ _u64.unpack_from(self._mm, base + i * _u64.size)[0] for i in range(count) ]

    # ------------------------------
    # Convenience: decode full glyph into list of contours of (x, y)
    # ------------------------------
    def decode_glyph_points(self, glyph_index: int) -> List[List[Tuple[float, float]]]:
        g = self.glyph(glyph_index)
        contours: List[List[Tuple[float, float]]] = []
        for cidx in range(g.first_contour, g.first_contour + g.num_contours):
            ci = self.contour(cidx)
            base = self._off_pnts_data + ci.first_point * _point.size
            pts = [ _point.unpack_from(self._mm, base + j * _point.size) for j in range(ci.num_points) ]
            contours.append(pts)
        return contours

    # ------------------------------
    # Safety / validation helpers (lightweight)
    # ------------------------------
    def validate(self) -> None:
        """Run a set of inexpensive structural checks.
        Raises ValueError on problems; does not fully validate geometry."""
        # Check font ranges lie within GLFS
        g_cnt = self.glyph_count
        for f in self.fonts:
            if f.first_glyph + f.num_glyphs > g_cnt:
                raise ValueError(f"Font {f.index} glyph range out of bounds")
        # Check contours lie within points
        p_cnt = self.point_count
        c_cnt = self.contour_count
        base = self._off_cnts_records
        for i in range(c_cnt):
            first_point, num_points = _cnts_rec.unpack_from(self._mm, base + i * _cnts_rec.size)
            if num_points % 2 != 0:
                raise ValueError(f"Contour {i} has odd num_points={num_points}")
            if first_point + num_points > p_cnt:
                raise ValueError(f"Contour {i} point range out of bounds")


# ------------------------------------------------
# Minimal CLI for quick introspection (optional)
#   python ftdb_reader.py /path/to/file.ftdb "A"
# ------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python ftdb_reader.py <file> [char]")
        raise SystemExit(2)
    path = sys.argv[1]
    char = sys.argv[2] if len(sys.argv) >= 3 else None

    with FTDB(path) as db:
        print(f"File: {path}")
        print(f"Size: {db.size:,} bytes")
        print(f"Points: {db.point_count:,}")
        print(f"Glyphs: {db.glyph_count:,}")
        print(f"Contours: {db.contour_count:,}")
        print(f"Fonts: {len(db.fonts):,}")
        if char:
            cp = ord(char) if len(char) == 1 else int(char, 0)
            gi = db.glyphs_for_codepoint(cp)
            print(f"Codepoint U+{cp:04X} -> {len(gi)} glyph(s): {gi[:8]}{'...' if len(gi)>8 else ''}")
            if gi:
                g0 = db.glyph(gi[0])
                print(f"  First glyph #{g0.index}: codepoint=U+{g0.codepoint:04X}, contours={g0.num_contours}")
                # Print first contour length
                if g0.num_contours:
                    c0 = db.contour(g0.first_contour)
                    print(f"  First contour points: {c0.num_points}")
