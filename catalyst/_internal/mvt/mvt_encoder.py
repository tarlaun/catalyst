"""Vectorised Mapbox Vector Tile encoder (numpy), byte-compatible with the
Rust core's encoder.

Geometry commands for every feature of a class (points, lines, polygons)
are produced in one numpy pass: coordinates are rounded, ring orientation
enforced (exterior rings clockwise on screen, i.e. positive shoelace area in
y-down tile units), deltas taken along the per-feature cursor, zig-zagged
and varint-encoded as arrays. Only the per-feature framing (tags, type,
length prefixes) is a Python loop, which is cheap next to the geometry.

Tile coordinates here are **y-down** (the MVT convention), unlike
``IntermediateVectorTile``'s y-up frame that ``mapbox_vector_tile`` flips.
"""
from __future__ import annotations

import struct
from typing import Any, Iterable, Sequence

import numpy as np
import shapely

MOVE_TO = 1
LINE_TO = 2
CLOSE_PATH = 7
CMD_CLOSE = (1 << 3) | CLOSE_PATH  # 15
CMD_MOVE1 = (1 << 3) | MOVE_TO  # 9

GEOM_POINT = 1
GEOM_LINE = 2
GEOM_POLYGON = 3


def round_half_away(v: np.ndarray) -> np.ndarray:
    """Rust's ``f64::round`` (half away from zero), unlike numpy's half-to-even."""
    return np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5))


# ---------------------------------------------------------------------------
# protobuf helpers
# ---------------------------------------------------------------------------

def put_varint(v: int) -> bytes:
    out = bytearray()
    while v >= 0x80:
        out.append((v & 0x7F) | 0x80)
        v >>= 7
    out.append(v)
    return bytes(out)


def _bytes_field(field: int, b: bytes) -> bytes:
    return put_varint((field << 3) | 2) + put_varint(len(b)) + b


def _varint_field(field: int, v: int) -> bytes:
    return put_varint(field << 3) + put_varint(v)


def encode_value(v: Any) -> bytes:
    """MVT ``Value`` message for a Python scalar (same choices as the Rust core)."""
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, bool):
        return _varint_field(7, int(v))
    if isinstance(v, int):
        if v >= 0:
            return _varint_field(5, v)
        return _varint_field(6, ((v << 1) ^ (v >> 63)) & 0xFFFFFFFFFFFFFFFF)
    if isinstance(v, float):
        return b"\x19" + struct.pack("<d", v)
    return _bytes_field(1, str(v).encode("utf-8"))


# ---------------------------------------------------------------------------
# vectorised geometry commands
# ---------------------------------------------------------------------------

def _varints_array(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Varint-encode an array of non-negative ints. Returns the flat byte
    array and the byte count of every value."""
    v = vals.astype(np.uint64, copy=False)
    n = len(v)
    nb = np.ones(n, dtype=np.int64)
    for i in range(1, 10):
        nb[v >= (np.uint64(1) << np.uint64(7 * i))] = i + 1
    maxb = int(nb.max()) if n else 1
    mat = np.empty((n, maxb), dtype=np.uint8)
    for i in range(maxb):
        chunk = ((v >> np.uint64(7 * i)) & np.uint64(0x7F)).astype(np.uint8)
        cont = (i < nb - 1).astype(np.uint8) << 7
        mat[:, i] = chunk | cont
    mask = np.arange(maxb)[None, :] < nb[:, None]
    return mat[mask], nb


def _zigzag(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.int64, copy=False)
    return ((v << 1) ^ (v >> 63)).astype(np.uint64)


def _deltas(X: np.ndarray, Y: np.ndarray, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cursor deltas: previous vertex of the same feature, (0, 0) at a feature start."""
    px = np.empty_like(X)
    py = np.empty_like(Y)
    px[1:] = X[:-1]
    py[1:] = Y[:-1]
    start = np.ones(len(X), dtype=bool)
    if len(X) > 1:
        start[1:] = feat[1:] != feat[:-1]
    px[start] = 0
    py[start] = 0
    return X - px, Y - py


def _ring_tokens(
    X: np.ndarray,
    Y: np.ndarray,
    ring_of_vertex: np.ndarray,
    feat_of_ring: np.ndarray,
    is_exterior: np.ndarray,
    closed: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Command tokens for rings (``closed``: polygon rings with CLOSE_PATH,
    otherwise line parts). ``X``/``Y`` are float vertex coordinates without
    a repeated closing vertex, grouped by ring in feature order.
    Returns ``(tokens, feature id of every token)``."""
    n_rings = len(feat_of_ring)
    if n_rings == 0 or len(X) == 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
    n_per_ring = np.bincount(ring_of_vertex, minlength=n_rings).astype(np.int64)
    ring_start = np.concatenate(([0], np.cumsum(n_per_ring)[:-1]))

    if closed:
        # orientation: exterior rings need positive shoelace area (y-down), holes negative
        x_next = np.empty_like(X)
        y_next = np.empty_like(Y)
        x_next[:-1] = X[1:]
        y_next[:-1] = Y[1:]
        last = ring_start + n_per_ring - 1
        x_next[last] = X[ring_start]
        y_next[last] = Y[ring_start]
        cross = X * y_next - x_next * Y
        area2 = np.bincount(ring_of_vertex, weights=cross, minlength=n_rings)
        flip = (area2 > 0) != is_exterior
        if flip.any():
            j = np.arange(len(X)) - ring_start[ring_of_vertex]
            rev = ring_start[ring_of_vertex] + (n_per_ring[ring_of_vertex] - 1 - j)
            src = np.where(flip[ring_of_vertex], rev, np.arange(len(X)))
            X = X[src]
            Y = Y[src]

    Xi = round_half_away(X).astype(np.int64)
    Yi = round_half_away(Y).astype(np.int64)
    feat_of_vertex = feat_of_ring[ring_of_vertex]
    dx, dy = _deltas(Xi, Yi, feat_of_vertex)
    zx = _zigzag(dx)
    zy = _zigzag(dy)

    extra = 3 if closed else 2  # MoveTo, LineTo, (ClosePath)
    tok_len = 2 * n_per_ring + extra
    tok_start = np.concatenate(([0], np.cumsum(tok_len)[:-1]))
    tokens = np.empty(int(tok_len.sum()), dtype=np.uint64)
    j = np.arange(len(X)) - ring_start[ring_of_vertex]
    pos_x = tok_start[ring_of_vertex] + 2 * j + 1 + (j > 0)
    tokens[pos_x] = zx
    tokens[pos_x + 1] = zy
    tokens[tok_start] = CMD_MOVE1
    tokens[tok_start + 3] = ((n_per_ring - 1).astype(np.uint64) << np.uint64(3)) | np.uint64(LINE_TO)
    if closed:
        tokens[tok_start + 2 * n_per_ring + 2] = CMD_CLOSE
    tok_feat = np.repeat(feat_of_ring, tok_len)
    return tokens, tok_feat


def _point_tokens(X: np.ndarray, Y: np.ndarray, feat_of_vertex: np.ndarray, n_feats: int):
    if len(X) == 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
    Xi = round_half_away(X).astype(np.int64)
    Yi = round_half_away(Y).astype(np.int64)
    dx, dy = _deltas(Xi, Yi, feat_of_vertex)
    n_per = np.bincount(feat_of_vertex, minlength=n_feats).astype(np.int64)
    used = np.flatnonzero(n_per)
    tok_len = 2 * n_per[used] + 1
    tok_start = np.concatenate(([0], np.cumsum(tok_len)[:-1]))
    start_of_feat = np.zeros(n_feats, dtype=np.int64)
    start_of_feat[used] = tok_start
    first_vertex = np.concatenate(([0], np.cumsum(n_per)[:-1]))
    j = np.arange(len(X)) - first_vertex[feat_of_vertex]
    tokens = np.empty(int(tok_len.sum()), dtype=np.uint64)
    pos_x = start_of_feat[feat_of_vertex] + 1 + 2 * j
    tokens[pos_x] = _zigzag(dx)
    tokens[pos_x + 1] = _zigzag(dy)
    tokens[tok_start] = (n_per[used].astype(np.uint64) << np.uint64(3)) | np.uint64(MOVE_TO)
    return tokens, np.repeat(used, tok_len)


def _split_by_feature(tokens: np.ndarray, tok_feat: np.ndarray, n_feats: int) -> list[bytes | None]:
    """Varint-encode tokens and return per-feature geometry bytes."""
    out: list[bytes | None] = [None] * n_feats
    if len(tokens) == 0:
        return out
    flat, nb = _varints_array(tokens)
    raw = flat.tobytes()
    per_feat = np.bincount(tok_feat, weights=nb, minlength=n_feats).astype(np.int64)
    # tokens are grouped by feature (sorted feature ids); byte offsets follow
    order_feats = np.flatnonzero(per_feat)
    ends = np.cumsum(per_feat[order_feats])
    starts = ends - per_feat[order_feats]
    for f, s, e in zip(order_feats.tolist(), starts.tolist(), ends.tolist()):
        out[f] = raw[s:e]
    return out


def _polygon_geometry_bytes(geoms: np.ndarray, feat_ids: np.ndarray, n_feats: int) -> list[bytes | None]:
    parts, part_geom = shapely.get_parts(geoms, return_index=True)
    if len(parts) == 0:
        return [None] * n_feats
    rings, ring_part = shapely.get_rings(parts, return_index=True)
    is_ext = np.ones(len(rings), dtype=bool)
    is_ext[1:] = ring_part[1:] != ring_part[:-1]
    coords, coord_ring = shapely.get_coordinates(rings, return_index=True)
    # drop the repeated closing vertex of every ring
    last = np.ones(len(coords), dtype=bool)
    last[:-1] = coord_ring[1:] != coord_ring[:-1]
    keep = ~last
    coords = coords[keep]
    coord_ring = coord_ring[keep]
    n_per_ring = np.bincount(coord_ring, minlength=len(rings))
    ok_ring = n_per_ring >= 3
    if not ok_ring.all():
        vmask = ok_ring[coord_ring]
        coords = coords[vmask]
        coord_ring = coord_ring[vmask]
        # renumber rings
        remap = np.cumsum(ok_ring) - 1
        coord_ring = remap[coord_ring]
        ring_part = ring_part[ok_ring]
        is_ext = is_ext[ok_ring]
    feat_of_ring = feat_ids[part_geom[ring_part]]
    tokens, tok_feat = _ring_tokens(coords[:, 0], coords[:, 1], coord_ring, feat_of_ring, is_ext, closed=True)
    return _split_by_feature(tokens, tok_feat, n_feats)


def _line_geometry_bytes(geoms: np.ndarray, feat_ids: np.ndarray, n_feats: int) -> list[bytes | None]:
    parts, part_geom = shapely.get_parts(geoms, return_index=True)
    if len(parts) == 0:
        return [None] * n_feats
    coords, coord_part = shapely.get_coordinates(parts, return_index=True)
    n_per = np.bincount(coord_part, minlength=len(parts))
    ok = n_per >= 2
    if not ok.all():
        vmask = ok[coord_part]
        coords = coords[vmask]
        coord_part = np.cumsum(ok)[coord_part[vmask]] - 1
        part_geom = part_geom[ok]
    feat_of_part = feat_ids[part_geom]
    tokens, tok_feat = _ring_tokens(
        coords[:, 0], coords[:, 1], coord_part, feat_of_part, np.ones(len(part_geom), dtype=bool), closed=False
    )
    return _split_by_feature(tokens, tok_feat, n_feats)


def _point_geometry_bytes(geoms: np.ndarray, feat_ids: np.ndarray, n_feats: int) -> list[bytes | None]:
    coords, coord_geom = shapely.get_coordinates(geoms, return_index=True)
    if len(coords) == 0:
        return [None] * n_feats
    tokens, tok_feat = _point_tokens(coords[:, 0], coords[:, 1], feat_ids[coord_geom], n_feats)
    return _split_by_feature(tokens, tok_feat, n_feats)


def geometry_commands(geoms: np.ndarray) -> tuple[list[bytes | None], np.ndarray]:
    """Per-geometry MVT command bytes and MVT geometry types for an array of
    shapely geometries in y-down tile units. Empty / unsupported -> None."""
    geoms = np.asarray(geoms, dtype=object)
    n = len(geoms)
    out: list[bytes | None] = [None] * n
    if n == 0:
        return out, np.zeros(0, dtype=np.int64)
    tid = shapely.get_type_id(geoms)
    mvt_type = np.zeros(n, dtype=np.int64)
    classes = (
        ((tid == 3) | (tid == 6), _polygon_geometry_bytes, GEOM_POLYGON),
        ((tid == 1) | (tid == 5) | (tid == 2), _line_geometry_bytes, GEOM_LINE),
        ((tid == 0) | (tid == 4), _point_geometry_bytes, GEOM_POINT),
    )
    for mask, fn, mtype in classes:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        res = fn(geoms[idx], idx, n)
        for i in idx.tolist():
            if res[i]:
                out[i] = res[i]
                mvt_type[i] = mtype
    return out, mvt_type


def packed_square_dots(centers: np.ndarray, half: float) -> bytes | None:
    """One MultiPolygon of axis-aligned squares (side ``2*half``) around
    ``centers`` (N×2, y-down tile units), sorted by (y, x) like the Rust core."""
    if len(centers) == 0:
        return None
    c = np.asarray(centers, dtype=np.float64)
    order = np.lexsort((round_half_away(c[:, 0]), round_half_away(c[:, 1])))
    c = c[order]
    n = len(c)
    X = np.empty(4 * n)
    Y = np.empty(4 * n)
    X[0::4] = c[:, 0] - half
    Y[0::4] = c[:, 1] - half
    X[1::4] = c[:, 0] + half
    Y[1::4] = c[:, 1] - half
    X[2::4] = c[:, 0] + half
    Y[2::4] = c[:, 1] + half
    X[3::4] = c[:, 0] - half
    Y[3::4] = c[:, 1] + half
    ring = np.repeat(np.arange(n), 4)
    tokens, tok_feat = _ring_tokens(X, Y, ring, np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool), closed=True)
    return _split_by_feature(tokens, tok_feat, 1)[0]


def packed_segment_dots(centers: np.ndarray, half: float) -> bytes | None:
    """One MultiLineString of horizontal one-pixel segments."""
    if len(centers) == 0:
        return None
    c = np.asarray(centers, dtype=np.float64)
    order = np.lexsort((round_half_away(c[:, 0]), round_half_away(c[:, 1])))
    c = c[order]
    n = len(c)
    X = np.empty(2 * n)
    Y = np.empty(2 * n)
    X[0::2] = c[:, 0] - half
    X[1::2] = c[:, 0] + half
    Y[0::2] = c[:, 1]
    Y[1::2] = c[:, 1]
    part = np.repeat(np.arange(n), 2)
    tokens, tok_feat = _ring_tokens(X, Y, part, np.zeros(n, dtype=np.int64), np.ones(n, dtype=bool), closed=False)
    return _split_by_feature(tokens, tok_feat, 1)[0]


# ---------------------------------------------------------------------------
# layer / tile framing
# ---------------------------------------------------------------------------

class LayerEncoder:
    """Collects features (already-encoded geometry) and frames the layer the
    way the Rust ``LayerBuilder`` does: keys and values are interned in first-
    use order, features keep insertion order."""

    def __init__(self, name: str, extent: int) -> None:
        self.name = name
        self.extent = int(extent)
        self._keys: list[bytes] = []
        self._key_idx: dict[str, int] = {}
        self._values: list[bytes] = []
        self._value_idx: dict[tuple, int] = {}
        self._features: list[bytes] = []
        self.feature_count = 0

    def _key(self, k: str) -> int:
        i = self._key_idx.get(k)
        if i is None:
            i = len(self._keys)
            self._keys.append(_bytes_field(3, k.encode("utf-8")))
            self._key_idx[k] = i
        return i

    def _value(self, v: Any) -> int:
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, bool):
            key = ("b", v)
        elif isinstance(v, int):
            key = ("i", v)
        elif isinstance(v, float):
            key = ("f", struct.pack("<d", v))
        else:
            key = ("s", str(v))
        i = self._value_idx.get(key)
        if i is None:
            i = len(self._values)
            self._values.append(_bytes_field(4, encode_value(v)))
            self._value_idx[key] = i
        return i

    def add(self, geom_bytes: bytes, mvt_type: int, props: Iterable[tuple[str, Any]] | None = None) -> None:
        body = bytearray()
        if props:
            tags = bytearray()
            for k, v in props:
                if v is None:
                    continue
                tags += put_varint(self._key(k))
                tags += put_varint(self._value(v))
            if tags:
                body += _bytes_field(2, bytes(tags))
        body += _varint_field(3, mvt_type)
        body += _bytes_field(4, geom_bytes)
        self._features.append(_bytes_field(2, bytes(body)))
        self.feature_count += 1

    def encode_layer(self) -> bytes:
        out = bytearray(_varint_field(15, 2))
        out += _bytes_field(1, self.name.encode("utf-8"))
        for f in self._features:
            out += f
        for k in self._keys:
            out += k
        for v in self._values:
            out += v
        out += _varint_field(5, self.extent)
        return bytes(out)

    def encode_tile(self) -> bytes:
        return _bytes_field(3, self.encode_layer())


def encode_tile(layers: Sequence[LayerEncoder]) -> bytes:
    return b"".join(_bytes_field(3, layer.encode_layer()) for layer in layers)
