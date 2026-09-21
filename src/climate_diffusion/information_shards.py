"""Immutable, independently verified physical-information chunks.

Readers never see a partial chunk. Missing chunks fail fast; orchestration waits
outside training. Affine statistics are fitted once on the complete train prefix,
not changed whenever more data arrives. The surface archive remains unchanged.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .physical_information import FORMAT, digest, validate_information
from .temporal_supervision import area_weights

STORE_FORMAT = 'climate_diffusion.information_shards.v1'


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, write):
    """Flush bytes before atomic, no-overwrite publication on the same volume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # raises if already published, including symlinks
        sync_directory(path.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_json(path, value):
    content = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    publish(path, lambda stream: stream.write(content))


def read_json(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f'Symlink forbidden in owned store: {path}')
    return json.loads(path.read_text())


class InformationShards:
    """A read-only [time, feature] array with a bounded chunk LRU.

    File receipts are pinned in checkpoints, in addition to the immutable store
    identity. Later publication cannot change a chunk consumed by an earlier stage.
    """

    def __init__(self, path, archive, times, schema, cache_chunks=4):
        self.path = Path(path)
        self.plan = read_json(self.path / 'plan.json')
        self.meta = read_json(self.path / 'metadata.json')
        self.plan_sha = digest(self.path / 'plan.json')
        self.times = np.asarray(times).astype('datetime64[ns]')
        self.schema = schema
        self.cache_chunks = cache_chunks
        self.cache = OrderedDict()
        self.used = {}
        self.expected = {}
        if self.plan['format'] != STORE_FORMAT or self.meta['format'] != FORMAT:
            raise ValueError('Unsupported information shard format')
        if self.plan['surface_sha256'] != digest(archive) or self.meta['surface_sha256'] != self.plan['surface_sha256']:
            raise ValueError('Information archive hash mismatch')
        if self.meta['source_sha256'] != self.plan_sha:
            raise ValueError('Information plan/metadata mismatch')
        if self.plan['schema_sha256'] != digest(Path(archive).with_suffix('.schema.json')):
            raise ValueError('Surface schema hash mismatch')
        if [v['name'] for v in self.meta['variables']] != self.plan['fields']:
            raise ValueError('Information variable order mismatch')
        if self.plan['grid'] != schema['variables'][0]['coords'] or self.meta['grid'] != self.plan['grid']:
            raise ValueError('Information grid mismatch')
        if self.plan['times_sha256'] != hashlib.sha256(self.times.tobytes()).hexdigest():
            raise ValueError('Information exact UTC times mismatch')
        if len(self.times) < 2 or not np.all(np.diff(self.times) == np.timedelta64(6, 'h')):
            raise ValueError('Information requires gap-free 6h times')
        self.ranges = self.plan['ranges']
        cursor = 0
        for start, stop in self.ranges:
            if start != cursor or stop <= start or stop > len(times):
                raise ValueError('Invalid shard boundaries')
            cursor = stop
        if cursor != len(times):
            raise ValueError('Shard plan does not cover archive')
        self.ends = np.array([r[1] for r in self.ranges])
        self.shape = (len(times), int(np.prod(self.meta['shape'])))
        self.dtype = np.dtype('float32')
        terrain = self.path / 'terrain.npz'
        if terrain.is_symlink() or digest(terrain) != self.meta['terrain_sha256']:
            raise ValueError('Terrain checksum mismatch')
        with np.load(terrain, allow_pickle=False) as f:
            self.terrain = f['data'].copy()

    def pin(self, expected):
        self.expected = dict(expected or {})
        for key, sha in self.expected.items():
            receipt = read_json(self.receipt_path(int(key)))
            if receipt['sha256'] != sha:
                raise ValueError('Previously consumed information shard changed')

    def receipt_path(self, i):
        return self.path / 'chunks' / f'{i:06d}.json'

    def chunk_path(self, i):
        return self.path / 'chunks' / f'{i:06d}.npz'

    def validate_chunk(self, i, path=None):
        path = Path(path) if path is not None else self.chunk_path(i)
        if path.is_symlink():
            raise ValueError('Symlink chunk forbidden')
        start, stop = self.ranges[i]
        with np.load(path, allow_pickle=False) as f:
            data = f['data']
            if data.dtype != np.float32:
                raise ValueError('Shard data must be float32')
            if str(f['plan_sha256']) != self.plan_sha:
                raise ValueError('Chunk plan identity mismatch')
            validate_information(data, self.meta, f['times'], f['observed_mask'],
                                 self.times[start:stop], self.schema)
            if not np.array_equal(data.reshape(stop-start, *self.meta['shape'])[:, -2:],
                                  np.broadcast_to(self.terrain, (stop-start, *self.terrain.shape))):
                raise ValueError('Terrain must be identical across chunks')
            sources = json.loads(str(f['sources_json']))
        return data, sources

    def chunk(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        record_path = self.receipt_path(i)
        if not record_path.exists():
            raise FileNotFoundError(f'Chunk {i} not committed: {record_path}; wait for producer readiness')
        record = read_json(record_path)
        if record['plan_sha256'] != self.plan_sha or record['range'] != self.ranges[i]:
            raise ValueError('Shard receipt identity mismatch')
        path = self.chunk_path(i)
        sha = digest(path)
        if record['sha256'] != sha or (str(i) in self.expected and self.expected[str(i)] != sha):
            raise ValueError('Shard checksum or checkpoint provenance mismatch')
        data, _ = self.validate_chunk(i)
        self.used[str(i)] = sha
        self.cache[i] = data
        if len(self.cache) > self.cache_chunks:
            self.cache.popitem(last=False)
        return data

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            j = int(key)
            if j < 0:
                j += len(self)
            if not 0 <= j < len(self):
                raise IndexError(j)
            i = int(np.searchsorted(self.ends, j, side='right'))
            return self.chunk(i)[j-self.ranges[i][0]]
        if not isinstance(key, slice):
            raise TypeError('Shards support integer rows or contiguous time slices')
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError('Expected contiguous slice')
        out = np.empty((max(0, stop-start), self.shape[1]), np.float32)
        cursor = start
        while cursor < stop:
            i = int(np.searchsorted(self.ends, cursor, side='right'))
            left, right = self.ranges[i]
            end = min(stop, right)
            out[cursor-start:end-start] = self.chunk(i)[cursor-left:end-left]
            cursor = end
        return out

    def normalized(self, mean, scale):
        return NormalizedInformation(self, mean, scale)

    def provenance(self):
        return {**self.expected, **self.used}

    def statistics(self, train_end, block_rows=256):
        """Two-pass float64 moments, unique train rows/pairs, boundary pairs kept."""
        if not 2 <= train_end <= len(self):
            raise ValueError('Need at least two train observations')
        shape = self.meta['shape']
        w = area_weights(self.schema)
        def blocks(delta=False):
            for start in range(0, train_end-(1 if delta else 0), block_rows):
                end = min(train_end-(1 if delta else 0), start+block_rows)
                a = self[start:end+(1 if delta else 0)].reshape(-1, *shape).astype(np.float64)
                yield np.diff(a, axis=0)/6 if delta else a
        def moments(delta=False):
            count = train_end-(1 if delta else 0)
            mean = sum((a*w).sum((0, 2, 3)) for a in blocks(delta))/count
            variance = sum(((a-mean[None,:,None,None])**2*w).sum((0, 2, 3)) for a in blocks(delta))/count
            return mean, np.sqrt(np.maximum(variance, 0))
        mean, sd = moments()
        scale = np.maximum(sd, np.maximum(np.abs(mean)*1e-6, 1e-6))
        _, tendency_sd = moments(True)
        tendency_scale = np.maximum(tendency_sd, np.maximum(scale*1e-3/6, 1e-8))
        expand = lambda x: np.broadcast_to(x[:,None,None], shape).copy().reshape(-1).astype(np.float32)
        return expand(mean), expand(scale), expand(tendency_scale)


class NormalizedInformation:
    def __init__(self, source, mean, scale):
        self.source = source
        self.mean, self.scale = np.asarray(mean), np.asarray(scale)
        self.shape = source.shape

    def __getitem__(self, key):
        return ((self.source[key]-self.mean)/self.scale).astype(np.float32)

    def provenance(self):
        return self.source.provenance()
