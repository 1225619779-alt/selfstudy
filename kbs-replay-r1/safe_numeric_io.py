"""Small, immutable numeric checkpoints for offline scientific experiments.

No solver, model, network service or project module is imported. Works on a
local filesystem supporting hard links (the project's WSL filesystem does).
Every expensive stage should persist its minimal numeric return BEFORE the
next stage. An arbitrary solver object's graph is deliberately unsupported.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import numpy as np


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _directory_sync(path: Path) -> None:
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _publish(path: Path, writer) -> None:
    """Publish a complete file without replacing any previous artifact.

    Temp and destination are on the SAME filesystem. A successful hard link
    is an atomic create-if-absent operation. A failure after publication may
    leave a valid destination: inspect it; never re-run a solver on this basis.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.tmp', dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as f:
            writer(f)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # refuses to overwrite; no silent compatibility fallback
        _directory_sync(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def write_json_new(path: str | Path, obj: Any) -> None:
    data = json.dumps(obj, ensure_ascii=False, allow_nan=False,
                      sort_keys=True, indent=2).encode('utf-8')
    _publish(Path(path), lambda f: f.write(data))


def save_numeric_new(path: str | Path, payload: Any, *, compressed: bool = False) -> dict:
    """Save arrays/scalars/JSON containers; reject custom objects and cycles.

    Non-finite scalars are losslessly tagged, not changed to zero. This is a
    transport rule, NOT a scientific declaration that those values are valid.
    Uncompressed storage is default for the hot path; compress final delivery.
    """
    arrays: dict[str, np.ndarray] = {}
    active: set[int] = set()

    def enc(x):
        if isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise TypeError('object-dtype arrays are not scientific checkpoints')
            a = np.ascontiguousarray(x).copy() if x.ndim else x.copy()
            key = f'a{len(arrays):05d}'
            arrays[key] = a
            return {'__array__': key, 'dtype': a.dtype.str, 'shape': list(a.shape),
                    'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
        if isinstance(x, np.generic):
            return enc(x.item())
        if isinstance(x, float) and not math.isfinite(x):
            return {'__float__': x.hex()}
        if isinstance(x, complex):
            return {'__complex__': [enc(x.real), enc(x.imag)]}
        if x is None or type(x) in (str, bool, int, float):
            return x
        if type(x) in (dict, list, tuple):
            if id(x) in active:
                raise TypeError('cyclic object graph: store numeric projection first')
            active.add(id(x))
            try:
                if type(x) is dict:
                    if any(type(k) is not str for k in x):
                        raise TypeError('metadata dictionary keys must be strings')
                    # Wrapping prevents user keys from colliding with transport tags.
                    return {'__dict__': [[k, enc(v)] for k, v in x.items()]}
                return {'__sequence__': [enc(v) for v in x], 'tuple': type(x) is tuple}
            finally:
                active.remove(id(x))
        raise TypeError(f'unsupported metadata type: {type(x).__module__}.{type(x).__name__}')

    meta = {'schema': 'P05C_NUMERIC_CHECKPOINT_V1', 'payload': enc(payload)}
    arrays['metadata_utf8'] = np.frombuffer(json.dumps(meta, ensure_ascii=False,
                                           allow_nan=False).encode('utf-8'), np.uint8)
    saver = np.savez_compressed if compressed else np.savez
    path = Path(path)
    _publish(path, lambda f: saver(f, **arrays))
    # Transport round-trip, not a new scientific computation.
    load_numeric(path)
    return {'path': str(path), 'sha256': sha256_file(path), 'bytes': path.stat().st_size}


def load_numeric(path: str | Path) -> Any:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(z['metadata_utf8'].tobytes())
        if meta.get('schema') != 'P05C_NUMERIC_CHECKPOINT_V1':
            raise ValueError('unknown numeric checkpoint schema')

        def dec(x):
            if not isinstance(x, dict):
                return x
            if '__array__' in x:
                a = z[x['__array__']]
                if a.dtype.hasobject or a.dtype.str != x['dtype'] or list(a.shape) != x['shape']:
                    raise ValueError('array type/shape mismatch')
                if hashlib.sha256(a.tobytes(order='C')).hexdigest() != x['sha256']:
                    raise ValueError('array byte digest mismatch')
                return a.copy()
            if '__float__' in x:
                return float.fromhex(x['__float__'])
            if '__complex__' in x:
                return complex(*[dec(v) for v in x['__complex__']])
            if '__dict__' in x:
                return {k: dec(v) for k, v in x['__dict__']}
            if '__sequence__' in x:
                seq = [dec(v) for v in x['__sequence__']]
                return tuple(seq) if x['tuple'] else seq
            raise ValueError('unknown transport tag')
        return dec(meta['payload'])


def minimal_opf_return(raw: Mapping[str, Any]) -> dict:
    """Snapshot only named numeric OPF fields, even on a solver failure.

    This is NOT the full Python return. Inputs, tolerances, reference and noise
    identity belong in the caller's separate request/checkpoint metadata.
    Solver statuses are preserved, never inferred from a finite voltage vector.
    """
    fields = ('version', 'baseMVA', 'bus', 'branch', 'gen', 'gencost', 'f', 'success', 'et')
    payload = {}
    for k in fields:
        if k in raw:
            v = raw[k]
            payload[k] = v.copy() if isinstance(v, np.ndarray) else v
    return {'identity': 'MINIMAL_NUMERIC_OPF_RETURN_NOT_FULL_OBJECT',
            'numeric': payload,
            'not_saved_fields': sorted(str(k) for k in raw if k not in fields)}


def content_key(context: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    """Exact cache identity; the caller must supply COMPLETE stage dependencies.

    Include code/config/environment, input and network identities, action/state
    history, clock and RNG identity as applicable. No approximate input rounding.
    A matching key says inputs match, not that a previous result was successful.
    """
    descriptor = {'context': dict(context), 'arrays': {}}
    for name, a in sorted(arrays.items()):
        a = np.asarray(a)
        if a.dtype.hasobject:
            raise TypeError('object array in cache key')
        descriptor['arrays'][name] = {'dtype': a.dtype.str, 'shape': list(a.shape),
                       'sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()}
    data = json.dumps(descriptor, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def original_clean_windows(saved_ordinals: set[int]) -> list[dict]:
    """The FOUR original P05B windows, never replacement windows."""
    out = []
    for clip in (0, 1):
        for end in (6, 7):
            ids = list(range(clip * 8 + end - 5, clip * 8 + end + 1))
            out.append({'clip': clip, 'end_local': end, 'ordinals': ids,
                        'end_source_row': 28228 + clip * 8 + end,
                        'available': all(i in saved_ordinals for i in ids)})
    return out


def select_nonempty_dev(arrivals: np.ndarray, values: np.ndarray, gate: float,
                        *, total_steps: int, width: int = 32, lead: int = 10) -> dict:
    """Select exposed development input by arrivals/forecasts, never outcomes.

    Start ten steps before the first arrival meeting the fixed arrival gate.
    Do not use this to select a confirmatory test set.
    """
    arrivals, values = np.asarray(arrivals), np.asarray(values)
    if arrivals.ndim != 1 or arrivals.shape != values.shape:
        raise ValueError('bad columns')
    if width <= 0 or lead < 0 or width > total_steps or not math.isfinite(gate):
        raise ValueError('bad selection settings')
    if not np.issubdtype(arrivals.dtype, np.integer) or np.any(arrivals < 0) or np.any(arrivals >= total_steps):
        raise ValueError('invalid arrival index')
    if not np.isfinite(values).all():
        raise ValueError('invalid prediction')
    accepted = np.flatnonzero(values >= gate)
    if not len(accepted):
        raise ValueError('no gate-eligible development job; do not silently change gate')
    first = int(arrivals[accepted].min())
    start = min(max(first - lead, 0), total_steps - width)
    selected = (arrivals >= start) & (arrivals < start + width)
    return {'role': 'EXPOSED_NONEMPTY_INTEGRATION_NOT_CONFIRMATION',
            'start': start, 'end_exclusive': start + width,
            'candidate_count': int(selected.sum()),
            'gate_accepted_count': int((selected & (values >= gate)).sum()),
            'selection_uses_labels_outcomes_or_actual_durations': False}

