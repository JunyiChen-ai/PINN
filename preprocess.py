import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import math
import warnings
from typing import Dict, List, Optional, Tuple, Any


def _read_shared_strings(z: zipfile.ZipFile) -> List[str]:
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    sst = ET.fromstring(z.read('xl/sharedStrings.xml'))
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    out: List[str] = []
    for si in sst.findall(f'.//{ns}si'):
        text = ''.join(t.text or '' for t in si.findall(f'.//{ns}t'))
        out.append(text)
    return out


def _workbook_sheet_map(z: zipfile.ZipFile) -> Dict[str, str]:
    wb = ET.fromstring(z.read('xl/workbook.xml'))
    ns = {
        'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }
    rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    rid_to_target = {rel.attrib['Id']: rel.attrib['Target'] for rel in rels}
    mapping: Dict[str, str] = {}
    for sh in wb.find('a:sheets', ns):
        name = sh.attrib.get('name')
        rid = sh.attrib.get('{%s}id' % ns['r'])
        target = rid_to_target.get(rid)
        if target:
            mapping[name] = 'xl/' + target if not target.startswith('xl/') else target
    return mapping


def _parse_sheet_rows(z: zipfile.ZipFile, sheet_path: str, shared: List[str]) -> List[List[Optional[str]]]:
    xml = ET.fromstring(z.read(sheet_path))
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    sdata = xml.find(f'.//{ns}sheetData')
    if sdata is None:
        return []

    def col_to_index(rref: str) -> int:
        col = ''.join(ch for ch in rref if ch.isalpha())
        i = 0
        for ch in col:
            i = i * 26 + (ord(ch.upper()) - 64)
        return i - 1

    rows: List[List[Optional[str]]] = []
    for r in sdata.findall(f'{ns}row'):
        cells: Dict[int, Optional[str]] = {}
        max_ci = -1
        for c in r.findall(f'{ns}c'):
            rref = c.attrib.get('r', 'A1')
            ci = col_to_index(rref)
            if ci > max_ci:
                max_ci = ci
            t = c.attrib.get('t')
            v = c.find(f'{ns}v')
            val: Optional[str] = None
            if v is not None and v.text is not None:
                if t == 's':
                    idx = int(v.text)
                    val = shared[idx] if 0 <= idx < len(shared) else None
                else:
                    val = v.text
            is_elem = c.find(f'{ns}is')
            if t == 'inlineStr' and is_elem is not None:
                tnodes = is_elem.findall(f'.//{ns}t')
                val = ''.join(tn.text or '' for tn in tnodes)
            cells[ci] = val
        row_vals = [cells.get(i) for i in range(max_ci + 1)] if max_ci >= 0 else []
        rows.append(row_vals)
    return rows


def load_experiments(excel_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Load experiment-level metadata from 'Experiments' sheet.
    Returns mapping: exp_id -> dict of fields.
    """
    p = Path(excel_path)
    with zipfile.ZipFile(p, 'r') as z:
        shared = _read_shared_strings(z)
        sheets = _workbook_sheet_map(z)
        if 'Experiments' not in sheets:
            return {}
        rows = _parse_sheet_rows(z, sheets['Experiments'], shared)
        header_idx = None
        for i, r in enumerate(rows[:10]):
            if r and r[0] == 'Experiment number':
                header_idx = i
                break
        if header_idx is None:
            return {}
        header = rows[header_idx]
        cols = {name: idx for idx, name in enumerate(header) if name}
        out: Dict[int, Dict[str, Any]] = {}
        for r in rows[header_idx + 2 :]:
            if not r or 'Experiment number' not in cols:
                continue
            v = r[cols['Experiment number']] if cols['Experiment number'] < len(r) else None
            if v in (None, ''):
                continue
            try:
                exp = int(float(v))
            except Exception:
                continue
            rec: Dict[str, Any] = {}
            for k, idx in cols.items():
                if idx < len(r):
                    rec[k] = r[idx]
            out[exp] = rec
        return out


def load_timeseries(excel_path: str) -> Dict[int, Dict[str, List[float]]]:
    """
    Load per-experiment time series from 'Data' sheet.
    Returns mapping: exp_id -> { 'Time': [...], 'HD1': [...], 'HD2': [...], 'HD3': [...] }
    """
    p = Path(excel_path)
    with zipfile.ZipFile(p, 'r') as z:
        shared = _read_shared_strings(z)
        sheets = _workbook_sheet_map(z)
        if 'Data' not in sheets:
            return {}
        rows = _parse_sheet_rows(z, sheets['Data'], shared)
        header_idx = None
        for i, r in enumerate(rows[:10]):
            if r and r[0] == 'Experiment':
                header_idx = i
                break
        if header_idx is None:
            return {}
        header = rows[header_idx]
        cols = {name: idx for idx, name in enumerate(header) if name}
        required = ['Experiment', 'Time', 'HD1 Temperature', 'HD2 Temperature', 'HD3 Temperature']
        for req in required:
            if req not in cols:
                raise ValueError(f"Missing column in Data sheet: {req}")
        out: Dict[int, Dict[str, List[float]]] = {}
        nonfinite_rows = 0
        for r in rows[header_idx + 2 :]:
            if not r:
                continue
            vexp = r[cols['Experiment']] if cols['Experiment'] < len(r) else None
            if vexp in (None, ''):
                continue
            try:
                exp = int(float(vexp))
            except Exception:
                continue
            t = r[cols['Time']] if cols['Time'] < len(r) else None
            h1 = r[cols['HD1 Temperature']] if cols['HD1 Temperature'] < len(r) else None
            h2 = r[cols['HD2 Temperature']] if cols['HD2 Temperature'] < len(r) else None
            h3 = r[cols['HD3 Temperature']] if cols['HD3 Temperature'] < len(r) else None
            try:
                tf = float(t) if t not in (None, '') else None
                v1 = float(h1) if h1 not in (None, '') else None
                v2 = float(h2) if h2 not in (None, '') else None
                v3 = float(h3) if h3 not in (None, '') else None
            except Exception:
                nonfinite_rows += 1
                continue
            if tf is None or v1 is None or v2 is None or v3 is None:
                continue
            if not (math.isfinite(tf) and math.isfinite(v1) and math.isfinite(v2) and math.isfinite(v3)):
                nonfinite_rows += 1
                continue
            rec = out.setdefault(exp, {'Time': [], 'HD1': [], 'HD2': [], 'HD3': []})
            rec['Time'].append(tf)
            rec['HD1'].append(v1)
            rec['HD2'].append(v2)
            rec['HD3'].append(v3)
        # sort by time
        for rec in out.values():
            zipped = list(zip(rec['Time'], rec['HD1'], rec['HD2'], rec['HD3']))
            zipped.sort(key=lambda x: x[0])
            t, a, b, c = zip(*zipped)
            rec['Time'], rec['HD1'], rec['HD2'], rec['HD3'] = list(t), list(a), list(b), list(c)
        if nonfinite_rows:
            warnings.warn(
                f'load_timeseries: skipped {nonfinite_rows} rows with non-finite values (NaN/Inf).',
                RuntimeWarning,
            )
        return out


def find_failure_time(values: List[float], times: List[float], threshold: float) -> Optional[float]:
    for v, t in zip(values, times):
        if v >= threshold:
            return t
    return None


def build_window_samples(
    series: Dict[int, Dict[str, List[float]]],
    history: int,
    horizon: int,
    stride: int = 1,
    use_channels: Tuple[str, ...] = ('HD1', 'HD2', 'HD3'),
    target_channel: str = 'HD1',
    trunc_temp: Optional[float] = None,
    drop_after_fail: bool = True,
    stop_when_all_channels_reach_trunc: bool = False,
) -> Tuple[List[List[List[float]]], List[List[float]], List[int]]:
    """
    Create sliding-window samples across experiments.
    Returns (X, Y, exp_ids):
      X: list of [T, D] windows
      Y: list of [H] targets of target_channel
      exp_ids: parallel list of experiment ids for each sample

    If trunc_temp is provided and drop_after_fail is True, any window whose
    input period touches t >= first failure time for any used channel will be dropped.
    Targets are always ground-truth (not truncated).
    """
    X: List[List[List[float]]] = []
    Y: List[List[float]] = []
    E: List[int] = []

    for exp, rec in series.items():
        T = len(rec['Time'])
        if T < history + horizon:
            continue
        # Pre-compute failure times per channel if needed
        fail_times: Dict[str, Optional[float]] = {}
        if trunc_temp is not None:
            for ch in use_channels:
                fail_times[ch] = find_failure_time(rec[ch], rec['Time'], trunc_temp)
        else:
            for ch in use_channels:
                fail_times[ch] = None

        # Optionally compute a global stop index where all channels are >= trunc_temp
        stop_idx: Optional[int] = None
        if trunc_temp is not None and stop_when_all_channels_reach_trunc:
            for j in range(T):
                if all(rec[ch][j] >= trunc_temp for ch in use_channels):
                    stop_idx = j
                    break

        i = 0
        while i + history + horizon <= T:
            start = i
            end = i + history  # exclusive
            # If requested, stop constructing windows once all channels have reached trunc at some time
            if stop_idx is not None:
                # Do not use windows whose input touches or passes stop, or whose targets go beyond stop
                last_in = end - 1
                last_out = end + horizon - 1
                if last_in >= stop_idx or last_out > stop_idx:
                    # Move to next position; if even the earliest window violates, we can break to speed up
                    i += stride
                    # If our start is already beyond or at stop, no further windows will be valid
                    if start >= stop_idx:
                        break
                    continue

            if trunc_temp is not None and drop_after_fail:
                invalid = False
                t_end = rec['Time'][end - 1]
                for ch in use_channels:
                    ft = fail_times.get(ch)
                    if ft is not None and t_end >= ft:
                        invalid = True
                        break
                if invalid:
                    i += stride
                    continue
            xw: List[List[float]] = []
            for j in range(start, end):
                row = []
                for ch in use_channels:
                    v = rec[ch][j]
                    if trunc_temp is not None:
                        # Clamp input values to trunc_temp to reflect sensor saturation
                        v = min(v, trunc_temp)
                    row.append(v)
                xw.append(row)
            yw: List[float] = []
            for j in range(end, end + horizon):
                yw.append(rec[target_channel][j])
            X.append(xw)
            Y.append(yw)
            E.append(exp)
            i += stride
    return X, Y, E


__all__ = [
    'load_experiments',
    'load_timeseries',
    'build_window_samples',
]
