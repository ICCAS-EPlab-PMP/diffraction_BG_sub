#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_window.py — 基于 silx 的 BGsub 主窗口
完整实现 substract_bg_gui.py 的所有功能
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import zipfile
from contextlib import nullcontext
from datetime import datetime
from difflib import SequenceMatcher
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# try:
from silx.gui import qt
from silx.gui.plot.StackView import StackViewMainWindow
from silx.gui.plot.ImageView import ImageView
from silx.gui.plot.CompareImages import CompareImages
from silx.gui.colors import Colormap
from silx.gui.dialog.ColormapDialog import ColormapDialog
from silx.gui.qt import QIcon

HAS_SILX = True
# except ImportError:
#     HAS_SILX = True

import h5py
import pandas as pd

import fabio

HAS_FABIO = True
import sys
import os

from BGsub.io.image_io import (
    CacheDir,
    encode_image_to_bytes,
    export_tiff_to_h5,
    load_image_file,
    load_h5_stack,
    probe_image_file,
    probe_h5_datasets,
    TIFF_EXTS,
    H5_EXTS,
)

# === Phase 1: 匹配结果数据结构 ===
from dataclasses import dataclass, field


@dataclass
class MatchCandidate:
    """单个候选电离室文件的匹配信息。"""

    path: str
    strategy: str
    score: float
    state_sim: float = 0.0
    base_match: bool = False
    state_match: bool = False
    num_match: bool = False


@dataclass
class MatchResult:
    """单个 TIFF 文件的完整匹配结果。"""

    tiff_name: str
    is_bg: bool
    matched_path: Optional[str] = None
    strategy: Optional[str] = None
    score: float = 0.0
    state_sim: float = 0.0
    transmission: Optional[float] = None
    ion_intensity: Optional[float] = None
    bg_intensity: Optional[float] = None
    success: bool = False
    error_msg: Optional[str] = None
    candidates: List[MatchCandidate] = field(default_factory=list)


@dataclass
class MatchStatistics:
    total: int
    passed: List[MatchResult] = field(default_factory=list)
    below_threshold: List[MatchResult] = field(default_factory=list)
    failed: List[MatchResult] = field(default_factory=list)
    processed: List[MatchResult] = field(default_factory=list)
    worst_case: Optional[MatchResult] = None
    worst_case_kind: str = "none"


def _ensure_float32_frame(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[0]
    if arr.ndim == 4:
        return arr[0, 0]
    raise ValueError(f"不支持的帧维度: {arr.ndim}")


class LazyFrameSource:
    def __init__(
        self,
        frame_count: int,
        loader: Callable[[int], np.ndarray],
        names: Optional[List[str]] = None,
    ):
        self._frame_count = max(0, int(frame_count))
        self._loader = loader
        self._names = names or [f"frame_{i}" for i in range(self._frame_count)]
        self._cached_index: Optional[int] = None
        self._cached_frame: Optional[np.ndarray] = None

    @classmethod
    def from_array(cls, data: np.ndarray, names: Optional[List[str]] = None) -> "LazyFrameSource":
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim == 4:
            arr = arr[:, 0, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"Stack data must be 2D/3D/4D, got {arr.ndim}D")
        return cls(arr.shape[0], lambda index: arr[index], names)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_name(self, index: int) -> str:
        if 0 <= index < len(self._names):
            return self._names[index]
        return f"frame_{index}"

    def get_frame(self, index: int) -> np.ndarray:
        if not 0 <= index < self._frame_count:
            raise IndexError(index)
        if self._cached_index == index and self._cached_frame is not None:
            return self._cached_frame
        frame = _ensure_float32_frame(self._loader(index))
        self._cached_index = index
        self._cached_frame = frame
        return frame

    def clear_cache(self) -> None:
        self._cached_index = None
        self._cached_frame = None


def get_resource_path(relative_path):
    """获取资源的绝对路径，兼容打包前和打包后的情况"""
    if getattr(sys, "frozen", False):
        # 如果是被打包成了 exe，去临时解压目录 _MEIPASS 找
        base_path = sys._MEIPASS
    else:
        # 如果是直接通过 python 运行，就在当前目录找
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def load_ionchamber(path: str) -> Optional[pd.DataFrame]:
    """加载电离室文件。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        lines = [ln for ln in content.strip().split("\n") if not ln.startswith("#")]
        data = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                data.append([" ".join(parts[:2])] + [float(x) for x in parts[2:5]])
        if not data:
            return None
        return pd.DataFrame(data, columns=["Time", "Ionchamber0", "Ionchamber1", "Ionchamber2"])
    except Exception:
        return None


def calc_intensity(df: pd.DataFrame, ch: str, method: str) -> Optional[float]:
    """计算电离室强度。"""
    try:
        v = df[ch].values
        if method == "median":
            return float(np.median(v))
        elif method == "trimmed_mean":
            s = np.sort(v)
            return float(np.mean(s[1:-1])) if len(s) > 2 else float(np.mean(s))
        else:
            return float(np.mean(v))
    except Exception:
        return None


def _split_name(stem: str) -> Tuple[str, Optional[str], Optional[int]]:
    """解析文件名，提取主名字、状态段、末尾编号。"""
    normalized = stem.replace("_", "-")
    parts = [p for p in normalized.split("-") if p]

    if not parts:
        return stem.lower(), None, None

    base = parts[0].lower()

    if len(parts) >= 2 and re.fullmatch(r"\d+", parts[-1]):
        num = int(parts[-1])
        middle = parts[1:-1]
    else:
        num = None
        middle = parts[1:]

    state = "-".join(p.lower() for p in middle) if middle else None
    return base, state, num


def _state_similarity(s1: Optional[str], s2: Optional[str]) -> float:
    """计算两个state字符串的相似度。"""
    if s1 is None and s2 is None:
        return 1.0
    if s1 is None or s2 is None:
        return 0.0
    return SequenceMatcher(None, s1, s2).ratio()


def _pick_best_state_match(
    candidates: List[str],
    ion_info: List[Tuple[str, str, Optional[str], Optional[int]]],
    target_state: Optional[str],
) -> Optional[str]:
    """从候选ion chamber中选择state最匹配的一个。"""
    if not candidates or target_state is None:
        return None
    info_map = {p: (ib, ist, inum) for p, ib, ist, inum in ion_info}
    best_p, best_s = None, -1.0
    for p in candidates:
        _, ist, _ = info_map.get(p, (None, None, None))
        sim = _state_similarity(target_state, ist)
        if sim > best_s:
            best_s, best_p = sim, p
    return best_p if best_s > 0 else None


def match_ionchamber(
    tiff_name: str,
    ion_paths: List[str],
    user_regex: Optional[str] = None,
) -> Optional[str]:
    """将 TIFF/EDF 文件与电离室文件进行匹配（返回路径）。"""
    p, _, _, _, _ = match_ionchamber_detail(tiff_name, ion_paths, user_regex)
    return p


# === Phase 1: 结构化匹配结果返回 ===
STRATEGY_EXACT = "exact"
STRATEGY_REGEX = "regex"
STRATEGY_BASE_STATE = "base_state_exact"
STRATEGY_BASE_STATE_MINNUM = "base_state_minnum"
STRATEGY_BASE_NUM_BEST_STATE = "base_num_best_state"
STRATEGY_BASE_ONLY_BEST_STATE = "base_only_best_state"
STRATEGY_FUZZY = "fuzzy"
DEFAULT_MATCH_TOP_K = 5

_STRATEGY_PRIORITY = {
    STRATEGY_REGEX: 0,
    STRATEGY_EXACT: 1,
    STRATEGY_BASE_STATE: 2,
    STRATEGY_BASE_STATE_MINNUM: 3,
    STRATEGY_BASE_NUM_BEST_STATE: 4,
    STRATEGY_BASE_ONLY_BEST_STATE: 5,
    STRATEGY_FUZZY: 6,
    "none": 999,
}


def _match_score(result: MatchResult) -> float:
    return result.score if result.score > 0 else (result.state_sim or 0.0)


def _match_status(result: MatchResult, threshold: float) -> str:
    if not result.success:
        return "failed"
    if _match_score(result) < threshold:
        return "below_threshold"
    return "passed"


def _worst_case_sort_key(result: MatchResult) -> Tuple[float, float, str]:
    return (_match_score(result), result.state_sim or 0.0, result.tiff_name.lower())


def _select_worst_case_result(
    match_results: List[MatchResult], threshold: float
) -> Tuple[Optional[MatchResult], str]:
    failed = sorted(
        [result for result in match_results if _match_status(result, threshold) == "failed"],
        key=_worst_case_sort_key,
    )
    if failed:
        return failed[0], "failed"

    below = sorted(
        [
            result
            for result in match_results
            if _match_status(result, threshold) == "below_threshold"
        ],
        key=_worst_case_sort_key,
    )
    if below:
        return below[0], "below_threshold"

    passed = sorted(
        [result for result in match_results if _match_status(result, threshold) == "passed"],
        key=_worst_case_sort_key,
    )
    if passed:
        return passed[0], "lowest_passing"

    return None, "none"


def _collect_match_statistics(
    match_results: List[MatchResult],
    threshold: float,
    trans_dict: Optional[Dict[str, float]] = None,
) -> MatchStatistics:
    processed_names = set(trans_dict.keys()) if trans_dict else set()
    stats = MatchStatistics(total=len(match_results))

    for result in match_results:
        status = _match_status(result, threshold)
        if status == "passed":
            stats.passed.append(result)
        elif status == "below_threshold":
            stats.below_threshold.append(result)
        else:
            stats.failed.append(result)

        if result.tiff_name in processed_names:
            stats.processed.append(result)

    stats.worst_case, stats.worst_case_kind = _select_worst_case_result(match_results, threshold)
    return stats


def _build_match_candidate(
    tiff_base: str,
    tiff_state: Optional[str],
    tiff_num: Optional[int],
    path: str,
    ion_base: str,
    ion_state: Optional[str],
    ion_num: Optional[int],
    strategy: str,
    score: float,
) -> MatchCandidate:
    return MatchCandidate(
        path=path,
        strategy=strategy,
        score=max(0.0, min(score, 1.0)),
        state_sim=_state_similarity(tiff_state, ion_state),
        base_match=ion_base == tiff_base,
        state_match=tiff_state is not None and ion_state == tiff_state,
        num_match=tiff_num is not None and ion_num == tiff_num,
    )


def _candidate_sort_key(
    candidate: MatchCandidate,
) -> Tuple[float, float, float, float, float, int, str]:
    return (
        -candidate.score,
        -(candidate.state_sim or 0.0),
        -float(candidate.base_match),
        -float(candidate.state_match),
        -float(candidate.num_match),
        _STRATEGY_PRIORITY.get(candidate.strategy, 999),
        os.path.basename(candidate.path).lower(),
    )


def _trim_match_candidates(
    candidates: List[MatchCandidate], top_k: int = DEFAULT_MATCH_TOP_K
) -> List[MatchCandidate]:
    best_by_path: Dict[str, MatchCandidate] = {}
    for candidate in candidates:
        existing = best_by_path.get(candidate.path)
        if existing is None or _candidate_sort_key(candidate) < _candidate_sort_key(existing):
            best_by_path[candidate.path] = candidate
    return sorted(best_by_path.values(), key=_candidate_sort_key)[:top_k]


def _collect_match_candidates(
    tiff_name: str,
    ion_paths: List[str],
    user_regex: Optional[str] = None,
    top_k: int = DEFAULT_MATCH_TOP_K,
) -> List[MatchCandidate]:
    tiff_stem = os.path.splitext(tiff_name)[0]
    tiff_base, tiff_state, tiff_num = _split_name(tiff_stem)

    ion_info: List[Tuple[str, str, Optional[str], Optional[int]]] = []
    for path in ion_paths:
        ion_stem = os.path.splitext(os.path.basename(path))[0]
        ion_info.append((path, *_split_name(ion_stem)))

    candidates: List[MatchCandidate] = []

    def add_candidate(
        path: str,
        ion_base: str,
        ion_state: Optional[str],
        ion_num: Optional[int],
        strategy: str,
        score: float,
    ) -> None:
        candidates.append(
            _build_match_candidate(
                tiff_base,
                tiff_state,
                tiff_num,
                path,
                ion_base,
                ion_state,
                ion_num,
                strategy,
                score,
            )
        )

    if user_regex:
        try:
            pattern = re.compile(user_regex)
            for path, ion_base, ion_state, ion_num in ion_info:
                if pattern.search(ion_base):
                    add_candidate(path, ion_base, ion_state, ion_num, STRATEGY_REGEX, 1.0)
        except re.error:
            pass

    if tiff_state is not None and tiff_num is not None:
        for path, ion_base, ion_state, ion_num in ion_info:
            if ion_base == tiff_base and ion_state == tiff_state and ion_num == tiff_num:
                add_candidate(path, ion_base, ion_state, ion_num, STRATEGY_EXACT, 1.0)

    same_base_state = [
        (path, ion_base, ion_state, ion_num)
        for path, ion_base, ion_state, ion_num in ion_info
        if ion_base == tiff_base and ion_state == tiff_state
    ]
    if tiff_state is not None and same_base_state:
        base_state_score = 1.0 if len(same_base_state) == 1 else 0.95
        for path, ion_base, ion_state, ion_num in same_base_state:
            add_candidate(path, ion_base, ion_state, ion_num, STRATEGY_BASE_STATE, base_state_score)
        numbered = [entry for entry in same_base_state if entry[3] is not None]
        if len(same_base_state) > 1 and numbered:
            add_candidate(
                *min(numbered, key=lambda entry: entry[3]), STRATEGY_BASE_STATE_MINNUM, 0.95
            )

    if tiff_num is not None:
        same_base_num = [
            (path, ion_base, ion_state, ion_num)
            for path, ion_base, ion_state, ion_num in ion_info
            if ion_base == tiff_base and ion_num == tiff_num
        ]
        if same_base_num:
            if len(same_base_num) == 1:
                add_candidate(*same_base_num[0], STRATEGY_BASE_NUM_BEST_STATE, 0.85)
            else:
                for path, ion_base, ion_state, ion_num in same_base_num:
                    sim = _state_similarity(tiff_state, ion_state)
                    score = 0.80 + sim * 0.2 if sim >= 0.6 else 0.70
                    add_candidate(
                        path,
                        ion_base,
                        ion_state,
                        ion_num,
                        STRATEGY_BASE_NUM_BEST_STATE,
                        score,
                    )
                for path, ion_base, ion_state, ion_num in ion_info:
                    if ion_base == tiff_base and ion_num is None:
                        add_candidate(
                            path,
                            ion_base,
                            ion_state,
                            ion_num,
                            STRATEGY_BASE_NUM_BEST_STATE,
                            _state_similarity(tiff_state, ion_state) * 0.9,
                        )

    same_base = [
        (path, ion_base, ion_state, ion_num)
        for path, ion_base, ion_state, ion_num in ion_info
        if ion_base == tiff_base
    ]
    if same_base:
        if len(same_base) == 1:
            add_candidate(*same_base[0], STRATEGY_BASE_ONLY_BEST_STATE, 0.75)
        else:
            numbered = [entry for entry in same_base if entry[3] is not None]
            source = numbered if numbered else same_base
            default_score = 0.70 if numbered else 0.65
            for path, ion_base, ion_state, ion_num in source:
                sim = _state_similarity(tiff_state, ion_state)
                add_candidate(
                    path,
                    ion_base,
                    ion_state,
                    ion_num,
                    STRATEGY_BASE_ONLY_BEST_STATE,
                    default_score if sim > 0 else default_score - 0.05,
                )

    best_fuzzy_score = 0.0
    best_fuzzy_entry: Optional[Tuple[str, str, Optional[str], Optional[int]]] = None
    for path, ion_base, ion_state, ion_num in ion_info:
        score = SequenceMatcher(None, tiff_base, ion_base).ratio()
        if ion_state is not None and tiff_state is not None and ion_state == tiff_state:
            score += 0.25
        if ion_num is not None and tiff_num is not None and ion_num == tiff_num:
            score += 0.25
        if ion_base.startswith(tiff_base) or tiff_base.startswith(ion_base):
            score += 0.1
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_fuzzy_entry = (path, ion_base, ion_state, ion_num)
    if best_fuzzy_entry is not None and best_fuzzy_score >= 0.4:
        add_candidate(*best_fuzzy_entry, STRATEGY_FUZZY, best_fuzzy_score)

    return _trim_match_candidates(candidates, top_k=top_k)


def match_ionchamber_detail(
    tiff_name: str,
    ion_paths: List[str],
    user_regex: Optional[str] = None,
    top_k: int = DEFAULT_MATCH_TOP_K,
) -> Tuple[Optional[str], str, float, float, List[MatchCandidate]]:
    """
    将 TIFF/EDF 文件与电离室文件进行匹配，返回结构化结果。

    Returns:
        (matched_path, strategy_name, composite_score, state_sim, candidates)
        - matched_path: 最佳匹配路径，无匹配时 None
        - strategy_name: 匹配策略标识符
        - composite_score: 综合评分（0.0~1.0）
        - state_sim: state 部分相似度（0.0~1.0）
        - candidates: 已按优先级裁剪的 top-k 结构化候选
    """
    candidates = _collect_match_candidates(tiff_name, ion_paths, user_regex, top_k=top_k)
    if not candidates:
        return None, "none", 0.0, 0.0, []

    top = candidates[0]
    return top.path, top.strategy, top.score, top.state_sim, candidates


# === Phase 1: Token 级差异高亮 ===
def _token_diff_html(tiff_stem: str, ion_stem: str) -> str:
    """
    对比 tiff 文件名与 ion 文件名 stem，生成带颜色的 HTML 差异文本。
    相同 token 绿色，差异 token 红色，ion 特有部分橙色。
    """
    tiff_parts = [p for p in re.split(r"[\-_]", tiff_stem) if p]
    ion_parts = [p for p in re.split(r"[\-_]", ion_stem) if p]

    matcher = SequenceMatcher(None, tiff_parts, ion_parts)
    tiff_html = []
    ion_html = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for p in tiff_parts[i1:i2]:
                tiff_html.append(f'<span style="color:#4c4">{p}</span>')
            for p in ion_parts[j1:j2]:
                ion_html.append(f'<span style="color:#4c4">{p}</span>')
        elif tag == "replace":
            for p in tiff_parts[i1:i2]:
                tiff_html.append(f'<span style="color:#f66">{p}</span>')
            for p in ion_parts[j1:j2]:
                ion_html.append(f'<span style="color:#f66">{p}</span>')
        elif tag == "delete":
            for p in tiff_parts[i1:i2]:
                tiff_html.append(f'<span style="color:#f96">{p}</span>')
        elif tag == "insert":
            for p in ion_parts[j1:j2]:
                ion_html.append(f'<span style="color:#e07b39">{p}</span>')

    tiff_label = '<span style="color:#aaa">sample:</span> '
    ion_label = '<span style="color:#aaa">ion:</span> '
    return f"{tiff_label}{''.join(tiff_html)}<br/>{ion_label}{''.join(ion_html)}"


# === Phase 1: 匹配结果导出功能 ===
def _build_match_summary_dict(
    bg_result: Optional[MatchResult],
    match_results: List[MatchResult],
    threshold: float,
    trans_dict: Optional[Dict[str, float]] = None,
) -> dict:
    """
    构建供导出的结构化摘要字典。

    Args:
        trans_dict: 已进入处理的样品名字典（key=样品名，value=transmission），
                    用于标记 processed 字段。
    """
    processed_names = set(trans_dict.keys()) if trans_dict else set()
    stats = _collect_match_statistics(match_results, threshold, trans_dict)
    rows = []
    for mr in match_results:
        score = _match_score(mr)
        status = _match_status(mr, threshold)
        rows.append(
            {
                "tiff_name": mr.tiff_name,
                "is_bg": mr.is_bg,
                "matched_ion": os.path.basename(mr.matched_path) if mr.matched_path else "",
                "strategy": mr.strategy or "",
                "score": round(score, 4),
                "state_sim": round(mr.state_sim, 4) if mr.state_sim else 0.0,
                "transmission": round(mr.transmission, 4) if mr.transmission else None,
                "above_threshold": status == "passed",
                "success": mr.success,
                "processed": mr.tiff_name in processed_names,
                "error_msg": mr.error_msg or "",
            }
        )

    bg_row = None
    if bg_result:
        bg_score = _match_score(bg_result)
        bg_row = {
            "tiff_name": bg_result.tiff_name,
            "is_bg": True,
            "matched_ion": os.path.basename(bg_result.matched_path)
            if bg_result.matched_path
            else "",
            "strategy": bg_result.strategy or "",
            "score": round(bg_score, 4),
            "state_sim": round(bg_result.state_sim, 4) if bg_result.state_sim else 0.0,
            "transmission": None,
            "above_threshold": True,
            "success": bg_result.success,
            "processed": False,
            "error_msg": bg_result.error_msg or "",
        }

    return {
        "threshold": threshold,
        "total_samples": stats.total,
        "total_above_threshold": len(stats.passed),
        "total_below_threshold": len(stats.below_threshold),
        "total_failed": len(stats.failed),
        "total_processed": len(stats.processed),
        "bg_match": bg_row,
        "sample_matches": rows,
    }


def _build_match_explain_html(stats: MatchStatistics, threshold: float) -> str:
    if stats.total == 0:
        return '<span style="color:#aaa;font-size:12px">暂无匹配结果</span>'

    if stats.worst_case is None:
        return (
            f'<span style="color:#4c4;font-size:13px">✓ 全部 {len(stats.passed)}/{stats.total} '
            f"个匹配均高于阈值 {threshold:.2f}</span>"
        )

    result = stats.worst_case
    candidate = result.candidates[0] if result.candidates else None
    ion_name = os.path.basename(candidate.path if candidate else (result.matched_path or ""))
    ion_stem = os.path.splitext(ion_name)[0] if ion_name else ""
    diff_html = (
        _token_diff_html(os.path.splitext(result.tiff_name)[0], ion_stem)
        if ion_stem
        else "<i>无候选文件</i>"
    )
    summary = f"通过 {len(stats.passed)} 个，低于阈值 {len(stats.below_threshold)} 个，失败 {len(stats.failed)} 个"
    score = _match_score(result)

    if stats.worst_case_kind == "failed":
        return (
            f'<span style="color:#f66;font-size:12px">✗ 最差案例=失败 '
            f"(score={score:.3f})，{summary}：</span><br/>"
            f'<span style="color:#f99">{result.error_msg or "无匹配"}</span><br/>{diff_html}'
        )

    if stats.worst_case_kind == "below_threshold":
        return (
            f'<span style="color:#e07b39;font-size:12px">⚠ 最差案例=低于阈值 '
            f"(score={score:.3f} &lt; {threshold:.2f})，{summary}：</span><br/>{diff_html}"
        )

    return (
        f'<span style="color:#e07b39;font-size:12px">⚠ 最差案例=最低通过 '
        f"(score={score:.3f} ≥ {threshold:.2f})，{summary}：</span><br/>{diff_html}"
    )


def _export_match_results_csv(summary: dict) -> str:
    """生成匹配结果的 CSV 格式字符串。"""
    import csv
    import io

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "tiff_name",
            "is_bg",
            "matched_ion",
            "strategy",
            "score",
            "state_sim",
            "transmission",
            "above_threshold",
            "success",
            "processed",
            "error_msg",
        ],
    )
    writer.writeheader()
    if summary.get("bg_match"):
        writer.writerow(summary["bg_match"])
    for row in summary.get("sample_matches", []):
        writer.writerow(row)
    return output.getvalue()


def _export_match_results_json(summary: dict) -> str:
    """生成匹配结果的 JSON 格式字符串。"""
    import json

    return json.dumps(summary, indent=2, ensure_ascii=False)


def _export_match_results_txt(summary: dict) -> str:
    """生成匹配结果的人类可读文本摘要。"""
    lines = [
        "=" * 60,
        "电离室匹配结果摘要 / Ion Chamber Match Summary",
        "=" * 60,
        f"阈值 / Threshold: {summary['threshold']:.2f}",
        f"样品总数: {summary['total_samples']}",
        f"通过阈值: {summary['total_above_threshold']}",
        f"低于阈值: {summary['total_below_threshold']}",
        f"匹配失败: {summary['total_failed']}",
        f"实际处理: {summary['total_processed']}",
        "",
        "── 背景匹配 / Background ──",
    ]

    bg = summary.get("bg_match")
    if bg:
        lines.append(
            f"  [{bg['tiff_name']}] → [{bg['matched_ion']}] "
            f"[{bg['strategy']} score={bg['score']:.3f} sim={bg['state_sim']:.3f}]"
        )
    else:
        lines.append("  (无)")

    lines += ["", "── 样品匹配 / Samples ──"]
    for r in summary.get("sample_matches", []):
        if not r["success"]:
            status = "✗ fail"
        elif not r["above_threshold"]:
            status = f"⚠ <{summary['threshold']:.2f}"
        else:
            status = "✓ passed"
        proc_mark = (
            " [已处理]"
            if r.get("processed")
            else (" [已排除]" if r.get("success") and not r.get("above_threshold") else "")
        )
        trans_str = f" T={r['transmission']:.2f}%" if r["transmission"] else ""
        lines.append(
            f"  {status} [{r['tiff_name']}] → [{r['matched_ion']}] "
            f"[{r['strategy']} s={r['score']:.3f} sim={r['state_sim']:.3f}]{trans_str}{proc_mark}"
        )
        if r["error_msg"]:
            lines.append(f"       错误: {r['error_msg']}")

    lines.append("=" * 60)
    return "\n".join(lines)


class LogWidget(qt.QTextEdit):
    """日志文本框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(80)
        self.setTextColor(qt.QColor("#4A5368"))

    def append_log(self, msg: str, level: str = "info"):
        color_map = {
            "ok": qt.QColor("#1B6B3A"),
            "warn": qt.QColor("#A04A00"),
            "err": qt.QColor("#8B2020"),
            "info": qt.QColor("#1A4A8A"),
            "dim": qt.QColor("#9AA3B2"),
        }
        ts = datetime.now().strftime("%H:%M:%S")
        self.setTextColor(color_map.get(level, qt.QColor("#4A5368")))
        self.append(f"[{ts}] {msg}")


@dataclass
class TiffWorkerResult:
    processed: List[Dict[str, Any]]
    processed_names: List[str]
    processed_count: int
    total_count: int
    cancelled: bool = False
    output_mode: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class H5WorkerResult:
    output_path: str
    processed_count: int
    total_count: int
    cancelled: bool = False


class BackgroundProcessingWorker(qt.QObject):
    progress = qt.Signal(int, str)
    log = qt.Signal(str, str)
    finished = qt.Signal(object)
    cancelled = qt.Signal(object)
    failed = qt.Signal(str)

    def __init__(self, job: Callable[["BackgroundProcessingWorker"], object], parent=None):
        super().__init__(parent)
        self._job = job
        self._cancel_event = threading.Event()

    @qt.Slot()
    def run(self):
        try:
            result = self._job(self)
            if getattr(result, "cancelled", False):
                self.cancelled.emit(result)
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))

    def request_cancel(self):
        self._cancel_event.set()

    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def emit_progress(self, value: int, text: str):
        self.progress.emit(value, text)

    def emit_log(self, msg: str, level: str = "info"):
        self.log.emit(msg, level)


def _process_tiff_sample(
    fd: Dict[str, Any],
    bg_data: np.ndarray,
    bg_shape: Tuple[int, int],
    transmission_percent: float,
) -> Tuple[np.ndarray, Dict[str, Any], bool]:
    samp_data, samp_hdr = load_image_file(fd["path"])
    if samp_data is None:
        raise ValueError("加载失败")

    sample = samp_data.astype(np.float32, copy=False)
    T = transmission_percent / 100.0
    bg_r, bg_c = bg_shape
    sr, sc = fd["shape"]
    cr, cc = min(bg_r, sr), min(bg_c, sc)
    proc = sample.copy()
    proc[:cr, :cc] = sample[:cr, :cc] / T - bg_data[:cr, :cc]
    hdr = (samp_hdr or {}).copy()
    hdr.update({"Processed": "BgSubtracted", "Transmission": f"{transmission_percent}%"})
    return proc.astype(np.float32, copy=False), hdr, (bg_r, bg_c) != (sr, sc)


def _run_tiff_h5_job(
    worker: BackgroundProcessingWorker,
    valid_samples: List[Dict[str, Any]],
    bg_data: np.ndarray,
    bg_shape: Tuple[int, int],
    transmissions: Dict[str, float],
) -> TiffWorkerResult:
    processed: List[Dict[str, Any]] = []
    processed_names: List[str] = []
    total = len(valid_samples)

    for i, fd in enumerate(valid_samples):
        if worker.is_cancelled():
            break
        try:
            proc, hdr, mismatch = _process_tiff_sample(
                fd, bg_data, bg_shape, transmissions[fd["name"]]
            )
            if mismatch:
                worker.emit_log(f"⚠ {fd['name']}: 尺寸不匹配，取公共区域", "warn")
            processed.append({"name": fd["name"], "data": proc, "header": hdr})
            processed_names.append(fd["name"])
        except Exception as exc:
            worker.emit_log(f"❌ {fd['name']}: {exc}", "err")

        pct = int((i + 1) / total * 100) if total else 100
        worker.emit_progress(pct, f"{i + 1}/{total}")

    return TiffWorkerResult(
        processed=processed,
        processed_names=processed_names,
        processed_count=len(processed_names),
        total_count=total,
        cancelled=worker.is_cancelled(),
    )


def _run_tiff_stream_job(
    worker: BackgroundProcessingWorker,
    valid_samples: List[Dict[str, Any]],
    bg_data: np.ndarray,
    bg_shape: Tuple[int, int],
    transmissions: Dict[str, float],
    fmt: str,
    output_mode: str,
    output_path: str,
) -> TiffWorkerResult:
    ext = fmt.lower()
    total = len(valid_samples)
    processed_names: List[str] = []
    needs_cache = ext in ("tif", "tiff")
    cache_ctx: Any = CacheDir() if needs_cache else nullcontext()

    with cache_ctx as cache_dir:
        if output_mode == "zip":
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for i, fd in enumerate(valid_samples):
                    if worker.is_cancelled():
                        break
                    try:
                        proc, hdr, mismatch = _process_tiff_sample(
                            fd, bg_data, bg_shape, transmissions[fd["name"]]
                        )
                        if mismatch:
                            worker.emit_log(f"⚠ {fd['name']}: 尺寸不匹配，取公共区域", "warn")

                        if ext == "edf":
                            edf_header = hdr.copy()
                            edf_header.update(
                                {
                                    "Dim_1": str(proc.shape[1]),
                                    "Dim_2": str(proc.shape[0]),
                                    "DataType": "Float32",
                                }
                            )
                            img_bytes = encode_image_to_bytes(proc, fmt="edf", header=edf_header)
                        else:
                            img_bytes = encode_image_to_bytes(
                                proc, fmt="tif", header=hdr, cache=cache_dir
                            )

                        zf.writestr(f"{os.path.splitext(fd['name'])[0]}.{ext}", img_bytes)
                        processed_names.append(fd["name"])
                    except Exception as exc:
                        worker.emit_log(f"❌ {fd['name']}: {exc}", "err")

                    pct = int((i + 1) / total * 100) if total else 100
                    worker.emit_progress(pct, f"{i + 1}/{total}")
        else:
            for i, fd in enumerate(valid_samples):
                if worker.is_cancelled():
                    break
                try:
                    proc, hdr, mismatch = _process_tiff_sample(
                        fd, bg_data, bg_shape, transmissions[fd["name"]]
                    )
                    if mismatch:
                        worker.emit_log(f"⚠ {fd['name']}: 尺寸不匹配，取公共区域", "warn")

                    out_name = f"{os.path.splitext(fd['name'])[0]}.{ext}"
                    out_file = os.path.join(output_path, out_name)
                    if ext == "edf":
                        edf_header = hdr.copy()
                        edf_header.update(
                            {
                                "Dim_1": str(proc.shape[1]),
                                "Dim_2": str(proc.shape[0]),
                                "DataType": "Float32",
                            }
                        )
                        fabio.edfimage.EdfImage(data=proc, header=edf_header).write(out_file)
                    else:
                        fabio.tifimage.TifImage(proc, hdr).write(out_file)
                    processed_names.append(fd["name"])
                except Exception as exc:
                    worker.emit_log(f"❌ {fd['name']}: {exc}", "err")

                pct = int((i + 1) / total * 100) if total else 100
                worker.emit_progress(pct, f"{i + 1}/{total}")

    return TiffWorkerResult(
        processed=[],
        processed_names=processed_names,
        processed_count=len(processed_names),
        total_count=total,
        cancelled=worker.is_cancelled(),
        output_mode=output_mode,
        output_path=output_path,
    )


def _run_h5_stream_job(
    worker: BackgroundProcessingWorker,
    sample_files: List[str],
    bg_path: str,
    output_path: str,
    unified_transmission: float,
) -> H5WorkerResult:
    bg_h5 = load_h5_stack(bg_path)
    if bg_h5 is None:
        raise ValueError(f"无法读取背景文件: {bg_path}")

    valid_files: List[Tuple[str, int]] = []
    for sample_path in sample_files:
        if worker.is_cancelled():
            break
        sample_meta = probe_h5_datasets(sample_path)
        if sample_meta is None:
            worker.emit_log(f"  加载失败: {sample_path}", "err")
            continue

        ss = sample_meta["shape"]
        bs = bg_h5["shape"]
        if ss[-2:] != bs[-2:]:
            worker.emit_log(f"  尺寸不匹配，跳过: {os.path.basename(sample_path)}", "warn")
            continue
        if sample_meta["effective_ndim"] != bg_h5["ndim"]:
            worker.emit_log(f"  维度不匹配，跳过: {os.path.basename(sample_path)}", "warn")
            continue

        N, Nb = sample_meta["n_frames"], bg_h5["n_frames"]
        uf = min(N, Nb) if (Nb != N and Nb != 1) else N
        valid_files.append((sample_path, uf))

    if not valid_files and not worker.is_cancelled():
        raise ValueError("没有可用的处理结果")
    if worker.is_cancelled() and not valid_files:
        return H5WorkerResult(
            output_path=output_path,
            processed_count=0,
            total_count=0,
            cancelled=True,
        )

    out_h5 = None
    processed_count = 0
    current_idx = 0
    try:
        out_h5 = h5py.File(output_path, "w")
        frame_shape = bg_h5["shape"][-2:]
        ndim = bg_h5["ndim"]
        max_shape = (None,) + frame_shape if ndim == 3 else (None, 1) + frame_shape
        dset = out_h5.create_dataset(
            "data",
            shape=(0,) + frame_shape if ndim == 3 else (0, 1) + frame_shape,
            maxshape=max_shape,
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
        )

        meta = out_h5.create_group("metadata")
        meta.attrs["created"] = datetime.now().isoformat()
        meta.attrs["background_file"] = os.path.basename(bg_path)
        meta.attrs["transmission_unit"] = "percent"
        meta.attrs["processing_formula"] = "processed[n] = sample[n] / T[n] - background[n]"
        out_h5.flush()

        total = len(valid_files)
        for i, (sample_path, uf) in enumerate(valid_files):
            if worker.is_cancelled():
                break
            worker.emit_log(f"处理 {i + 1}/{total}: {os.path.basename(sample_path)}", "info")

            sample_h5 = load_h5_stack(sample_path)
            if sample_h5 is None:
                worker.emit_log(f"  二次加载失败: {os.path.basename(sample_path)}", "err")
                pct = int((i + 1) / total * 100) if total else 100
                worker.emit_progress(pct, f"处理中... {i + 1}/{total}")
                continue

            Nb = bg_h5["n_frames"]
            is_4d = sample_h5["ndim"] == 4
            sa = sample_h5["data"][:uf]
            ba = np.broadcast_to(bg_h5["data"], sa.shape).copy() if Nb == 1 else bg_h5["data"][:uf]
            T = np.full(uf, unified_transmission, dtype=np.float64) / 100.0
            Tb = T[:, None, None, None] if is_4d else T[:, None, None]
            processed = (sa / Tb - ba).astype(np.float32)

            if is_4d:
                processed = processed[:, 0]

            dset.resize(current_idx + processed.shape[0], axis=0)
            dset[current_idx : current_idx + processed.shape[0]] = processed
            current_idx += processed.shape[0]
            out_h5.flush()

            del sa, ba, processed, sample_h5["data"]

            processed_count += 1
            pct = int((i + 1) / total * 100) if total else 100
            worker.emit_progress(pct, f"处理中... {i + 1}/{total}")

        return H5WorkerResult(
            output_path=output_path,
            processed_count=processed_count,
            total_count=len(valid_files),
            cancelled=worker.is_cancelled(),
        )
    finally:
        if out_h5 is not None:
            out_h5.close()


class AppendReplaceDialog(qt.QDialog):
    """新增或取代对话框。"""

    def __init__(self, parent, item_type):
        super().__init__(parent)
        self.setWindowTitle(f"导入{item_type}")
        layout = qt.QVBoxLayout(self)
        layout.addWidget(qt.QLabel(f"请选择导入方式："))
        self._result = None
        btn_replace = qt.QPushButton("取代现有数据")
        btn_replace.clicked.connect(lambda: self._select("replace"))
        layout.addWidget(btn_replace)
        btn_append = qt.QPushButton("新增到现有数据")
        btn_append.clicked.connect(lambda: self._select("append"))
        layout.addWidget(btn_append)
        btn_cancel = qt.QPushButton("取消")
        btn_cancel.clicked.connect(self.reject)
        layout.addWidget(btn_cancel)

    def _select(self, result):
        self._result = result
        self.accept()

    def get_result(self):
        return self._result


class TiffTab(qt.QWidget):
    """TIFF/EDF 标签页。"""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self._main_window = main_window
        self._files_data: List[Dict] = []
        self._processed: List[Dict] = []
        self._trans_dict: Dict[str, float] = {}
        self._ion_paths: List[str] = []

        self._trans_src = "manual"
        self._bg_name = ""
        self._uni_T = 100.0
        self._per_T: Dict[str, float] = {}
        self._manual_mode = "unified"
        self._out_fmt = "EDF"
        self._h5_stack = "stacked"
        self._user_regex = ""

        # For streaming processing / 流式处理相关
        self._last_zip_path: Optional[str] = None
        self._last_output_folder: Optional[str] = None
        self._last_samples: List[Dict] = []

        # === Phase 1: 匹配结果相关状态 ===
        self._match_results: List[MatchResult] = []
        self._bg_match_result: Optional[MatchResult] = None
        self._match_threshold: float = 0.60
        self._worker_thread: Optional[qt.QThread] = None
        self._worker: Optional[BackgroundProcessingWorker] = None

        self._init_ui()

    def _init_ui(self):
        layout = qt.QVBoxLayout(self)

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(qt.QFrame.NoFrame)
        content = qt.QWidget()
        content_layout = qt.QVBoxLayout(content)

        file_group = qt.QGroupBox("① 文件选择 / File Selection")
        file_layout = qt.QVBoxLayout(file_group)

        btn_row = qt.QHBoxLayout()
        self._btn_files = qt.QPushButton("📂 选择文件")
        self._btn_files.clicked.connect(self._pick_files)
        self._btn_folder = qt.QPushButton("📁 选择文件夹")
        self._btn_folder.clicked.connect(self._pick_folder)
        self._btn_clear = qt.QPushButton("🗑 清空")
        self._btn_clear.clicked.connect(self._clear)
        self._btn_stack = qt.QPushButton("📊 堆叠视图")
        self._btn_stack.clicked.connect(self._show_stack_viewer)
        btn_row.addWidget(self._btn_files)
        btn_row.addWidget(self._btn_folder)
        btn_row.addWidget(self._btn_clear)
        btn_row.addWidget(self._btn_stack)
        self._count_lbl = qt.QLabel("")
        btn_row.addWidget(self._count_lbl)
        btn_row.addStretch()
        file_layout.addLayout(btn_row)

        self._file_list = qt.QListWidget()
        self._file_list.setSelectionMode(qt.QAbstractItemView.ExtendedSelection)
        self._file_list.itemClicked.connect(self._on_file_clicked)
        self._file_list.itemDoubleClicked.connect(self._on_file_double_clicked)
        file_layout.addWidget(self._file_list)

        bg_row = qt.QHBoxLayout()
        bg_row.addWidget(qt.QLabel("背景文件:"))
        self._bg_combo = qt.QComboBox()
        self._bg_combo.setMinimumWidth(200)
        self._bg_combo.currentIndexChanged.connect(self._on_bg_changed)
        bg_row.addWidget(self._bg_combo)
        self._load_bg_btn = qt.QPushButton("加载独立背景...")
        self._load_bg_btn.clicked.connect(self._load_separate_bg)
        bg_row.addWidget(self._load_bg_btn)
        bg_row.addStretch()
        file_layout.addLayout(bg_row)

        content_layout.addWidget(file_group)

        trans_group = qt.QGroupBox("② 透过率设置 / Transmission Source")
        trans_layout = qt.QVBoxLayout(trans_group)

        src_row = qt.QHBoxLayout()
        self._src_manual = qt.QRadioButton("手动输入")
        self._src_manual.setChecked(True)
        self._src_manual.toggled.connect(lambda: self._on_src_changed("manual"))
        self._src_ion = qt.QRadioButton("电离室文件 (SSRF)")
        self._src_ion.toggled.connect(lambda: self._on_src_changed("ionchamber"))
        src_row.addWidget(self._src_manual)
        src_row.addWidget(self._src_ion)
        src_row.addStretch()
        trans_layout.addLayout(src_row)

        self._manual_widget = qt.QWidget()
        self._manual_layout = qt.QVBoxLayout(self._manual_widget)

        mode_row = qt.QHBoxLayout()
        self._mode_unified = qt.QRadioButton("统一值")
        self._mode_unified.setChecked(True)
        self._mode_unified.toggled.connect(lambda: self._on_manual_mode("unified"))
        self._mode_perfile = qt.QRadioButton("逐文件设置")
        self._mode_perfile.toggled.connect(lambda: self._on_manual_mode("perfile"))
        mode_row.addWidget(self._mode_unified)
        mode_row.addWidget(self._mode_perfile)
        mode_row.addStretch()
        self._manual_layout.addLayout(mode_row)

        self._uni_widget = qt.QWidget()
        uni_layout = qt.QHBoxLayout(self._uni_widget)
        uni_layout.addWidget(qt.QLabel("统一透过率 T (%):"))
        self._trans_spin = qt.QDoubleSpinBox()
        self._trans_spin.setRange(0.001, 10000.0)
        self._trans_spin.setValue(100.0)
        self._trans_spin.setDecimals(3)
        self._trans_spin.valueChanged.connect(self._on_trans_changed)
        uni_layout.addWidget(self._trans_spin)
        uni_layout.addWidget(qt.QLabel("%"))
        uni_layout.addStretch()
        self._manual_layout.addWidget(self._uni_widget)

        self._per_scroll = qt.QScrollArea()
        self._per_scroll.setWidgetResizable(True)
        self._per_scroll.setFrameShape(qt.QFrame.NoFrame)
        self._per_scroll.setMinimumHeight(100)
        self._per_scroll.setMaximumHeight(200)
        self._per_scroll.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        self._per_widget = qt.QWidget()
        self._per_layout = qt.QVBoxLayout(self._per_widget)
        self._per_layout.addStretch()
        self._per_scroll.setWidget(self._per_widget)
        self._manual_layout.addWidget(self._per_scroll)
        self._per_scroll.setVisible(False)

        self._manual_layout.addWidget(self._uni_widget)
        trans_layout.addWidget(self._manual_widget)

        self._ion_widget = qt.QWidget()
        ion_layout = qt.QVBoxLayout(self._ion_widget)
        self._ion_widget.setMaximumHeight(280)
        self._ion_widget.setVisible(False)

        ion_btn_row = qt.QHBoxLayout()
        self._ion_pick = qt.QPushButton("📂 选择电离室文件")
        self._ion_pick.clicked.connect(self._pick_ion)
        self._ion_folder = qt.QPushButton("📁 从文件夹导入")
        self._ion_folder.clicked.connect(self._pick_ion_folder)
        self._ion_lbl = qt.QLabel("未选择")
        ion_btn_row.addWidget(self._ion_pick)
        ion_btn_row.addWidget(self._ion_folder)
        ion_btn_row.addWidget(self._ion_lbl)
        ion_btn_row.addStretch()
        ion_layout.addLayout(ion_btn_row)

        ion_ch_row = qt.QHBoxLayout()
        ion_ch_row.addWidget(qt.QLabel("背景通道:"))
        self._ion_bg_ch = qt.QComboBox()
        self._ion_bg_ch.addItems(["Ionchamber0", "Ionchamber1", "Ionchamber2"])
        self._ion_bg_ch.setCurrentText("Ionchamber1")
        ion_ch_row.addWidget(self._ion_bg_ch)
        ion_ch_row.addWidget(qt.QLabel("方法:"))
        self._ion_bg_mth = qt.QComboBox()
        self._ion_bg_mth.addItems(["mean", "median", "trimmed_mean"])
        self._ion_bg_mth.setCurrentText("median")
        ion_ch_row.addWidget(self._ion_bg_mth)
        ion_ch_row.addStretch()
        ion_layout.addLayout(ion_ch_row)

        ion_sa_row = qt.QHBoxLayout()
        ion_sa_row.addWidget(qt.QLabel("样品通道:"))
        self._ion_sa_ch = qt.QComboBox()
        self._ion_sa_ch.addItems(["Ionchamber0", "Ionchamber1", "Ionchamber2"])
        self._ion_sa_ch.setCurrentText("Ionchamber1")
        ion_sa_row.addWidget(self._ion_sa_ch)
        ion_sa_row.addWidget(qt.QLabel("方法:"))
        self._ion_sa_mth = qt.QComboBox()
        self._ion_sa_mth.addItems(["mean", "median", "trimmed_mean"])
        self._ion_sa_mth.setCurrentText("median")
        ion_sa_row.addWidget(self._ion_sa_mth)
        ion_sa_row.addStretch()
        ion_layout.addLayout(ion_sa_row)

        regex_row = qt.QHBoxLayout()
        regex_row.addWidget(qt.QLabel("自定义正则:"))
        self._regex_edit = qt.QLineEdit()
        self._regex_edit.setPlaceholderText(r"可选，如 ^data.*_(\d+)_(\d+)$")
        regex_row.addWidget(self._regex_edit)
        regex_row.addStretch()
        ion_layout.addLayout(regex_row)

        self._ion_calc_btn = qt.QPushButton("⚙ 自动匹配并计算透过率")
        self._ion_calc_btn.clicked.connect(self._calc_ion)
        ion_layout.addWidget(self._ion_calc_btn)

        # === Phase 1: 阈值滑块 ===
        thresh_row = qt.QHBoxLayout()
        thresh_row.addWidget(qt.QLabel("匹配阈值:"))
        self._thresh_slider = qt.QSlider(qt.Qt.Horizontal)
        self._thresh_slider.setMinimum(0)
        self._thresh_slider.setMaximum(100)
        self._thresh_slider.setValue(int(self._match_threshold * 100))
        self._thresh_slider.setTickPosition(qt.QSlider.TicksBelow)
        self._thresh_slider.setTickInterval(10)
        self._thresh_slider.setMinimumWidth(200)
        self._thresh_slider.valueChanged.connect(self._on_thresh_changed)
        thresh_row.addWidget(self._thresh_slider)
        self._thresh_lbl = qt.QLabel(f"{self._match_threshold:.2f}")
        self._thresh_lbl.setMinimumWidth(40)
        thresh_row.addWidget(self._thresh_lbl)
        thresh_row.addStretch()
        ion_layout.addLayout(thresh_row)

        # === Phase 1: 匹配结果表格 ===
        self._match_table = qt.QTableWidget()
        self._match_table.setColumnCount(6)
        self._match_table.setHorizontalHeaderLabels(
            ["图像文件", "电离室文件", "策略", "分数", "状态相似度", "状态"]
        )
        self._match_table.setMinimumHeight(120)
        self._match_table.setMaximumHeight(200)
        self._match_table.setEditTriggers(qt.QAbstractItemView.NoEditTriggers)
        self._match_table.setSelectionBehavior(qt.QAbstractItemView.SelectRows)
        self._match_table.setAlternatingRowColors(True)
        ion_layout.addWidget(self._match_table)

        # === Phase 1: 最低成功匹配说明 + 警告 ===
        self._match_explain = qt.QLabel("")
        self._match_explain.setWordWrap(True)
        self._match_explain.setStyleSheet("color: #e07b39; font-size: 12px; padding: 4px;")
        ion_layout.addWidget(self._match_explain)

        # === 原日志区降级为摘要 ===
        self._ion_scroll = qt.QScrollArea()
        self._ion_scroll.setWidgetResizable(True)
        self._ion_scroll.setFrameShape(qt.QFrame.NoFrame)
        self._ion_scroll.setMinimumHeight(50)
        self._ion_scroll.setMaximumHeight(80)
        self._ion_scroll.setVerticalScrollBarPolicy(qt.Qt.ScrollBarAsNeeded)
        self._ion_result = qt.QTextEdit()
        self._ion_result.setReadOnly(True)
        self._ion_result.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
        self._ion_result.setStyleSheet("border: none; background: transparent; font-size: 11px;")
        self._ion_scroll.setWidget(self._ion_result)
        ion_layout.addWidget(self._ion_scroll)

        trans_layout.addWidget(self._ion_widget)
        content_layout.addWidget(trans_group)

        run_group = qt.QGroupBox("③ 执行 / Run")
        run_layout = qt.QVBoxLayout(run_group)

        fmt_row = qt.QHBoxLayout()
        fmt_row.addWidget(qt.QLabel("输出格式:"))
        self._fmt_group = qt.QButtonGroup()
        for fmt in ["EDF", "TIFF", "HDF5 (.h5)"]:
            rb = qt.QRadioButton(fmt)
            rb.setChecked(fmt == "EDF")
            rb.toggled.connect(lambda checked, f=fmt: self._on_fmt_changed(f) if checked else None)
            self._fmt_group.addButton(rb)
            fmt_row.addWidget(rb)
        self._h5_opt_widget = qt.QWidget()
        h5_opt_layout = qt.QHBoxLayout(self._h5_opt_widget)
        h5_opt_layout.addWidget(qt.QLabel("存储:"))
        self._h5_split = qt.QRadioButton("分解")
        self._h5_stacked = qt.QRadioButton("连续(N,H,W)")
        self._h5_stacked.setChecked(True)
        h5_opt_layout.addWidget(self._h5_split)
        h5_opt_layout.addWidget(self._h5_stacked)
        h5_opt_layout.addStretch()
        self._h5_opt_widget.setVisible(False)
        fmt_row.addWidget(self._h5_opt_widget)
        fmt_row.addStretch()
        run_layout.addLayout(fmt_row)

        run_btn_row = qt.QHBoxLayout()
        self._run_btn = qt.QPushButton("▶ 开始处理")
        self._run_btn.clicked.connect(self._run)
        self._cancel_btn = qt.QPushButton("■ 取消")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_processing)
        self._prog = qt.QProgressBar()
        self._prog.setMinimumWidth(200)
        self._prog_lbl = qt.QLabel("")
        run_btn_row.addWidget(self._run_btn)
        run_btn_row.addWidget(self._cancel_btn)
        run_btn_row.addWidget(self._prog)
        run_btn_row.addWidget(self._prog_lbl)
        run_btn_row.addStretch()
        run_layout.addLayout(run_btn_row)
        content_layout.addWidget(run_group)

        export_group = qt.QGroupBox("④ 导出 / Export")
        export_layout = qt.QVBoxLayout(export_group)

        export_btn_row = qt.QHBoxLayout()
        self._save_btn = qt.QPushButton("💾 保存结果")
        self._save_btn.clicked.connect(self._save)
        self._compare_btn = qt.QPushButton("🖼 对比预览")
        self._compare_btn.clicked.connect(self._show_compare)
        self._compare_btn.setEnabled(False)
        export_btn_row.addWidget(self._save_btn)
        export_btn_row.addWidget(self._compare_btn)
        export_btn_row.addStretch()
        export_layout.addLayout(export_btn_row)
        content_layout.addWidget(export_group)

        log_group = qt.QGroupBox("日志 / Log")
        log_layout = qt.QVBoxLayout(log_group)
        self._log = LogWidget()
        log_layout.addWidget(self._log)
        content_layout.addWidget(log_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _pick_files(self):
        qt.QMessageBox.warning(
            self,
            "文件数量提示",
            "⚠️ 大量文件（如 1000+）可能导致电脑卡顿或内存不足。\n\n"
            "建议：\n"
            "• 优先使用 EDF/TIFF 格式（支持流式处理）\n"
            "• 避免一次加载过多文件\n"
            "• 大型 HDF5 堆叠建议切换到 H5 模式处理（支持后台按帧/按文件写出）",
            qt.QMessageBox.Ok,
        )
        dlg = AppendReplaceDialog(self, "文件")
        dlg.exec_()
        mode = dlg.get_result()
        if mode is None:
            return

        if mode == "replace":
            self._files_data.clear()
            self._per_T.clear()

        dialog = qt.QFileDialog(self)
        dialog.setNameFilters(
            [
                "探测器图像 (*.tif *.tiff *.edf)",
                "所有文件 (*)",
            ]
        )
        dialog.setFileMode(qt.QFileDialog.ExistingFiles)
        if dialog.exec_():
            paths = dialog.selectedFiles()
            if paths:
                self._load_paths(paths)

    def _pick_folder(self):
        qt.QMessageBox.warning(
            self,
            "文件数量提示",
            "⚠️ 大量文件（如 1000+）可能导致电脑卡顿或内存不足。\n\n"
            "建议：\n"
            "• 优先使用 EDF/TIFF 格式（支持流式处理）\n"
            "• 避免一次加载过多文件\n"
            "• 大型 HDF5 堆叠建议切换到 H5 模式处理（支持后台按帧/按文件写出）",
            qt.QMessageBox.Ok,
        )
        dlg = AppendReplaceDialog(self, "文件夹")
        dlg.exec_()
        mode = dlg.get_result()
        if mode is None:
            return

        if mode == "replace":
            self._files_data.clear()
            self._per_T.clear()

        dialog = qt.QFileDialog(self)
        dialog.setFileMode(qt.QFileDialog.Directory)
        if dialog.exec_():
            dirs = dialog.selectedFiles()
            if dirs:
                folder = dirs[0]
                found = []
                for root, _, files in os.walk(folder):
                    for fn in sorted(files):
                        if os.path.splitext(fn)[1].lower() in TIFF_EXTS:
                            found.append(os.path.join(root, fn))
                if found:
                    self._log.append_log(f"文件夹扫描: 找到 {len(found)} 个文件", "info")
                    self._load_paths(found)
                else:
                    qt.QMessageBox.information(
                        self, "未找到文件", f"文件夹中没有 .tif / .tiff / .edf 文件:\n{folder}"
                    )

    def _load_paths(self, paths: List[str]):
        failed = []
        new_count = 0

        for p in paths:
            shape, hdr = probe_image_file(p)
            if shape is None:
                failed.append(os.path.basename(p))
                continue
            # Lazy loading: only store path and metadata, data loaded on demand
            # 延迟加载：仅存储路径和元信息，数据按需加载
            self._files_data.append(
                {
                    "name": os.path.basename(p),
                    "path": p,
                    "data": None,
                    "shape": shape,
                    "header": hdr,
                    "is_bg": False,
                }
            )
            new_count += 1

        if failed:
            self._log.append_log(f"加载失败: {', '.join(failed)}", "warn")

        names = [f["name"] for f in self._files_data]
        self._file_list.clear()
        for n in names:
            self._file_list.addItem(n)

        self._bg_combo.clear()
        self._bg_combo.addItems(names)
        if names:
            self._bg_combo.setCurrentIndex(0)
            self._files_data[0]["is_bg"] = True
            # Load background data immediately, keep in memory
            # 立即加载背景数据并保留在内存
            first_path = self._files_data[0]["path"]
            bg_data, bg_hdr = load_image_file(first_path)
            self._files_data[0]["data"] = bg_data.astype(np.float32)
            self._files_data[0]["header"] = bg_hdr
            self._bg_name = names[0]

        self._count_lbl.setText(f"{len(names)} 个文件")
        # Log changed to reflect lazy loading
        # 日志改为反映延迟加载模式
        self._log.append_log(
            f"已索引 {new_count} 个文件（总计 {len(self._files_data)} 个，仅背景加载到内存）", "ok"
        )
        self._update_file_display()
        self._refresh_per_widget()

        # Auto-select first non-background file and show in image view
        # 自动选择第一个非背景文件并在图像视图中显示
        samples = [f for f in self._files_data if not f["is_bg"]]
        if samples:
            first_sample = samples[0]
            if first_sample["data"] is None:
                data, hdr = load_image_file(first_sample["path"])
                if data is not None:
                    first_sample["data"] = data.astype(np.float32)
                    first_sample["header"] = hdr
            if first_sample["data"] is not None:
                self._main_window.show_in_image_view(first_sample["data"], first_sample["name"])
                # Select it in the file list
                # 在文件列表中选中它
                for i in range(self._file_list.count()):
                    item = self._file_list.item(i)
                    if item.text().lstrip("★ ") == first_sample["name"]:
                        item.setSelected(True)
                        break

    def _clear(self):
        self._files_data.clear()
        self._processed.clear()
        self._trans_dict.clear()
        self._per_T.clear()
        self._file_list.clear()
        self._bg_combo.clear()
        self._count_lbl.setText("")
        self._prog.setValue(0)
        self._prog_lbl.setText("")
        self._compare_btn.setEnabled(False)
        self._last_zip_path = None
        self._last_samples = []
        # === Phase 1: 清空匹配结果 ===
        self._match_results.clear()
        self._bg_match_result = None
        self._match_table.setRowCount(0)
        self._match_explain.setText("")
        self._ion_result.setText("")
        self._log.append_log("已清空文件列表", "dim")
        self._main_window.clear_views()

    def _show_stack_viewer(self):
        samples = [f for f in self._files_data if not f["is_bg"]]
        if not samples:
            qt.QMessageBox.information(self, "提示", "没有可用的图像进行堆叠视图")
            return

        try:
            names = [f["name"] for f in samples]
            source = LazyFrameSource(
                len(samples),
                lambda index: self._load_sample_frame(samples[index]),
                names,
            )
            self._main_window.set_stack_source(source)
            self._log.append_log(f"堆叠视图已切换为按帧加载: {len(samples)} 帧", "ok")
            self._prog.setValue(0)
            self._prog_lbl.setText("")
        except Exception as e:
            self._log.append_log(f"❌ 堆叠视图加载失败: {e}", "err")
            self._prog.setValue(0)
            self._prog_lbl.setText("")

    def _on_file_clicked(self, item):
        pass

    def _on_file_double_clicked(self, item):
        path = item.text().lstrip("★ ")
        for f in self._files_data:
            if f["name"] == path:
                proc_data = None
                proc_name = None

                # Check if we have processed data in memory (HDF5 mode)
                # 检查内存中是否有处理结果（HDF5模式）
                if self._processed:
                    for p in self._processed:
                        if p["name"] == f["name"]:
                            proc_data = p["data"]
                            proc_name = f"处理后: {f['name']}"
                            break

                # Check if we have processed output in ZIP or folder
                # 检查ZIP或文件夹中是否有处理结果
                if proc_data is None:
                    use_zip = self._last_zip_path and os.path.exists(self._last_zip_path)
                    use_folder = self._last_output_folder and os.path.isdir(
                        self._last_output_folder
                    )

                    if use_zip or use_folder:
                        proc_name_base = os.path.splitext(f["name"])[0]
                        ext = self._out_fmt.lower()
                        proc_filename = f"{proc_name_base}.{ext}"

                        try:
                            if use_zip:
                                with zipfile.ZipFile(self._last_zip_path, "r") as zf:
                                    if proc_filename in zf.namelist():
                                        proc_raw = zf.read(proc_filename)
                                        tmp = tempfile.NamedTemporaryFile(
                                            delete=False, suffix=f".{ext}"
                                        )
                                        tmp.write(proc_raw)
                                        tmp.close()
                                        proc_data, _ = load_image_file(tmp.name)
                                        os.unlink(tmp.name)
                                        proc_name = f"处理后: {f['name']}"
                            elif use_folder:
                                proc_path = os.path.join(self._last_output_folder, proc_filename)
                                if os.path.exists(proc_path):
                                    proc_data, _ = load_image_file(proc_path)
                                    proc_name = f"处理后: {f['name']}"
                        except Exception as e:
                            self._log.append_log(f"加载处理结果失败: {e}", "err")

                # Show processed image if available, otherwise show original
                # 如果有处理结果显示处理后的，否则显示原始图像
                if proc_data is not None:
                    self._main_window.show_in_image_view(proc_data.astype(np.float32), proc_name)
                else:
                    # Lazy load original on demand / 按需延迟加载原始图像
                    if f["data"] is None:
                        data, hdr = load_image_file(f["path"])
                        if data is not None:
                            f["data"] = data.astype(np.float32)
                            f["header"] = hdr
                    if f["data"] is not None:
                        self._main_window.show_in_image_view(f["data"], f["name"])
                    # Release non-background data after viewing to save memory
                    # 查看后释放非背景数据以节省内存
                    if not f["is_bg"]:
                        f["data"] = None
                break

    def _on_bg_changed(self, index):
        if index < 0:
            return
        self._bg_name = self._bg_combo.currentText()
        # Find the new background and load it if needed
        # 查找新背景并在需要时加载
        for f in self._files_data:
            if f["name"] == self._bg_name:
                if f["data"] is None:
                    data, hdr = load_image_file(f["path"])
                    if data is not None:
                        f["data"] = data.astype(np.float32)
                        f["header"] = hdr
                f["is_bg"] = True
            else:
                f["is_bg"] = False
        self._update_file_display()

    def _update_file_display(self):
        self._file_list.clear()
        for f in self._files_data:
            prefix = "★ " if f["is_bg"] else "  "
            self._file_list.addItem(prefix + f["name"])

    def _load_separate_bg(self):
        dialog = qt.QFileDialog(self)
        dialog.setNameFilters(
            [
                "探测器图像 (*.tif *.tiff *.edf)",
                "所有文件 (*)",
            ]
        )
        if dialog.exec_():
            paths = dialog.selectedFiles()
            if paths:
                path = paths[0]
                data, hdr = load_image_file(path)
                if data is not None:
                    name = f"[独立] {os.path.basename(path)}"
                    # Mark all other files as non-background
                    # 将所有其他文件标记为非背景
                    for f in self._files_data:
                        f["is_bg"] = False
                    # Add separate background with float32
                    # 添加独立背景，使用float32节省内存
                    self._files_data.append(
                        {
                            "name": name,
                            "path": path,
                            "data": data.astype(np.float32),
                            "shape": data.shape,
                            "header": hdr,
                            "is_bg": True,
                        }
                    )
                    self._bg_combo.addItem(name)
                    self._bg_combo.setCurrentIndex(self._bg_combo.count() - 1)
                    self._bg_name = name
                    self._update_file_display()
                    self._log.append_log(f"独立背景已加载: {path}", "ok")

    def _update_stack_viewer(self):
        if self._files_data:
            samples = [f for f in self._files_data if not f["is_bg"]]
            if samples:
                source = LazyFrameSource(
                    len(samples),
                    lambda index: self._load_sample_frame(samples[index]),
                    [f["name"] for f in samples],
                )
                self._main_window.set_stack_source(source)

    def _load_sample_frame(self, file_data: Dict[str, Any]) -> np.ndarray:
        data = file_data.get("data")
        if data is not None:
            return _ensure_float32_frame(data)
        loaded, _ = load_image_file(file_data["path"])
        if loaded is None:
            raise ValueError(f"无法读取图像: {file_data['name']}")
        return _ensure_float32_frame(loaded)

    def _load_processed_output_frame(
        self,
        file_data: Dict[str, Any],
        ext: str,
        use_zip: bool,
        use_folder: bool,
    ) -> np.ndarray:
        proc_name_base = os.path.splitext(file_data["name"])[0]
        proc_filename = f"{proc_name_base}.{ext}"
        proc_data = None
        zip_path = self._last_zip_path if use_zip and self._last_zip_path else None
        output_folder = (
            self._last_output_folder if use_folder and self._last_output_folder else None
        )

        if zip_path:
            with zipfile.ZipFile(zip_path, "r") as zf:
                if proc_filename not in zf.namelist():
                    raise FileNotFoundError(proc_filename)
                proc_raw = zf.read(proc_filename)
                tmp_path = None
                try:
                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
                    tmp.write(proc_raw)
                    tmp_path = tmp.name
                    tmp.close()
                    proc_data, _ = load_image_file(tmp_path)
                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        elif output_folder:
            proc_path = os.path.join(output_folder, proc_filename)
            if not os.path.exists(proc_path):
                raise FileNotFoundError(proc_filename)
            proc_data, _ = load_image_file(proc_path)

        if proc_data is None:
            raise ValueError(f"无法读取处理结果: {proc_filename}")
        return _ensure_float32_frame(proc_data)

    def _on_src_changed(self, src: str):
        self._trans_src = src
        self._manual_widget.setVisible(src == "manual")
        self._ion_widget.setVisible(src == "ionchamber")

    def _on_manual_mode(self, mode: str):
        self._manual_mode = mode
        self._uni_widget.setVisible(mode == "unified")
        self._per_scroll.setVisible(mode == "perfile")
        if mode == "perfile":
            self._refresh_per_widget()

    def _on_trans_changed(self, value: float):
        self._uni_T = value

    def _refresh_per_widget(self):
        while self._per_layout.count() > 1:
            item = self._per_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for f in self._files_data:
            if not f["is_bg"]:
                row = qt.QHBoxLayout()
                lbl = qt.QLabel(f["name"])
                lbl.setFixedWidth(200)
                lbl.setTextInteractionFlags(qt.Qt.TextSelectableByMouse)
                row.addWidget(lbl)
                spin = qt.QDoubleSpinBox()
                spin.setRange(0.001, 10000.0)
                spin.setValue(self._per_T.get(f["name"], 100.0))
                spin.valueChanged.connect(lambda v, name=f["name"]: self._per_T.update({name: v}))
                row.addWidget(spin)
                row.addWidget(qt.QLabel("%"))
                row.addStretch()
                self._per_layout.insertLayout(self._per_layout.count() - 1, row)

    def _pick_ion(self):
        dlg = AppendReplaceDialog(self, "电离室文件")
        dlg.exec_()
        mode = dlg.get_result()
        if mode is None:
            return

        if mode == "replace":
            self._ion_paths.clear()

        dialog = qt.QFileDialog(self)
        dialog.setNameFilters(
            [
                "电离室文件 (*.ionchamber *.txt)",
                "所有文件 (*)",
            ]
        )
        dialog.setFileMode(qt.QFileDialog.ExistingFiles)
        if dialog.exec_():
            paths = dialog.selectedFiles()
            if paths:
                self._ion_paths.extend(paths)
                self._ion_lbl.setText(f"已选 {len(self._ion_paths)} 个")

    def _pick_ion_folder(self):
        dlg = AppendReplaceDialog(self, "电离室文件夹")
        dlg.exec_()
        mode = dlg.get_result()
        if mode is None:
            return

        if mode == "replace":
            self._ion_paths.clear()

        dialog = qt.QFileDialog(self)
        dialog.setFileMode(qt.QFileDialog.Directory)
        if dialog.exec_():
            dirs = dialog.selectedFiles()
            if dirs:
                folder = dirs[0]
                found = []
                for root, _, files in os.walk(folder):
                    for fn in sorted(files):
                        low = fn.lower()
                        if low.endswith(".ionchamber") or low.endswith(".txt"):
                            found.append(os.path.join(root, fn))
                if found:
                    self._ion_paths.extend(found)
                    self._ion_lbl.setText(f"已选 {len(self._ion_paths)} 个文件")
                    self._log.append_log(f"电离室文件夹: {len(found)} 个文件", "info")

    # === Phase 1: 阈值滑块回调 ===
    def _on_thresh_changed(self, val: int):
        """阈值滑块改变时只更新标签和表格筛选，不重新读文件。"""
        self._match_threshold = val / 100.0
        self._thresh_lbl.setText(f"{self._match_threshold:.2f}")
        if self._match_results or self._bg_match_result:
            self._refresh_match_table()

    # === Phase 1: 构建结构化匹配结果（单一数据源） ===
    def _build_match_results(self):
        """
        为背景和所有样品构建 MatchResult 列表。

        统一数据流：
        1. 解析背景 + 样品 ion 匹配
        2. 计算各样品 transmission
        3. 判断各样品是否 above_threshold
        4. 只将 above_threshold 的样品写入 _trans_dict
        """
        bg_fd = next((f for f in self._files_data if f["is_bg"]), None)
        samples = [f for f in self._files_data if not f["is_bg"]]
        if not samples or bg_fd is None:
            return
        if not self._ion_paths:
            return

        self._user_regex = self._regex_edit.text().strip()
        thresh = self._match_threshold

        bg_ion, bg_strat, bg_score, bg_state_sim, bg_candidates = match_ionchamber_detail(
            bg_fd["name"], self._ion_paths, self._user_regex
        )
        df_bg = load_ionchamber(bg_ion) if bg_ion else None
        I_bg = (
            calc_intensity(df_bg, self._ion_bg_ch.currentText(), self._ion_bg_mth.currentText())
            if df_bg is not None
            else None
        )

        self._bg_match_result = MatchResult(
            tiff_name=bg_fd["name"],
            is_bg=True,
            matched_path=bg_ion,
            strategy=bg_strat,
            score=bg_score,
            state_sim=bg_state_sim,
            success=bg_ion is not None and I_bg is not None and I_bg != 0,
            error_msg=None if (bg_ion and I_bg) else ("无匹配" if not bg_ion else "电离室读取失败"),
            ion_intensity=I_bg,
            candidates=bg_candidates,
        )

        self._trans_dict.clear()
        self._match_results.clear()
        below_threshold_names = []

        for fd in samples:
            ion_p, ion_strat, ion_score, ion_state_sim, ion_candidates = match_ionchamber_detail(
                fd["name"], self._ion_paths, self._user_regex
            )
            df_s = load_ionchamber(ion_p) if ion_p else None
            I_s = (
                calc_intensity(df_s, self._ion_sa_ch.currentText(), self._ion_sa_mth.currentText())
                if df_s is not None
                else None
            )

            T = None
            if I_s is not None and I_bg is not None and I_bg != 0:
                T = I_s / I_bg * 100.0

            ion_success = ion_p is not None and T is not None
            above_thresh = ion_success and ion_score >= thresh

            mr = MatchResult(
                tiff_name=fd["name"],
                is_bg=False,
                matched_path=ion_p,
                strategy=ion_strat,
                score=ion_score,
                state_sim=ion_state_sim,
                transmission=T,
                ion_intensity=I_s,
                bg_intensity=I_bg,
                success=ion_success,
                error_msg=(
                    None
                    if ion_success
                    else (
                        "无匹配电离室文件"
                        if not ion_p
                        else ("电离室读取失败" if df_s is None else "强度计算失败")
                    )
                ),
                candidates=ion_candidates,
            )
            self._match_results.append(mr)

            if above_thresh:
                self._trans_dict[fd["name"]] = T
            else:
                below_threshold_names.append(fd["name"])

        return below_threshold_names

    # === Phase 1: 刷新匹配结果表格 ===
    def _refresh_match_table(self):
        """
        根据当前阈值筛选并刷新匹配结果表格。

        排序规则：背景行置顶；样品行按分数升序（最危险排在最前）。
        超长文件名自动设置 tooltip 显示完整路径。

        状态语义：
          passed     : 成功匹配 且 score >= threshold（绿色 ✓）
          below_thresh: 成功匹配 但 score <  threshold（橙色 ⚠）
          match_fail : 匹配失败（红色 ✗）
        """
        self._match_table.setRowCount(0)
        thresh = self._match_threshold
        stats = _collect_match_statistics(self._match_results, thresh, self._trans_dict)

        bg_row_data = None
        passed_rows = []
        below_thresh_rows = []
        failed_rows = []
        lowest_ok_item = stats.worst_case if stats.worst_case_kind == "lowest_passing" else None

        for mr in self._match_results:
            score = _match_score(mr)
            status_key = _match_status(mr, thresh)

            if status_key == "failed":
                status = "✗ fail"
                ion_name = os.path.basename(mr.matched_path) if mr.matched_path else "—"
                row_data = [
                    mr.tiff_name,
                    ion_name,
                    mr.strategy or "—",
                    f"{score:.3f}",
                    f"{(mr.state_sim or 0.0):.3f}",
                    status,
                ]
                failed_rows.append((row_data, score, mr))
            elif status_key == "passed":
                status = "✓ passed"
                ion_name = os.path.basename(mr.matched_path) if mr.matched_path else "—"
                row_data = [
                    mr.tiff_name,
                    ion_name,
                    mr.strategy or "—",
                    f"{score:.3f}",
                    f"{(mr.state_sim or 0.0):.3f}",
                    status,
                ]
                passed_rows.append((row_data, score, mr))
            else:
                status = f"⚠ <{thresh:.2f}"
                ion_name = os.path.basename(mr.matched_path) if mr.matched_path else "—"
                row_data = [
                    mr.tiff_name,
                    ion_name,
                    mr.strategy or "—",
                    f"{score:.3f}",
                    f"{(mr.state_sim or 0.0):.3f}",
                    status,
                ]
                below_thresh_rows.append((row_data, score, mr))

        if self._bg_match_result:
            mr = self._bg_match_result
            score = _match_score(mr)
            ion_name = os.path.basename(mr.matched_path) if mr.matched_path else "—"
            bg_row_data = (
                [
                    mr.tiff_name,
                    ion_name,
                    mr.strategy or "—",
                    f"{score:.3f}",
                    f"{(mr.state_sim or 0.0):.3f}",
                    "背景",
                ],
                score,
                mr,
            )

        row_idx = 0

        if bg_row_data:
            row_data, score, mr = bg_row_data
            self._match_table.insertRow(row_idx)
            for col, val in enumerate(row_data):
                item = qt.QTableWidgetItem(val)
                item.setBackground(qt.QColor(30, 30, 60))
                item.setForeground(qt.QColor(160, 160, 255))
                if len(val) > 30:
                    item.setToolTip(val)
                self._match_table.setItem(row_idx, col, item)
            row_idx += 1

        for row_data, score, mr in sorted(failed_rows, key=lambda x: x[1]):
            self._match_table.insertRow(row_idx)
            for col, val in enumerate(row_data):
                item = qt.QTableWidgetItem(val)
                item.setBackground(qt.QColor(80, 20, 20))
                item.setForeground(qt.QColor(255, 100, 100))
                if len(val) > 30:
                    item.setToolTip(val)
                self._match_table.setItem(row_idx, col, item)
            row_idx += 1

        for row_data, score, mr in sorted(below_thresh_rows, key=lambda x: x[1]):
            self._match_table.insertRow(row_idx)
            for col, val in enumerate(row_data):
                item = qt.QTableWidgetItem(val)
                item.setBackground(qt.QColor(60, 50, 20))
                item.setForeground(qt.QColor(255, 180, 80))
                if len(val) > 30:
                    item.setToolTip(val)
                self._match_table.setItem(row_idx, col, item)
            row_idx += 1

        for row_data, score, mr in sorted(passed_rows, key=lambda x: x[1]):
            self._match_table.insertRow(row_idx)
            for col, val in enumerate(row_data):
                item = qt.QTableWidgetItem(val)
                if lowest_ok_item is mr:
                    item.setBackground(qt.QColor(50, 70, 50))
                    item.setForeground(qt.QColor(255, 200, 80))
                elif score >= thresh + 0.10:
                    item.setBackground(qt.QColor(20, 50, 20))
                    item.setForeground(qt.QColor(80, 255, 80))
                else:
                    item.setBackground(qt.QColor(60, 50, 15))
                    item.setForeground(qt.QColor(255, 180, 60))
                if len(val) > 30:
                    item.setToolTip(val)
                self._match_table.setItem(row_idx, col, item)
            row_idx += 1

        self._match_table.resizeColumnsToContents()
        self._match_table.setColumnWidth(0, 260)
        self._match_table.setColumnWidth(1, 200)
        self._match_table.setColumnWidth(2, 150)
        self._match_table.setColumnWidth(3, 70)
        self._match_table.setColumnWidth(4, 80)
        self._match_table.setColumnWidth(5, 100)
        self._match_explain.setText(_build_match_explain_html(stats, thresh))

    def _calc_ion(self):
        """
        Phase 1 统一数据流：
        1. 前置校验
        2. 调用 _build_match_results()（单一数据源）
        3. 生成日志摘要 + 刷新 UI
        """
        bg_fd = next((f for f in self._files_data if f["is_bg"]), None)
        samples = [f for f in self._files_data if not f["is_bg"]]
        if not samples or bg_fd is None:
            qt.QMessageBox.warning(self, "提示", "请先加载文件并选定背景文件")
            return
        if not self._ion_paths:
            qt.QMessageBox.warning(self, "提示", "请先选择电离室文件")
            return

        self._log.append_log("─" * 50, "dim")
        self._log.append_log(
            f"电离室匹配 | 电离室文件 {len(self._ion_paths)} 个 | 图像 {len(samples) + 1} 个",
            "info",
        )

        below_threshold_names = self._build_match_results()

        if self._bg_match_result is None or not self._bg_match_result.success:
            bg_name = bg_fd["name"]
            self._log.append_log(f"背景 [{bg_name}] → 无匹配或强度异常", "err")
            qt.QMessageBox.warning(
                self, "背景无匹配", f"背景文件 [{bg_name}] 找不到匹配的电离室文件或强度异常"
            )
            return

        bg_ion = self._bg_match_result.matched_path
        self._log.append_log(
            f"背景: {bg_fd['name']} → {os.path.basename(bg_ion)} "
            f"[{self._bg_match_result.strategy} score={self._bg_match_result.score:.3f} "
            f"sim={self._bg_match_result.state_sim:.3f}]",
            "ok",
        )

        thresh = self._match_threshold
        lines = []
        below_threshold_names = []

        for mr in self._match_results:
            ion_p = mr.matched_path
            ion_strat = mr.strategy
            ion_score = mr.score
            ion_state_sim = mr.state_sim
            ion_success = mr.success
            T = mr.transmission
            fd_name = mr.tiff_name

            if not ion_success:
                lines.append(f"✗ {fd_name}: {mr.error_msg or '匹配失败'}")
                self._log.append_log(f"  ✗ {fd_name} → {mr.error_msg or '匹配失败'}", "warn")
            elif ion_p and T is None:
                lines.append(f"✗ {fd_name}: 强度计算失败")
                self._log.append_log(f"  ✗ {fd_name} → 强度计算失败", "err")
            elif fd_name in self._trans_dict:
                lines.append(
                    f"✓ {fd_name} ← {os.path.basename(ion_p)} T={T:.2f}% [s={ion_score:.3f}]"
                )
                self._log.append_log(
                    f"  ✓ {fd_name} → {os.path.basename(ion_p)} "
                    f"[{ion_strat} s={ion_score:.3f} sim={ion_state_sim:.3f}] T={T:.2f}%",
                    "ok",
                )
            else:
                below_threshold_names.append(fd_name)
                lines.append(f"⚠ {fd_name}: 分数 {ion_score:.3f} < 阈值 {thresh:.2f}")
                self._log.append_log(
                    f"  ⚠ {fd_name} → {os.path.basename(ion_p)} "
                    f"[{ion_strat} s={ion_score:.3f} sim={ion_state_sim:.3f}] "
                    f"分数低于阈值 {thresh:.2f}",
                    "warn",
                )

        n_ok = len(self._trans_dict)
        n_total = len(samples)
        n_below = len(below_threshold_names)

        if n_below > 0:
            self._log.append_log(
                f"⚠ {n_below}/{n_total} 个样品匹配分数低于阈值 {thresh:.2f}，已排除", "warn"
            )
            qt.QMessageBox.warning(
                self,
                "阈值过滤",
                f"有 {n_below} 个样品匹配分数低于当前阈值 {thresh:.2f}：\n"
                + "\n".join(f"  • {n}" for n in below_threshold_names)
                + f"\n\n这些样品已排除，不进入后续处理。\n"
                + "请调整阈值或检查匹配结果后重试。",
                qt.QMessageBox.Ok,
            )

        if n_ok == 0 and n_total > 0:
            summary = f"匹配: 0/{n_total} 个\n" + "\n".join(lines)
            self._ion_result.setText(summary)
            self._refresh_match_table()
            return

        summary = f"匹配: {n_ok}/{n_total} 个\n" + "\n".join(lines)
        self._ion_result.setText(summary)
        self._refresh_match_table()

    def _on_fmt_changed(self, fmt: str):
        self._out_fmt = fmt
        self._h5_opt_widget.setVisible(fmt == "HDF5 (.h5)")
        if fmt == "HDF5 (.h5)":
            qt.QMessageBox.information(
                self,
                "提示",
                "当前 TIFF/EDF → HDF5 导出会先汇总结果后再写入。\n"
                "如果要处理大型 HDF5 堆叠，请切换到 H5 模式使用后台流式写出。",
                qt.QMessageBox.Ok,
            )

    def _get_trans(self) -> Optional[Dict[str, float]]:
        samples = [f for f in self._files_data if not f["is_bg"]]
        if self._trans_src == "manual":
            if self._manual_mode == "unified":
                T = self._uni_T
                return {f["name"]: T for f in samples}
            else:
                d = {}
                for f in samples:
                    d[f["name"]] = self._per_T.get(f["name"], 100.0)
                return d
        else:
            if not self._trans_dict:
                return None
            return self._trans_dict

    def _set_running_state(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        for widget in (
            self._btn_files,
            self._btn_folder,
            self._btn_clear,
            self._btn_stack,
            self._bg_combo,
            self._load_bg_btn,
            self._save_btn,
            self._compare_btn,
        ):
            widget.setEnabled(not running)

    def _cleanup_worker(self):
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread.deleteLater()
        self._worker_thread = None
        self._worker = None

    def _cancel_processing(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self._cancel_btn.setEnabled(False)
            self._prog_lbl.setText("正在取消...")
            self._log.append_log("已请求取消，等待当前文件完成...", "warn")

    def _on_worker_progress(self, value: int, text: str):
        self._prog.setValue(value)
        self._prog_lbl.setText(text)

    def _on_worker_failed(self, message: str):
        self._cleanup_worker()
        self._set_running_state(False)
        self._compare_btn.setEnabled(
            bool(self._processed) or bool(self._last_zip_path) or bool(self._last_output_folder)
        )
        self._log.append_log(f"❌ 处理失败: {message}", "err")

    def _finalize_stream_result(self, result: TiffWorkerResult):
        if result.output_mode == "zip":
            self._last_zip_path = result.output_path
            self._last_output_folder = None
        elif result.output_mode == "folder":
            self._last_zip_path = None
            self._last_output_folder = result.output_path

        processed_name_set = set(result.processed_names)
        self._last_samples = [
            fd for fd in self._files_data if fd["name"] in processed_name_set and not fd["is_bg"]
        ]
        self._compare_btn.setEnabled(result.processed_count > 0)

    def _auto_open_compare_preview(self):
        """自动切换到对比视图，仅首组先渲染。"""
        self._show_compare()
        self._log.append_log("已自动切换到对比视图，当前仅首组已渲染，可继续逐帧查看。", "ok")

    def _on_h5_memory_finished(self, result: TiffWorkerResult):
        self._cleanup_worker()
        self._set_running_state(False)
        self._processed = result.processed
        processed_name_set = set(result.processed_names)
        self._last_samples = [
            fd for fd in self._files_data if fd["name"] in processed_name_set and not fd["is_bg"]
        ]
        self._compare_btn.setEnabled(result.processed_count > 0)
        self._log.append_log(f"完成: {result.processed_count} 个文件已处理，请在导出中保存", "ok")
        if result.processed_count > 0:
            self._auto_open_compare_preview()

    def _on_h5_memory_cancelled(self, result: TiffWorkerResult):
        self._cleanup_worker()
        self._set_running_state(False)
        self._processed = result.processed
        processed_name_set = set(result.processed_names)
        self._last_samples = [
            fd for fd in self._files_data if fd["name"] in processed_name_set and not fd["is_bg"]
        ]
        self._compare_btn.setEnabled(result.processed_count > 0)
        self._prog_lbl.setText("已取消")
        self._log.append_log(
            f"已取消: {result.processed_count}/{result.total_count} 个文件已完成，可按需导出当前结果",
            "warn",
        )
        if result.processed_count > 0:
            self._auto_open_compare_preview()

    def _on_stream_finished(self, result: TiffWorkerResult):
        self._cleanup_worker()
        self._set_running_state(False)
        self._finalize_stream_result(result)
        self._log.append_log(
            f"完成: {result.processed_count}/{result.total_count} 个文件已保存到 {result.output_path}",
            "ok",
        )
        if result.processed_count > 0:
            self._auto_open_compare_preview()

    def _on_stream_cancelled(self, result: TiffWorkerResult):
        self._cleanup_worker()
        self._set_running_state(False)
        self._finalize_stream_result(result)
        self._prog_lbl.setText("已取消")
        self._log.append_log(
            f"已取消: {result.processed_count}/{result.total_count} 个文件已保存到 {result.output_path}",
            "warn",
        )
        if result.processed_count > 0:
            self._auto_open_compare_preview()

    def _run(self):
        if len(self._files_data) < 2:
            qt.QMessageBox.warning(self, "文件不足", "至少需要1个背景+1个样品文件")
            return

        bg_fd = next((f for f in self._files_data if f["is_bg"]), None)
        samples = [f for f in self._files_data if not f["is_bg"]]
        if bg_fd is None:
            qt.QMessageBox.warning(self, "未选背景", "请在下拉框中指定背景文件")
            return

        td = self._get_trans()
        if td is None:
            qt.QMessageBox.warning(self, "透过率不完整", "请先完成透过率计算")
            return

        # 只处理阈值通过（已在 _trans_dict 中）的样品
        valid_samples = [fd for fd in samples if fd["name"] in td]
        excluded = [fd for fd in samples if fd["name"] not in td]
        if excluded:
            self._log.append_log(f"⚠ 已跳过 {len(excluded)} 个低于阈值的样品", "warn")

        fmt = self._out_fmt

        # TIFF/EDF → HDF5：当前仍为汇总后写出模式
        if fmt == "HDF5 (.h5)":
            if bg_fd["data"] is None:
                qt.QMessageBox.warning(self, "背景无效", "背景图像尚未加载成功")
                return

            self._processed.clear()
            self._last_zip_path = None
            self._last_output_folder = None
            self._last_samples = []
            self._prog.setValue(0)
            self._set_running_state(True)
            self._log.append_log("开始处理（HDF5 非流式，后台）...", "info")

            bg_data = bg_fd["data"].astype(np.float32, copy=True)
            bg_shape = tuple(bg_fd["shape"])
            sample_snapshots = [
                {"name": fd["name"], "path": fd["path"], "shape": tuple(fd["shape"])}
                for fd in valid_samples
            ]
            self._worker_thread = qt.QThread(self)
            self._worker = BackgroundProcessingWorker(
                lambda worker: _run_tiff_h5_job(worker, sample_snapshots, bg_data, bg_shape, td)
            )
            self._worker.moveToThread(self._worker_thread)
            self._worker_thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._on_worker_progress)
            self._worker.log.connect(self._log.append_log)
            self._worker.finished.connect(self._on_h5_memory_finished)
            self._worker.cancelled.connect(self._on_h5_memory_cancelled)
            self._worker.failed.connect(self._on_worker_failed)
            self._worker_thread.start()
            return

        # EDF/TIFF: 流式处理（写入 ZIP 或文件夹）
        # 让用户选择输出方式
        msg_box = qt.QMessageBox(self)
        msg_box.setWindowTitle("选择输出方式")
        msg_box.setText("请选择输出方式：")
        zip_btn = msg_box.addButton("打包 ZIP", qt.QMessageBox.AcceptRole)
        folder_btn = msg_box.addButton("输出到文件夹", qt.QMessageBox.AcceptRole)
        cancel_btn = msg_box.addButton("取消", qt.QMessageBox.RejectRole)
        msg_box.setDefaultButton(zip_btn)
        msg_box.exec_()

        clicked = msg_box.clickedButton()
        if clicked == zip_btn:
            sp = qt.QFileDialog.getSaveFileName(self, "保存为 ZIP", "processed.zip", "ZIP (*.zip)")[
                0
            ]
            if not sp:
                self._run_btn.setEnabled(True)
                return
            output_mode = "zip"
            output_path = sp
            self._log.append_log("开始流式处理（ZIP）...", "info")
        elif clicked == folder_btn:
            folder = qt.QFileDialog.getExistingDirectory(self, "选择输出文件夹")
            if not folder:
                self._run_btn.setEnabled(True)
                return
            output_mode = "folder"
            output_path = folder
            self._log.append_log("开始流式处理（文件夹）...", "info")
        elif clicked == cancel_btn:
            self._run_btn.setEnabled(True)
            return

        self._processed.clear()
        self._prog.setValue(0)
        if bg_fd["data"] is None:
            qt.QMessageBox.warning(self, "背景无效", "背景图像尚未加载成功")
            return

        self._last_zip_path = None
        self._last_output_folder = None
        self._last_samples = []
        self._set_running_state(True)
        bg_data = bg_fd["data"].astype(np.float32, copy=True)
        bg_shape = tuple(bg_fd["shape"])
        sample_snapshots = [
            {"name": fd["name"], "path": fd["path"], "shape": tuple(fd["shape"])}
            for fd in valid_samples
        ]
        self._worker_thread = qt.QThread(self)
        self._worker = BackgroundProcessingWorker(
            lambda worker: _run_tiff_stream_job(
                worker,
                sample_snapshots,
                bg_data,
                bg_shape,
                td,
                fmt,
                output_mode,
                output_path,
            )
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.log.connect(self._log.append_log)
        self._worker.finished.connect(self._on_stream_finished)
        self._worker.cancelled.connect(self._on_stream_cancelled)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker_thread.start()

    # === Phase 1: 导出匹配结果文件 ===
    def _run_export_match_results(self, output_path: str, output_mode: str):
        """将匹配结果（CSV/JSON/TXT）写入 ZIP 或输出文件夹。"""
        try:
            summary = _build_match_summary_dict(
                self._bg_match_result, self._match_results, self._match_threshold, self._trans_dict
            )
            csv_data = _export_match_results_csv(summary)
            json_data = _export_match_results_json(summary)
            txt_data = _export_match_results_txt(summary)

            if output_mode == "zip":
                with zipfile.ZipFile(output_path, "a", zipfile.ZIP_DEFLATED) as zf:
                    zf.writestr("match_results.csv", csv_data)
                    zf.writestr("match_results.json", json_data)
                    zf.writestr("match_results.txt", txt_data)
                self._log.append_log("匹配结果已写入 ZIP（match_results.csv/.json/.txt）", "ok")
            else:
                base = output_path
                for fname, data in [
                    ("match_results.csv", csv_data),
                    ("match_results.json", json_data),
                    ("match_results.txt", txt_data),
                ]:
                    p = os.path.join(base, fname)
                    with open(p, "w", encoding="utf-8") as f:
                        f.write(data)
                self._log.append_log(
                    "匹配结果已写入输出文件夹（match_results.csv/.json/.txt）", "ok"
                )
        except Exception as e:
            self._log.append_log(f"⚠ 写入匹配结果文件失败: {e}", "warn")

    def _save(self):
        fmt = self._out_fmt

        # If we already have a ZIP from streaming _run, just confirm
        # 如果已经有流式处理生成的ZIP，直接确认
        if self._last_zip_path and os.path.exists(self._last_zip_path):
            if fmt != "HDF5 (.h5)":
                qt.QMessageBox.information(
                    self, "已保存", f"处理结果已保存到:\n{self._last_zip_path}"
                )
                return

        # If streaming was done to folder (_last_zip_path is None but _last_samples has data)
        # 流式处理输出到文件夹的情况
        if self._processed and self._last_samples:
            if fmt != "HDF5 (.h5)":
                qt.QMessageBox.information(
                    self,
                    "提示",
                    "处理结果已在处理时直接保存到文件夹。\n如需重新保存，请再次执行处理。",
                )
                return

        # Otherwise, process and save if _processed has data
        if not self._processed:
            qt.QMessageBox.information(self, "提示", "请先执行处理")
            return

        fmt = self._out_fmt
        if fmt == "HDF5 (.h5)":
            sp = qt.QFileDialog.getSaveFileName(
                self, "保存 H5 结果", f"processed_results.h5", "HDF5 (*.h5)"
            )[0]
            if not sp:
                return
            td = self._get_trans() or {}
            buf = export_tiff_to_h5(self._processed, td, self._bg_name, self._h5_stack == "stacked")
            open(sp, "wb").write(buf)
            self._log.append_log(f"H5: {sp}", "ok")
        else:
            sp = qt.QFileDialog.getSaveFileName(self, "保存结果", "processed.zip", "ZIP (*.zip)")[0]
            if not sp:
                return
            ext = fmt.lower()
            self._prog.setValue(0)
            self._prog_lbl.setText("")
            total = len(self._processed)
            needs_cache = ext in ("tif", "tiff")
            cache_ctx: Any = CacheDir() if needs_cache else nullcontext()
            with cache_ctx as cache_dir:
                with zipfile.ZipFile(sp, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, pf in enumerate(self._processed):
                        try:
                            arr = pf["data"].astype(np.float32)
                            if ext == "edf":
                                h = (pf["header"] or {}).copy()
                                h.update(
                                    {
                                        "Dim_1": str(arr.shape[1]),
                                        "Dim_2": str(arr.shape[0]),
                                        "DataType": "Float32",
                                    }
                                )
                                img_bytes = encode_image_to_bytes(arr, fmt="edf", header=h)
                            else:
                                img_bytes = encode_image_to_bytes(
                                    arr, fmt="tif", header=pf["header"], cache=cache_dir
                                )
                            zf.writestr(f"{os.path.splitext(pf['name'])[0]}.{ext}", img_bytes)
                        except Exception as e:
                            self._log.append_log(f"编码失败 {pf['name']}: {e}", "warn")

                    pct = int((i + 1) / total * 100)
                    self._prog.setValue(pct)
                    self._prog_lbl.setText(f"{i + 1}/{total}")
            self._log.append_log(f"ZIP: {sp}", "ok")

    def _show_compare(self):
        samples = [f for f in self._files_data if not f["is_bg"]]
        if not samples:
            return

        if self._processed:
            processed_by_name = {p["name"]: p for p in self._processed if p.get("data") is not None}
            compare_samples = [fd for fd in samples if fd["name"] in processed_by_name]
            if not compare_samples:
                qt.QMessageBox.information(self, "提示", "没有可用的处理结果进行对比")
                return

            orig_source = LazyFrameSource(
                len(compare_samples),
                lambda index: self._load_sample_frame(compare_samples[index]),
                [fd["name"] for fd in compare_samples],
            )
            proc_source = LazyFrameSource(
                len(compare_samples),
                lambda index: _ensure_float32_frame(
                    processed_by_name[compare_samples[index]["name"]]["data"]
                ),
                [f"processed_{fd['name']}" for fd in compare_samples],
            )
            self._main_window.show_compare_window(
                orig_source,
                proc_source,
                [f["name"] for f in compare_samples],
                [f"processed_{fd['name']}" for fd in compare_samples],
            )
            self._log.append_log(f"对比视图已切换为按帧加载: {len(compare_samples)} 组", "ok")
            return

        zip_path = (
            self._last_zip_path
            if self._last_zip_path and os.path.exists(self._last_zip_path)
            else None
        )
        output_folder = (
            self._last_output_folder
            if self._last_output_folder and os.path.isdir(self._last_output_folder)
            else None
        )
        use_zip = zip_path is not None
        use_folder = output_folder is not None

        if not use_zip and not use_folder:
            qt.QMessageBox.information(self, "提示", "没有可用的处理结果进行对比")
            return

        try:
            compare_samples = []
            proc_names = []
            ext = self._out_fmt.lower()

            for fd in samples:
                proc_name_base = os.path.splitext(fd["name"])[0]
                proc_filename = f"{proc_name_base}.{ext}"
                if zip_path:
                    with zipfile.ZipFile(zip_path, "r") as zf:
                        if proc_filename not in zf.namelist():
                            continue
                elif output_folder:
                    proc_path = os.path.join(output_folder, proc_filename)
                    if not os.path.exists(proc_path):
                        continue

                compare_samples.append(fd)
                if output_folder:
                    proc_names.append(os.path.basename(os.path.join(output_folder, proc_filename)))
                else:
                    proc_names.append(f"processed_{fd['name']}")

            if not compare_samples:
                self._log.append_log("未找到处理结果", "err")
                return

            orig_source = LazyFrameSource(
                len(compare_samples),
                lambda index: self._load_sample_frame(compare_samples[index]),
                [fd["name"] for fd in compare_samples],
            )
            proc_source = LazyFrameSource(
                len(compare_samples),
                lambda index: self._load_processed_output_frame(
                    compare_samples[index],
                    ext,
                    bool(use_zip),
                    bool(use_folder),
                ),
                proc_names,
            )

            self._main_window.show_compare_window(
                orig_source,
                proc_source,
                [f["name"] for f in compare_samples],
                proc_names,
            )
            self._prog.setValue(0)
            self._prog_lbl.setText("")
            self._log.append_log(f"对比视图已切换为按帧加载: {len(compare_samples)} 组", "ok")
        except Exception as e:
            self._log.append_log(f"❌ 加载对比数据失败: {e}", "err")


class H5Tab(qt.QWidget):
    """H5 标签页。"""

    def __init__(self, parent, main_window):
        super().__init__(parent)
        self._main_window = main_window
        self._sample_h5: Optional[Dict] = None
        self._bg_h5: Optional[Dict] = None
        self._processed: Optional[np.ndarray] = None
        self._trans_arr: Optional[np.ndarray] = None

        self._trans_src = "unified"
        self._uni_T = 100.0
        self._h5_suffix = ""

        self._root_folder = ""
        self._sample_files: List[str] = []
        self._bg_files: List[str] = []

        # For streaming processing / 流式处理相关
        self._last_h5_path: Optional[str] = None
        self._worker_thread: Optional[qt.QThread] = None
        self._worker: Optional[BackgroundProcessingWorker] = None

        self._init_ui()

    def _init_ui(self):
        layout = qt.QVBoxLayout(self)

        scroll = qt.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(qt.QFrame.NoFrame)
        content = qt.QWidget()
        content_layout = qt.QVBoxLayout(content)

        file_group = qt.QGroupBox("① 选择 H5 文件")
        file_layout = qt.QVBoxLayout(file_group)

        sample_row = qt.QHBoxLayout()
        sample_row.addWidget(qt.QLabel("样品文件:"))
        sample_row.addStretch()
        self._s_folder_btn = qt.QPushButton("📁 文件夹")
        self._s_folder_btn.clicked.connect(lambda: self._pick_h5_folder("sample"))
        self._s_file_btn = qt.QPushButton("📂 文件")
        self._s_file_btn.clicked.connect(lambda: self._pick_h5("sample"))
        sample_row.addWidget(self._s_folder_btn)
        sample_row.addWidget(self._s_file_btn)
        file_layout.addLayout(sample_row)

        self._sample_list = qt.QListWidget()
        self._sample_list.setMaximumHeight(100)
        self._sample_list.setAcceptDrops(True)
        self._sample_list.setDragDropMode(qt.QAbstractItemView.DropOnly)
        self._sample_list.setDefaultDropAction(qt.Qt.CopyAction)
        self._sample_list.installEventFilter(self)
        file_layout.addWidget(self._sample_list)

        bg_row = qt.QHBoxLayout()
        bg_row.addWidget(qt.QLabel("背景文件:"))
        bg_row.addStretch()
        self._b_folder_btn = qt.QPushButton("📁 文件夹")
        self._b_folder_btn.clicked.connect(lambda: self._pick_h5_folder("bg"))
        self._b_file_btn = qt.QPushButton("📂 文件")
        self._b_file_btn.clicked.connect(lambda: self._pick_h5("bg"))
        bg_row.addWidget(self._b_folder_btn)
        bg_row.addWidget(self._b_file_btn)
        file_layout.addLayout(bg_row)

        self._bg_list = qt.QListWidget()
        self._bg_list.setMaximumHeight(100)
        self._bg_list.setAcceptDrops(True)
        self._bg_list.setDragDropMode(qt.QAbstractItemView.DropOnly)
        self._bg_list.setDefaultDropAction(qt.Qt.CopyAction)
        self._bg_list.installEventFilter(self)
        file_layout.addWidget(self._bg_list)

        content_layout.addWidget(file_group)

        trans_group = qt.QGroupBox("② 透过率设置 / Transmission")
        trans_layout = qt.QVBoxLayout(trans_group)

        src_row = qt.QHBoxLayout()
        src_row.addWidget(qt.QLabel("透过率:"))
        self._uni_spin = qt.QDoubleSpinBox()
        self._uni_spin.setRange(0.001, 10000.0)
        self._uni_spin.setValue(100.0)
        self._uni_spin.valueChanged.connect(lambda v: setattr(self, "_uni_T", v))
        src_row.addWidget(self._uni_spin)
        src_row.addWidget(qt.QLabel("%"))
        src_row.addStretch()
        trans_layout.addLayout(src_row)

        content_layout.addWidget(trans_group)

        run_group = qt.QGroupBox("③ 执行 / Run")
        run_layout = qt.QVBoxLayout(run_group)
        run_row = qt.QHBoxLayout()
        self._run_btn = qt.QPushButton("▶ 开始处理 (H5)")
        self._run_btn.clicked.connect(self._run)
        self._cancel_btn = qt.QPushButton("■ 取消")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self._cancel_processing)
        self._h5_pb = qt.QProgressBar()
        self._h5_pb.setRange(0, 0)
        self._h5_pb.setVisible(False)
        self._h5_st = qt.QLabel("")
        run_row.addWidget(self._run_btn)
        run_row.addWidget(self._cancel_btn)
        run_row.addWidget(self._h5_pb)
        run_row.addWidget(self._h5_st)
        run_row.addStretch()
        run_layout.addLayout(run_row)
        content_layout.addWidget(run_group)

        export_group = qt.QGroupBox("④ 导出 / Export")
        export_layout = qt.QVBoxLayout(export_group)
        export_row = qt.QHBoxLayout()
        self._save_btn = qt.QPushButton("💾 保存 H5 结果")
        self._save_btn.clicked.connect(self._save)
        self._compare_btn = qt.QPushButton("🖼 对比预览")
        self._compare_btn.clicked.connect(self._show_compare)
        self._compare_btn.setEnabled(False)
        export_row.addWidget(self._save_btn)
        export_row.addWidget(self._compare_btn)
        export_row.addStretch()
        export_layout.addLayout(export_row)
        content_layout.addWidget(export_group)

        log_group = qt.QGroupBox("日志 / Log")
        log_layout = qt.QVBoxLayout(log_group)
        self._log = LogWidget()
        log_layout.addWidget(self._log)
        content_layout.addWidget(log_group)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _pick_h5(self, role: str):
        path = qt.QFileDialog.getOpenFileName(
            self, f"选择{'样品' if role == 'sample' else '背景'} H5 文件", "", "HDF5 (*.h5 *.hdf5)"
        )[0]
        if path:
            if role == "sample":
                self._add_sample_file(path)
            else:
                self._add_bg_file(path)

    def _pick_h5_folder(self, role: str):
        folder = qt.QFileDialog.getExistingDirectory(
            self, f"选择包含{'样品' if role == 'sample' else '背景'} H5 的文件夹"
        )
        if not folder:
            return

        found = []
        for root, _, files in os.walk(folder):
            for fn in sorted(files):
                low = fn.lower()
                if any(low.endswith(e) for e in H5_EXTS):
                    found.append(os.path.join(root, fn))

        if not found:
            qt.QMessageBox.information(self, "未找到文件", f"文件夹中没有 H5 文件:\n{folder}")
            return

        for p in found:
            if role == "sample":
                self._add_sample_file(p)
            else:
                self._add_bg_file(p)

    def eventFilter(self, obj, event):
        if event.type() == qt.QEvent.DragEnter:
            if obj in (self._bg_list, self._sample_list):
                event.acceptProposedAction()
                return True
        if event.type() == qt.QEvent.Drop:
            if obj == self._bg_list:
                mime = event.mimeData()
                if mime.hasUrls():
                    for url in mime.urls():
                        path = url.toLocalFile()
                        if path.lower().endswith((".h5", ".hdf5")):
                            self._add_bg_file(path)
                event.acceptProposedAction()
                return True
            elif obj == self._sample_list:
                mime = event.mimeData()
                if mime.hasUrls():
                    for url in mime.urls():
                        path = url.toLocalFile()
                        if path.lower().endswith((".h5", ".hdf5")):
                            self._add_sample_file(path)
                event.acceptProposedAction()
                return True
        return super().eventFilter(obj, event)

    def _add_bg_file(self, path: str):
        """添加背景文件到列表。"""
        if path not in self._bg_files:
            self._bg_files.append(path)
            self._bg_list.addItem(os.path.basename(path))
            self._log.append_log(f"添加背景: {os.path.basename(path)}", "ok")

    def _add_sample_file(self, path: str):
        """添加样品文件到列表。"""
        if path not in self._sample_files:
            self._sample_files.append(path)
            self._sample_list.addItem(os.path.basename(path))
            self._log.append_log(f"添加样品: {os.path.basename(path)}", "ok")

    def _get_trans_arr(self, n_frames: int) -> Optional[np.ndarray]:
        return np.full(n_frames, self._uni_T, dtype=np.float64)

    def _build_original_h5_source(self) -> Optional[LazyFrameSource]:
        if not self._sample_files or not self._bg_files:
            return None

        bg_meta = probe_h5_datasets(self._bg_files[0])
        if bg_meta is None:
            return None

        entries: List[Tuple[str, int]] = []
        names: List[str] = []
        bg_shape = bg_meta["shape"]
        bg_ndim = bg_meta["effective_ndim"]
        bg_frames = bg_meta["n_frames"]

        for sample_path in self._sample_files:
            sample_meta = probe_h5_datasets(sample_path)
            if sample_meta is None:
                continue
            if sample_meta["shape"][-2:] != bg_shape[-2:]:
                continue
            if sample_meta["effective_ndim"] != bg_ndim:
                continue
            sample_frames = sample_meta["n_frames"]
            usable_frames = (
                min(sample_frames, bg_frames)
                if (bg_frames != sample_frames and bg_frames != 1)
                else sample_frames
            )
            for frame_idx in range(usable_frames):
                entries.append((sample_path, frame_idx))
                names.append(f"{os.path.basename(sample_path)} [{frame_idx + 1}]")

        if not entries:
            return None

        cache: Dict[str, Dict[str, Any]] = {}

        def loader(index: int) -> np.ndarray:
            sample_path, frame_idx = entries[index]
            if sample_path not in cache:
                loaded = load_h5_stack(sample_path)
                if loaded is None:
                    raise ValueError(f"无法读取 H5 文件: {sample_path}")
                cache[sample_path] = loaded
            sample_h5 = cache[sample_path]
            data = sample_h5["data"]
            if sample_h5["ndim"] == 4:
                return _ensure_float32_frame(data[frame_idx, 0])
            return _ensure_float32_frame(data[frame_idx])

        return LazyFrameSource(len(entries), loader, names)

    def _build_processed_h5_source(self) -> Optional[LazyFrameSource]:
        if not self._last_h5_path or not os.path.exists(self._last_h5_path):
            return None

        with h5py.File(self._last_h5_path, "r") as fh:
            if "data" not in fh:
                return None
            dset = fh["data"]
            if not isinstance(dset, h5py.Dataset):
                return None
            frame_count = int(dset.shape[0])

        def loader(index: int) -> np.ndarray:
            with h5py.File(self._last_h5_path, "r") as fh:
                dset = fh["data"]
                return _ensure_float32_frame(np.asarray(dset[index], dtype=np.float32))

        names = [f"processed_frame_{i + 1}" for i in range(frame_count)]
        return LazyFrameSource(frame_count, loader, names)

    def _set_running_state(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)
        for widget in (
            self._s_folder_btn,
            self._s_file_btn,
            self._b_folder_btn,
            self._b_file_btn,
            self._save_btn,
            self._compare_btn,
        ):
            widget.setEnabled(not running)

    def _cleanup_worker(self):
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait()
            self._worker_thread.deleteLater()
        self._worker_thread = None
        self._worker = None

    def _cancel_processing(self):
        if self._worker is not None:
            self._worker.request_cancel()
            self._cancel_btn.setEnabled(False)
            self._h5_st.setText("正在取消...")
            self._log.append_log("已请求取消，等待当前 H5 文件完成...", "warn")

    def _on_worker_progress(self, value: int, text: str):
        if self._h5_pb.maximum() == 0:
            self._h5_pb.setRange(0, 100)
        self._h5_pb.setValue(value)
        self._h5_st.setText(text)

    def _on_worker_failed(self, message: str):
        self._cleanup_worker()
        self._set_running_state(False)
        self._h5_pb.setVisible(False)
        self._h5_st.setText("处理失败")
        self._compare_btn.setEnabled(False)
        self._log.append_log(f"❌ 处理失败: {message}", "err")

    def _on_processing_finished(self, result: H5WorkerResult):
        self._cleanup_worker()
        self._set_running_state(False)
        self._h5_pb.setVisible(False)
        self._h5_st.setText(f"✓ 完成 {result.processed_count} 个文件")
        self._log.append_log(
            f"流式处理完成: {result.processed_count}/{result.total_count} 个文件，已保存到 {result.output_path}",
            "ok",
        )
        self._last_h5_path = result.output_path
        self._compare_btn.setEnabled(result.processed_count > 0)
        if result.processed_count > 0:
            self._show_compare()
            self._log.append_log("已自动切换到对比视图，当前仅首帧已渲染，可继续逐帧查看。", "ok")

    def _on_processing_cancelled(self, result: H5WorkerResult):
        self._cleanup_worker()
        self._set_running_state(False)
        self._h5_pb.setVisible(False)
        self._h5_st.setText("已取消")
        self._log.append_log(
            f"已取消: {result.processed_count}/{result.total_count} 个文件已写入 {result.output_path}",
            "warn",
        )
        self._last_h5_path = result.output_path if result.processed_count > 0 else None
        self._compare_btn.setEnabled(result.processed_count > 0)
        if result.processed_count > 0:
            self._show_compare()
            self._log.append_log("已自动切换到对比视图，当前仅首帧已渲染，可继续逐帧查看。", "ok")

    def _run(self):
        if not self._sample_files:
            qt.QMessageBox.warning(self, "文件缺失", "请先加载样品 H5 文件")
            return
        if not self._bg_files:
            qt.QMessageBox.warning(self, "文件缺失", "请先加载背景 H5 文件")
            return

        bg_path = self._bg_files[0]
        self._log.append_log(f"使用背景: {os.path.basename(bg_path)}", "info")

        sp = qt.QFileDialog.getSaveFileName(
            self, "保存 H5 结果", "processed_results.h5", "HDF5 (*.h5)"
        )[0]
        if not sp:
            return

        self._h5_pb.setRange(0, 100)
        self._h5_pb.setValue(0)
        self._h5_pb.setVisible(True)
        self._h5_st.setText("处理中...")
        self._last_h5_path = None
        self._set_running_state(True)
        self._compare_btn.setEnabled(False)
        self._log.append_log("开始流式处理（后台）...", "info")

        sample_files = list(self._sample_files)
        self._worker_thread = qt.QThread(self)
        self._worker = BackgroundProcessingWorker(
            lambda worker: _run_h5_stream_job(worker, sample_files, bg_path, sp, self._uni_T)
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_worker_progress)
        self._worker.log.connect(self._log.append_log)
        self._worker.finished.connect(self._on_processing_finished)
        self._worker.cancelled.connect(self._on_processing_cancelled)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker_thread.start()

    def _save(self):
        if self._last_h5_path and os.path.exists(self._last_h5_path):
            qt.QMessageBox.information(self, "已保存", f"处理结果已保存到:\n{self._last_h5_path}")
            return
        if self._processed is None:
            qt.QMessageBox.information(self, "提示", "请先执行处理")
            return
        sp = qt.QFileDialog.getSaveFileName(
            self, "保存 H5 结果", "processed_results.h5", "HDF5 (*.h5)"
        )[0]
        if not sp:
            return
        self._log.append_log(f"保存功能待实现: {sp}", "info")

    def _show_compare(self):
        if self._processed is not None and hasattr(self, "_orig_data"):
            self._main_window.show_compare_window(
                self._processed,
                self._orig_data,
                [f"frame_{i}" for i in range(self._processed.shape[0])],
                [f"frame_{i}" for i in range(self._orig_data.shape[0])],
            )
            return

        orig_source = self._build_original_h5_source()
        proc_source = self._build_processed_h5_source()
        if orig_source is None or proc_source is None:
            qt.QMessageBox.information(self, "提示", "没有可用的 H5 对比结果")
            return

        self._main_window.show_compare_window(
            orig_source,
            proc_source,
            [orig_source.get_name(i) for i in range(orig_source.frame_count)],
            [proc_source.get_name(i) for i in range(proc_source.frame_count)],
        )


class H5PickDialog(qt.QDialog):
    """H5 文件夹多文件选择对话框。"""

    def __init__(self, parent, paths: List[str], role: str, cb):
        super().__init__(parent)
        self._paths = paths
        self._role = role
        self._cb = cb

        self.setWindowTitle("选择 H5 文件")
        self.setMinimumWidth(500)

        layout = qt.QVBoxLayout(self)

        layout.addWidget(qt.QLabel(f"文件夹中找到 {len(paths)} 个 H5 文件，请选择一个:"))

        self._list = qt.QListWidget()
        self._list.addItems(["  " + os.path.relpath(p) for p in paths])
        self._list.setCurrentRow(0)
        layout.addWidget(self._list)

        btn_row = qt.QHBoxLayout()
        ok_btn = qt.QPushButton("✓ 确认选择")
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = qt.QPushButton("✕ 取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _on_ok(self):
        row = self._list.currentRow()
        if row >= 0:
            self._cb(self._paths[row], self._role)
            self.accept()


class CompareWindow(qt.QMainWindow):
    """堆叠对比窗口 - 左右同步播放对比。"""

    def __init__(self, parent, orig_stack, proc_stack, names_orig, names_proc):
        super().__init__(parent)
        self.setWindowTitle("全部对比")
        self.setMinimumSize(1200, 700)

        self._orig = orig_stack.astype(np.float32)
        self._proc = proc_stack.astype(np.float32)
        self._names_orig = names_orig
        self._names_proc = names_proc
        self._n = orig_stack.shape[0]
        self._current = 0

        central = qt.QWidget()
        self.setCentralWidget(central)
        layout = qt.QHBoxLayout(central)

        self._compare = CompareImages()
        plot = self._compare.getPlot()
        if hasattr(plot, "setKeepDataAspectRatio"):
            plot.setKeepDataAspectRatio(True)
        layout.addWidget(self._compare)

        control = qt.QWidget()
        control.setMaximumWidth(200)
        ctrl_layout = qt.QVBoxLayout(control)

        ctrl_layout.addWidget(qt.QLabel(f"<b>全部对比 ({self._n} 帧)</b>"))

        slider_group = qt.QGroupBox("帧选择")
        slider_layout = qt.QVBoxLayout(slider_group)
        self._slider = qt.QSlider(qt.Qt.Horizontal)
        self._slider.setRange(0, self._n - 1)
        self._slider.setValue(0)
        self._slider.valueChanged.connect(self._on_frame_changed)
        slider_layout.addWidget(self._slider)
        self._frame_lbl = qt.QLabel("帧 0 / 0")
        slider_layout.addWidget(self._frame_lbl)
        ctrl_layout.addWidget(slider_group)

        play_group = qt.QGroupBox("播放控制")
        play_layout = qt.QHBoxLayout(play_group)
        self._prev_btn = qt.QPushButton("◀")
        self._prev_btn.clicked.connect(self._prev_frame)
        self._play_btn = qt.QPushButton("▶")
        self._play_btn.clicked.connect(self._toggle_play)
        self._next_btn = qt.QPushButton("▶")
        self._next_btn.clicked.connect(self._next_frame)
        play_layout.addWidget(self._prev_btn)
        play_layout.addWidget(self._play_btn)
        play_layout.addWidget(self._next_btn)
        ctrl_layout.addWidget(play_group)

        ctrl_layout.addStretch()

        close_btn = qt.QPushButton("✕ 关闭")
        close_btn.clicked.connect(self.close)
        ctrl_layout.addWidget(close_btn)

        layout.addWidget(control)

        self._timer = qt.QTimer()
        self._timer.timeout.connect(self._next_frame)
        self._playing = False

        self._update_display()

    def _on_frame_changed(self, idx):
        self._current = idx
        self._frame_lbl.setText(f"帧 {idx + 1} / {self._n}")
        self._update_display()

    def _update_display(self):
        if self._current >= self._n:
            return
        left = self._orig[self._current]
        right = self._proc[self._current]
        self._compare.setData(left, right)

    def _prev_frame(self):
        self._current = (self._current - 1) % self._n
        self._slider.setValue(self._current)

    def _next_frame(self):
        self._current = (self._current + 1) % self._n
        self._slider.setValue(self._current)

    def _toggle_play(self):
        if self._playing:
            self._timer.stop()
            self._play_btn.setText("▶")
        else:
            self._timer.start(500)
            self._play_btn.setText("⏸")
        self._playing = not self._playing


class BgSubMainWindow(qt.QMainWindow):
    """BGsub 应用主窗口。"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("BGsub — 背景扣除工具")
        self.setMinimumSize(1200, 800)

        self._stack_source: Optional[LazyFrameSource] = None
        self._orig_source: Optional[LazyFrameSource] = None
        self._proc_source: Optional[LazyFrameSource] = None
        self._names_orig: List[str] = []
        self._names_proc: List[str] = []

        self._setup_ui()
        self._create_actions()
        self._create_menu()

    def _setup_ui(self):
        # 可拖拽分割面板 / User-resizable splitter panel
        self._main_splitter = qt.QSplitter(qt.Qt.Horizontal)
        self.setCentralWidget(self._main_splitter)

        left_panel = qt.QWidget()
        left_layout = qt.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self._tab_widget = qt.QTabWidget()
        self._tab_widget.addTab(TiffTab(self._tab_widget, self), "📷 TIFF / EDF 模式")
        self._tab_widget.addTab(H5Tab(self._tab_widget, self), "🗃 H5 模式（堆叠数据）")
        left_layout.addWidget(self._tab_widget)

        view_control = qt.QGroupBox("视图控制 / View Control")
        view_layout = qt.QHBoxLayout(view_control)
        self._view_group = qt.QButtonGroup()
        self._view_stack_rb = qt.QRadioButton("🖼 堆叠视图")
        self._view_stack_rb.setChecked(True)
        self._view_stack_rb.clicked.connect(lambda: self._set_view_mode("stack"))
        self._view_image_rb = qt.QRadioButton("📷 图像视图")
        self._view_image_rb.clicked.connect(lambda: self._set_view_mode("image"))
        self._view_compare_rb = qt.QRadioButton("🔄 对比视图")
        self._view_compare_rb.clicked.connect(lambda: self._set_view_mode("compare"))
        self._view_group.addButton(self._view_stack_rb)
        self._view_group.addButton(self._view_image_rb)
        self._view_group.addButton(self._view_compare_rb)
        view_layout.addWidget(self._view_stack_rb)
        view_layout.addWidget(self._view_image_rb)
        view_layout.addWidget(self._view_compare_rb)
        view_layout.addStretch()
        left_layout.addWidget(view_control)

        self._main_splitter.addWidget(left_panel)

        right_panel = qt.QWidget()
        right_layout = qt.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self._view_stack = qt.QStackedWidget()

        self._stack_viewer = StackViewMainWindow()
        self._stack_viewer.setColormap("jet")
        self._stack_viewer.setKeepDataAspectRatio(True)
        self._view_stack.addWidget(self._stack_viewer)

        self._image_view = ImageView()
        self._image_view.setColormap("jet", normalization="linear")
        self._image_view.setKeepDataAspectRatio(True)
        self._image_view.setSideHistogramDisplayed(False)
        self._view_stack.addWidget(self._image_view)

        # 使用 silx CompareImages 实现交互式对比（滑动/混合/差异）
        # Interactive compare via silx CompareImages (slide/blend/difference)
        self._compare_container = qt.QWidget()
        compare_vlayout = qt.QVBoxLayout(self._compare_container)
        compare_vlayout.setContentsMargins(0, 0, 0, 0)
        compare_vlayout.setSpacing(2)

        compare_header = qt.QHBoxLayout()
        self._orig_label = qt.QLabel("原始")
        self._orig_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        compare_header.addWidget(self._orig_label)
        compare_header.addStretch()
        self._proc_label = qt.QLabel("处理后")
        self._proc_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        compare_header.addWidget(self._proc_label)
        compare_vlayout.addLayout(compare_header)

        self._compare_widget = CompareImages()
        _cp = self._compare_widget.getPlot()
        if hasattr(_cp, "setKeepDataAspectRatio"):
            _cp.setKeepDataAspectRatio(True)
        compare_vlayout.addWidget(self._compare_widget)
        self._view_stack.addWidget(self._compare_container)

        right_layout.addWidget(self._view_stack, 3)

        self._compare_control = qt.QWidget()
        self._compare_control.setMaximumHeight(80)
        compare_layout = qt.QHBoxLayout(self._compare_control)

        self._compare_slider = qt.QSlider(qt.Qt.Horizontal)
        self._compare_slider.setRange(0, 0)
        self._compare_slider.setValue(0)
        self._compare_slider.valueChanged.connect(self._on_compare_slider_changed)
        compare_layout.addWidget(self._compare_slider)

        self._compare_label = qt.QLabel("帧 0 / 0")
        compare_layout.addWidget(self._compare_label)

        self._prev_frame_btn = qt.QPushButton("◀")
        self._prev_frame_btn.setMaximumWidth(40)
        self._prev_frame_btn.clicked.connect(self._prev_compare_frame)
        compare_layout.addWidget(self._prev_frame_btn)

        self._play_btn = qt.QPushButton("▶")
        self._play_btn.setMaximumWidth(40)
        self._play_btn.clicked.connect(self._toggle_compare_play)
        compare_layout.addWidget(self._play_btn)

        self._next_frame_btn = qt.QPushButton("▶")
        self._next_frame_btn.setMaximumWidth(40)
        self._next_frame_btn.clicked.connect(self._next_compare_frame)
        compare_layout.addWidget(self._next_frame_btn)

        self._restore_btn = qt.QPushButton("恢复视图")
        self._restore_btn.clicked.connect(self._restore_view)
        compare_layout.addWidget(self._restore_btn)

        compare_layout.addWidget(qt.QLabel("模式:"))
        self._compare_mode_combo = qt.QComboBox()
        self._compare_mode_combo.addItems(["滑动", "混合", "差异"])
        self._compare_mode_combo.currentIndexChanged.connect(self._on_compare_mode_changed)
        compare_layout.addWidget(self._compare_mode_combo)

        self._compare_control.setVisible(False)
        right_layout.addWidget(self._compare_control)

        self._stack_control = qt.QWidget()
        self._stack_control.setMaximumHeight(80)
        stack_layout = qt.QHBoxLayout(self._stack_control)

        self._stack_slider = qt.QSlider(qt.Qt.Horizontal)
        self._stack_slider.setRange(0, 0)
        self._stack_slider.setValue(0)
        self._stack_slider.valueChanged.connect(self._on_stack_slider_changed)
        stack_layout.addWidget(self._stack_slider)

        self._stack_label = qt.QLabel("帧 0 / 0")
        stack_layout.addWidget(self._stack_label)

        self._stack_prev_btn = qt.QPushButton("◀")
        self._stack_prev_btn.setMaximumWidth(40)
        self._stack_prev_btn.clicked.connect(self._prev_stack_frame)
        stack_layout.addWidget(self._stack_prev_btn)

        self._stack_play_btn = qt.QPushButton("▶")
        self._stack_play_btn.setMaximumWidth(40)
        self._stack_play_btn.clicked.connect(self._toggle_stack_play)
        stack_layout.addWidget(self._stack_play_btn)

        self._stack_next_btn = qt.QPushButton("▶")
        self._stack_next_btn.setMaximumWidth(40)
        self._stack_next_btn.clicked.connect(self._next_stack_frame)
        stack_layout.addWidget(self._stack_next_btn)

        self._stack_control.setVisible(False)
        right_layout.addWidget(self._stack_control)

        self._main_splitter.addWidget(right_panel)
        # 初始比例约 1:3 / Initial ratio ~1:3
        self._main_splitter.setStretchFactor(0, 1)
        self._main_splitter.setStretchFactor(1, 3)
        self._main_splitter.setSizes([350, 850])

        self._view_stack.setCurrentWidget(self._image_view)

        self._tab_widget.currentChanged.connect(self._on_tab_changed)

        self._compare_timer = qt.QTimer()
        self._compare_timer.timeout.connect(self._next_compare_frame)
        self._compare_playing = False

        self._stack_timer = qt.QTimer()
        self._stack_timer.timeout.connect(self._next_stack_frame)
        self._stack_playing = False

        self._stack_source = None
        self._orig_source = None
        self._proc_source = None
        self._stack_current = 0
        self._stack_n = 0
        self._compare_current = 0
        self._compare_n = 0

        status_bar = qt.QStatusBar(self)
        self.setStatusBar(status_bar)
        self._status_label = qt.QLabel("就绪")
        status_bar.addWidget(self._status_label)

        self._init_colormap_dialog()

    def _init_colormap_dialog(self):
        """初始化调色板对话框（不自动显示）。"""
        self._cmap_dialog = ColormapDialog(parent=self)
        self._cmap_dialog.setWindowTitle("调色板设置 / Colormap Settings")
        dummy_data = np.array([[0, 100]], dtype=np.float64)
        self._cmap_dialog.setData(dummy_data)
        self._cmap_dialog.setDataRange(minimum=0, maximum=100)
        cmap = Colormap(name="jet", normalization="linear")
        self._cmap_dialog.setColormap(cmap)
        self._cmap_dialog.accepted.connect(self._apply_colormap)

    def _show_colormap_dialog(self):
        """显示调色板对话框。"""
        if hasattr(self, "_cmap_dialog") and self._cmap_dialog is not None:
            self._cmap_dialog.show()
            self._cmap_dialog.raise_()
            self._cmap_dialog.activateWindow()

    def _apply_colormap(self):
        """应用调色板对话框的设置到所有视图。"""
        cmap = self._cmap_dialog.getColormap()
        self._stack_viewer.setColormap(cmap)
        self._image_view.setColormap(cmap)

    def _set_view_mode(self, mode):
        if mode == "stack":
            self._view_stack.setCurrentWidget(self._image_view)
            self._compare_control.setVisible(False)
            self._stack_control.setVisible(self._stack_n > 0)
            self._update_stack_display()
        elif mode == "image":
            self._view_stack.setCurrentWidget(self._image_view)
            self._compare_control.setVisible(False)
            self._stack_control.setVisible(False)
        elif mode == "compare":
            self._view_stack.setCurrentWidget(self._compare_container)
            self._stack_control.setVisible(False)
            if self._compare_n > 0:
                self._compare_control.setVisible(True)

    def _on_stack_slider_changed(self, idx):
        self._stack_current = idx
        self._stack_label.setText(f"帧 {idx + 1} / {self._stack_n}")
        self._update_stack_display()

    def _update_stack_display(self):
        if self._stack_current >= self._stack_n or self._stack_source is None:
            return
        frame = self._stack_source.get_frame(self._stack_current)
        self._image_view.setImage(frame)
        name = self._stack_source.get_name(self._stack_current)
        self._image_view.setWindowTitle(name)
        self._status_label.setText(f"堆叠浏览: {name}")

    def _prev_stack_frame(self):
        if self._stack_n <= 0:
            return
        self._stack_current = (self._stack_current - 1) % self._stack_n
        self._stack_slider.setValue(self._stack_current)

    def _next_stack_frame(self):
        if self._stack_n <= 0:
            return
        self._stack_current = (self._stack_current + 1) % self._stack_n
        self._stack_slider.setValue(self._stack_current)

    def _toggle_stack_play(self):
        if self._stack_n <= 0:
            return
        if self._stack_playing:
            self._stack_timer.stop()
            self._stack_play_btn.setText("▶")
        else:
            self._stack_timer.start(500)
            self._stack_play_btn.setText("⏸")
        self._stack_playing = not self._stack_playing

    def _on_compare_slider_changed(self, idx):
        self._compare_current = idx
        self._compare_label.setText(f"帧 {idx + 1} / {self._compare_n}")
        self._update_compare_display()

    def _update_compare_display(self):
        if self._compare_current >= self._compare_n:
            return
        if self._orig_source is None or self._proc_source is None:
            return
        left = self._orig_source.get_frame(self._compare_current)
        right = self._proc_source.get_frame(self._compare_current)
        self._compare_widget.setData(left, right)
        self._orig_label.setText(self._orig_source.get_name(self._compare_current))
        self._proc_label.setText(self._proc_source.get_name(self._compare_current))

    def _prev_compare_frame(self):
        if self._compare_n <= 0:
            return
        self._compare_current = (self._compare_current - 1) % self._compare_n
        self._compare_slider.setValue(self._compare_current)

    def _next_compare_frame(self):
        if self._compare_n <= 0:
            return
        self._compare_current = (self._compare_current + 1) % self._compare_n
        self._compare_slider.setValue(self._compare_current)

    def _toggle_compare_play(self):
        if self._compare_n <= 0:
            return
        if self._compare_playing:
            self._compare_timer.stop()
            self._play_btn.setText("▶")
        else:
            self._compare_timer.start(500)
            self._play_btn.setText("⏸")
        self._compare_playing = not self._compare_playing

    def _on_compare_mode_changed(self, index: int):
        viz_cls = getattr(CompareImages, "VisualizationMode", None)
        mode_map = [
            getattr(viz_cls, "SLIDE", None),
            getattr(viz_cls, "BLEND", None),
            getattr(viz_cls, "DIFFERENCE", None),
        ]
        if 0 <= index < len(mode_map) and mode_map[index] is not None:
            self._compare_widget.setVisualizationMode(mode_map[index])

    def _restore_view(self):
        self._compare_timer.stop()
        self._compare_playing = False
        self._play_btn.setText("▶")
        self._set_view_mode("stack")
        self._view_stack_rb.setChecked(True)
        self._compare_control.setVisible(False)

    def _create_actions(self):
        self._action_open = qt.QAction("打开文件...", self)
        self._action_open.setShortcut("Ctrl+O")
        self._action_open.triggered.connect(self._open_file)

        self._action_save = qt.QAction("保存结果...", self)
        self._action_save.setShortcut("Ctrl+S")
        self._action_save.triggered.connect(self._save_result)

        self._action_exit = qt.QAction("退出", self)
        self._action_exit.setShortcut("Ctrl+Q")
        self._action_exit.triggered.connect(self.close)

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件 / File")
        file_menu.addAction(self._action_open)
        file_menu.addAction(self._action_save)
        file_menu.addSeparator()
        file_menu.addAction(self._action_exit)

    def _open_file(self):
        self._tab_widget.currentWidget()._pick_files()

    def _save_result(self):
        self._tab_widget.currentWidget()._save()

    def _on_tab_changed(self, index: int):
        pass

    def clear_views(self):
        """清空所有视图。"""
        self._stack_timer.stop()
        self._compare_timer.stop()
        self._stack_playing = False
        self._compare_playing = False
        self._stack_play_btn.setText("▶")
        self._play_btn.setText("▶")
        self._stack_viewer.clear()
        self._image_view.clear()
        self._stack_source = None
        self._orig_source = None
        self._proc_source = None
        self._stack_current = 0
        self._stack_n = 0
        self._compare_current = 0
        self._compare_n = 0
        self._stack_control.setVisible(False)
        self._compare_control.setVisible(False)

    def show_in_image_view(self, data: np.ndarray, name: str = ""):
        """在图像视图中显示数据。"""
        if name:
            self._image_view.setWindowTitle(name)
        self._image_view.setImage(_ensure_float32_frame(data))
        self._view_stack.setCurrentWidget(self._image_view)
        self._view_image_rb.setChecked(True)
        self._stack_control.setVisible(False)
        self._compare_control.setVisible(False)

    def set_stack_data(self, data: np.ndarray, names: Optional[List[str]] = None):
        """在堆叠视图中显示数据。"""
        if data is None:
            return
        self.set_stack_source(LazyFrameSource.from_array(data, names))

    def set_stack_source(self, source: LazyFrameSource):
        if source is None or source.frame_count <= 0:
            return
        self._stack_timer.stop()
        self._stack_playing = False
        self._stack_play_btn.setText("▶")
        self._stack_source = source
        self._stack_n = source.frame_count
        self._stack_current = 0
        self._stack_slider.setRange(0, self._stack_n - 1)
        self._stack_slider.setValue(0)
        self._stack_label.setText(f"帧 1 / {self._stack_n}")
        self._set_view_mode("stack")
        self._view_stack_rb.setChecked(True)

    def set_original_stack(self, data: np.ndarray, names: Optional[List[str]] = None):
        """保存原始数据堆叠用于对比。"""
        if data is None:
            return
        self._orig_source = LazyFrameSource.from_array(data, names)
        self._names_orig = names or [f"frame_{i}" for i in range(self._orig_source.frame_count)]

    def show_compare_window(self, orig_stack, proc_stack, names_orig, names_proc):
        """显示对比视图。"""
        if orig_stack is None or proc_stack is None:
            return

        self._orig_source = (
            orig_stack
            if isinstance(orig_stack, LazyFrameSource)
            else LazyFrameSource.from_array(orig_stack, names_orig)
        )
        self._proc_source = (
            proc_stack
            if isinstance(proc_stack, LazyFrameSource)
            else LazyFrameSource.from_array(proc_stack, names_proc)
        )
        self._compare_n = min(self._orig_source.frame_count, self._proc_source.frame_count)
        if self._compare_n <= 0:
            return
        self._compare_current = 0
        self._compare_slider.setRange(0, self._compare_n - 1)
        self._compare_slider.setValue(0)
        self._compare_label.setText(f"帧 1 / {self._compare_n}")
        self._names_orig = names_orig or [f"frame_{i}" for i in range(self._compare_n)]
        self._names_proc = names_proc or [f"frame_{i}" for i in range(self._compare_n)]

        self._update_compare_display()
        self._view_stack.setCurrentWidget(self._compare_container)
        self._view_compare_rb.setChecked(True)
        self._compare_control.setVisible(True)


def main():
    """BGsub 应用程序入口。"""
    # if not HAS_SILX:
    #     print("错误: 需要安装 silx 库来运行 GUI。请执行: pip install silx")
    #     import traceback
    #     traceback.print_exc()
    #     sys.exit(1)
    # if not HAS_FABIO:
    #     print("错误: 需要安装 fabio 库来加载图像。请执行: pip install fabio")
    #     sys.exit(1)

    app = qt.QApplication(sys.argv)
    app.setApplicationName("BGsub")
    app.setOrganizationName("WAXS-SAXS Manager")

    window = BgSubMainWindow()
    window.show()

    sys.exit(app.exec_())


# if __name__ == "__main__":
#     main()
