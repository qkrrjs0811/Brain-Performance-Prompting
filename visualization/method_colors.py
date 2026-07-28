"""
시각화 전역에서 동일 방법론에 동일 색을 쓰기 위한 팔레트.

기준: visualization/cost_comparison.py 버블 차트의
`methods` / `colors` 순서와 hex 값을 단일 소스로 둔다.
"""

from __future__ import annotations

from typing import Dict, Tuple

# cost_comparison.py 와 동일 순서·값
METHOD_ORDER_DISPLAY: Tuple[str, ...] = (
    "Standard",
    "CoT",
    "Self Refine",
    "SPP",
    "BPP",
)
METHOD_COLORS_HEX: Tuple[str, ...] = (
    "#8DB7D8",  # Standard
    "#F2BC8E",  # CoT
    "#D98989",  # Self Refine
    "#E8A35A",  # SPP
    "#A8D5BA",  # BPP
)

METHOD_COLORS_DISPLAY: Dict[str, str] = dict(zip(METHOD_ORDER_DISPLAY, METHOD_COLORS_HEX))

METHOD_COLORS_SLUG: Dict[str, str] = {
    "standard": METHOD_COLORS_DISPLAY["Standard"],
    "cot": METHOD_COLORS_DISPLAY["CoT"],
    "self_refine": METHOD_COLORS_DISPLAY["Self Refine"],
    "spp": METHOD_COLORS_DISPLAY["SPP"],
    "bpp": METHOD_COLORS_DISPLAY["BPP"],
}

# cost_comparison.py 에서 BPP 강조용 마커 테두리
BPP_MARKER_EDGE = "#4E7F45"

# CoT 점만 어둡게 처리할 때 (cost_comparison 전용 시각 효과)
COT_SCATTER_DIMMED = "#1A2336"
COT_LEGEND_DIMMED = "#B9BEC7"

# dynamic_static_bpp.py: Macro/Meso/Micro는 전용 파스텔 톤(첨부 레퍼런스와 동일 계열),
# 막대 "BPP"만 canonical BPP 색(METHOD_COLORS_DISPLAY["BPP"]).
SCALED_BPP_VARIANT_BAR_COLORS: Dict[str, str] = {
    "Macro-BPP": "#8EAFD1",  # pastel blue
    "Meso-BPP": "#EBB68E",  # pastel peach
    "Micro-BPP": "#D98A8A",  # pastel coral / red family
    "BPP": METHOD_COLORS_DISPLAY["BPP"],
}
