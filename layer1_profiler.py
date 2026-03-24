"""
Layer 1 — cheap text profiling: regex/heuristics → keywords, category, surface hints.
Keyword lists mix EN/TR spellings so multilingual PDFs still match.
"""

import re


# (pattern, weight) — higher weight = stronger signal for this category
PROBLEM_CATEGORIES = {
    # --- Surface integral types ---
    "scalar_surface_integral": [
        (r"∬.*dS", 3), (r"\\iint.*dS", 3), (r"surface integral", 3),
        (r"find the area", 1), (r"evaluate.*surface", 2),
    ],
    "flux_integral": [
        (r"flux", 3), (r"F\s*[·⋅]\s*dS", 3), (r"F\s*\\cdot\s*dS", 3),
        (r"across.*surface", 2), (r"through.*surface", 2),
    ],
    "divergence_theorem": [
        (r"divergence", 3), (r"Gauss", 3), (r"\\nabla\s*\\cdot", 3),
        (r"div\s*F", 3), (r"outward flux.*closed", 2),
    ],
    "stokes_theorem": [
        (r"Stokes", 3), (r"curl", 2), (r"\\nabla\s*\\times", 3),
        (r"line integral.*surface", 2), (r"circulation", 2),
    ],
    # --- General math types ---
    "indefinite_integral": [
        (r"\\int(?!\w).*dx", 2), (r"∫.*dx", 2), (r"∫.*dy", 2),
        (r"∫.*dt", 2), (r"∫.*du", 2),
        (r"antiderivative", 3), (r"belirsiz integral", 3),
    ],
    "definite_integral": [
        (r"\\int_", 2), (r"∫_", 2), (r"\\int\\limits", 2),
        (r"evaluate.*integral", 2), (r"belirli integral", 3),
    ],
    "double_integral": [
        (r"\\iint", 3), (r"∬", 3), (r"double integral", 3), (r"çift integral", 3),
    ],
    "triple_integral": [
        (r"\\iiint", 3), (r"∭", 3), (r"triple integral", 3), (r"üçlü integral", 3),
    ],
    "derivative": [
        (r"\\frac\{d", 2), (r"\\frac\{\\partial", 2), (r"derivative", 3),
        (r"differentiate", 3), (r"türev", 3), (r"d/dx", 2),
    ],
    "limit": [
        (r"\\lim", 3), (r"limit", 2), (r"→", 1), (r"\\to", 1), (r"\\rightarrow", 1),
    ],
    "series": [
        (r"\\sum", 2), (r"∑", 2), (r"series", 2), (r"convergence", 2),
        (r"Taylor", 3), (r"Maclaurin", 3), (r"power series", 3), (r"seri", 2),
    ],
    "differential_equation": [
        (r"differential equation", 3), (r"ODE", 3), (r"PDE", 3),
        (r"y'", 1), (r"y''", 2), (r"diferansiyel denklem", 3),
    ],
    "linear_algebra": [
        (r"matrix", 2), (r"matris", 2), (r"eigenvalue", 3), (r"determinant", 2),
        (r"\\det", 2), (r"rank", 1), (r"eigenvector", 3),
    ],
    "equation": [
        (r"solve.*equation", 2), (r"find.*x", 1), (r"çöz", 2),
        (r"roots", 2), (r"kök", 2), (r"denklem", 2),
    ],
}

SURFACE_TYPES = {
    "sphere": [r"sphere", r"x\^?2\s*\+\s*y\^?2\s*\+\s*z\^?2\s*=", r"kure"],
    "paraboloid": [r"paraboloid", r"z\s*=\s*x\^?2\s*\+\s*y\^?2", r"z\s*=\s*[0-9].*-.*x\^?2"],
    "cylinder": [r"cylinder", r"x\^?2\s*\+\s*y\^?2\s*=", r"silindir"],
    "cone": [r"cone", r"z\s*=\s*\\?sqrt", r"koni"],
    "plane": [r"plane", r"\d+x\s*\+\s*\d+y\s*\+\s*\d+z\s*=", r"duzlem"],
    "hemisphere": [r"hemisphere", r"upper half", r"ust yarikure"],
    "torus": [r"torus"],
}

MATH_KEYWORDS_POOL = [
    ("Surface Integral", r"surface integral|∬|\\iint"),
    ("Flux", r"flux|akı"),
    ("Divergence Theorem", r"divergence|Gauss"),
    ("Stokes Theorem", r"Stokes|curl"),
    ("Parametrization", r"parametri[zs]"),
    ("Cross Product", r"cross product|×|\\times"),
    ("Normal Vector", r"normal vector|unit normal"),
    ("Partial Derivatives", r"partial|∂|\\partial"),
    ("Double Integral", r"double integral|\\iint"),
    ("Triple Integral", r"triple integral|\\iiint"),
    ("Polar Coordinates", r"polar|r\s*dr\s*d.*θ"),
    ("Spherical Coordinates", r"spherical|\\phi.*\\theta"),
    ("Cylindrical Coordinates", r"cylindrical"),
    ("Vector Field", r"vector field|F\s*=|F\s*\("),
    ("Scalar Field", r"scalar field|f\s*\(x"),
    ("Orientation", r"orientation|outward|inward"),
    ("Closed Surface", r"closed surface|kapalı yüzey"),
]


class Layer1_Profiler:
    """Regex-heavy tagging; `latex_text` helps when native PDF text is empty."""

    def profile(self, fname: str, metadata: dict, raw_text: str,
                latex_text: str = "") -> dict:
        combined = raw_text + "\n" + latex_text
        combined_lower = combined.lower()

        keywords = self._extract_keywords(combined)
        category, secondary_categories = self._classify_problem(combined_lower)
        surface = self._detect_surface_type(combined_lower)
        summary = self._generate_summary(fname, category, surface, keywords, metadata)

        domain = self._get_domain(category)

        return {
            "keywords": keywords,
            "category": category,
            "secondary_categories": secondary_categories,
            "domain": domain,
            "surface_type": surface,
            "summary": summary,
            "pages_count": metadata.get("pages", 0),
            "file_size_bytes": int(metadata.get("file_size_kb", 0) * 1024),
            "author": metadata.get("author", ""),
            "producer": metadata.get("producer", ""),
            "creator": metadata.get("creator", ""),
            "creation_date": metadata.get("creation_date", ""),
            "modification_date": metadata.get("modification_date", ""),
        }

    SURFACE_CATEGORIES = {
        "scalar_surface_integral", "flux_integral",
        "divergence_theorem", "stokes_theorem",
    }

    def _get_domain(self, category: str) -> str:
        if category in self.SURFACE_CATEGORIES:
            return "surface_integral"
        if category == "unknown":
            return "general_math"
        return "general_math"

    def _extract_keywords(self, text: str) -> list[str]:
        keywords = []
        for kw, pattern in MATH_KEYWORDS_POOL:
            if re.search(pattern, text, re.IGNORECASE):
                keywords.append(kw)
        return keywords

    def _classify_problem(self, text_lower: str) -> tuple[str, list[str]]:
        """Return (primary_category, secondary_categories_list)."""
        scores = {}
        for cat, weighted_patterns in PROBLEM_CATEGORIES.items():
            total = sum(w for p, w in weighted_patterns if re.search(p, text_lower))
            if total > 0:
                scores[cat] = total

        if not scores:
            return "unknown", []

        ranked = sorted(scores, key=scores.get, reverse=True)
        primary = ranked[0]
        top_score = scores[primary]
        secondary = [c for c in ranked[1:] if scores[c] >= top_score * 0.5]
        return primary, secondary

    def _detect_surface_type(self, text_lower: str) -> str:
        for stype, patterns in SURFACE_TYPES.items():
            for p in patterns:
                if re.search(p, text_lower):
                    return stype
        return "unknown"

    def _generate_summary(self, fname: str, category: str, surface: str,
                          keywords: list[str], metadata: dict) -> str:
        cat_names = {
            "scalar_surface_integral": "scalar surface integral",
            "flux_integral": "flux (vector surface) integral",
            "divergence_theorem": "volume integral via Divergence Theorem",
            "stokes_theorem": "line/surface integral via Stokes' Theorem",
            "unknown": "surface integral problem",
        }
        cat_name = cat_names.get(category, category)

        surf_phrase = f" over a {surface}" if surface != "unknown" else ""
        kw_short = ", ".join(keywords[:5]) if keywords else "surface integral"

        pages = metadata.get("pages", "?")
        return (
            f"Problem {fname} is a {cat_name}{surf_phrase}. "
            f"The PDF has {pages} page(s) and involves: {kw_short}."
        )
