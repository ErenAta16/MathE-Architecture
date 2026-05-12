"""
Controlled mathematics taxonomy used by the web UI and Layer 1b.

Mirrors the MathE question schema (Topic -> Subtopic -> Keywords) so the
pipeline can expose a consistent vocabulary to the user instead of raw
regex labels. Keep the lists alphabetically sorted within each bucket so
diffs stay readable when new items are added.

The matching rules are intentionally simple (case-insensitive regex
patterns). They are good enough for the short problem statements produced
by Layer 0/3; for anything more ambitious a proper classifier could be
plugged behind the same ``classify_taxonomy`` call.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# For each subtopic we store:
#   * display name (what the UI shows)
#   * a list of regex patterns that, when any matches, promote this subtopic
#   * the allowed keyword pool with their own patterns
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KeywordRule:
    name: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class SubtopicRule:
    name: str
    patterns: tuple[str, ...]
    keywords: tuple[KeywordRule, ...]


@dataclass(frozen=True)
class TopicRule:
    name: str
    subtopics: tuple[SubtopicRule, ...]


def _kw(name: str, *patterns: str) -> KeywordRule:
    return KeywordRule(name=name, patterns=tuple(patterns))


# Shared keyword pools reused across integration subtopics.
_INT_TECHNIQUES: tuple[KeywordRule, ...] = (
    _kw("Additivity property", r"additivity\s+property"),
    _kw("Average value", r"average\s+value"),
    _kw("Center of mass of a 1D object", r"center\s+of\s+mass"),
    _kw("Direct integrals", r"\bdirect\s+integrals?\b", r"\belementary integral\b"),
    _kw("Expected value (Mean)", r"expected\s+value", r"\bmean\b"),
    _kw("Fundamental theorem of Calculus",
        r"fundamental\s+theorem", r"\bFTC\b"),
    _kw("Integrable function", r"integrable\s+function"),
    _kw("Integration by parts",
        r"integration\s+by\s+parts", r"\bby\s+parts\b",
        r"\bu\s*dv\b", r"\\int\s*u\s*dv"),
    _kw("Left-endpoint approximation", r"left[-\s]?endpoint\s+approximation"),
    _kw("Mean Value Theorem for Integrals", r"mean\s+value\s+theorem\s+for\s+integrals?"),
    _kw("Midpoint rule", r"midpoint\s+rule"),
    _kw("Partial fractions decomposition",
        r"partial\s+fractions?", r"\bdecomposition\b"),
    _kw("Partition", r"\bpartition\b"),
    _kw("Region decomposition",
        r"region\s+decomposition", r"split(ting)?\s+the\s+region"),
    _kw("Riemann sum", r"Riemann\s+sum"),
    _kw("Right-endpoint approximation", r"right[-\s]?endpoint\s+approximation"),
    _kw("Substitution",
        r"\bsubstitution\b", r"\bu-?substitution\b", r"\bchange\s+of\s+variable"),
    _kw("Symmetry property", r"symmetry\s+property"),
    _kw("Total area", r"total\s+area"),
    _kw("Trigonometric functions",
        r"\\sin", r"\\cos", r"\\tan", r"\bsin\b", r"\bcos\b", r"\btan\b"),
    _kw("Trigonometric substitution",
        r"trigonometric\s+substitution", r"trig\s+sub"),
    _kw("Area of a planar region",
        r"area\s+of\s+(a\s+)?(planar\s+)?region", r"bounded\s+region"),
    _kw("Area of a region between two or more curves",
        r"area\s+between\s+(the\s+)?curves", r"between\s+two\s+curves"),
    _kw("Volume of revolution",
        r"volume\s+of\s+revolution", r"solid\s+of\s+revolution"),
    _kw("Volume of a solid of revolution",
        r"volume\s+of\s+a\s+solid\s+of\s+revolution", r"solid\s+of\s+revolution"),
    _kw("Power rule",
        r"\bpower\s+rule\b", r"power\s+rule\s+for\s+integration"),
    _kw("Sum rule",
        r"\bsum\s+rule\b", r"linearity\s+of\s+integration"),
    _kw("Logarithmic integration",
        r"\\int\s*1\s*/\s*x", r"\\int\s*\\frac\{1\}\{x\}",
        r"integral\s+of\s+1\s*/\s*x", r"\\ln\s*\|x\|", r"natural\s+log"),
    _kw("Polynomial integration",
        r"polynomial\s+(term|integration|integral)"),
    _kw("Rational functions",
        r"rational\s+function"),
    _kw("Exponential integration",
        r"\\int\s*e\^", r"exponential\s+(function|integration)"),
)


_SURFACE_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Parametrization", r"parametri[sz]"),
    _kw("Cross product", r"cross\s+product", r"\\times"),
    _kw("Normal vector", r"normal\s+vector", r"unit\s+normal"),
    _kw("Orientation", r"orientation", r"outward", r"inward"),
    _kw("Spherical coordinates", r"spherical"),
    _kw("Cylindrical coordinates", r"cylindrical"),
    _kw("Flux", r"\bflux\b"),
)


_DIFF_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Chain rule", r"chain\s+rule"),
    _kw("Constant rule", r"constant\s+rule"),
    _kw("Derivative definition", r"derivative\s+definition", r"definition\s+of\s+(the\s+)?derivative"),
    _kw("Differentiability", r"differentiab"),
    _kw("Differential", r"\bdifferential\b"),
    _kw("Exponential rule", r"exponential\s+rule"),
    _kw("First derivative test", r"first\s+derivative\s+test"),
    _kw("Product rule", r"product\s+rule"),
    _kw("Quotient rule", r"quotient\s+rule"),
    _kw("Implicit differentiation", r"implicit"),
    _kw("Leibniz notation", r"Leibniz\s+notation", r"\bdy\s*/\s*dx\b"),
    _kw("Logarithmic differentiation", r"logarithmic\s+differentiation"),
    _kw("Logarithmic rule", r"logarithmic\s+rule"),
    _kw("Mean Value Theorem", r"mean\s+value\s+theorem"),
    _kw("Normal line", r"normal\s+line"),
    _kw("Power rule", r"\bpower\s+rule\b"),
    _kw("Higher-order derivatives",
        r"second\s+derivative", r"higher[-\s]order", r"y''|y\(\s*n\s*\)"),
    _kw("Rolle's Theorem", r"Rolle"),
    _kw("Second derivative test", r"second\s+derivative\s+test"),
    _kw("Second order", r"second\s+order"),
    _kw("Slope", r"\bslope\b"),
    _kw("Smooth function", r"smooth\s+function"),
    _kw("Sum/Difference rule", r"sum\s*/\s*difference\s+rule", r"sum\s+rule", r"difference\s+rule"),
    _kw("Tangent line", r"tangent\s+line"),
    _kw("Trigonometric functions", r"\\sin|\\cos|\\tan|\bsin\b|\bcos\b|\btan\b"),
    _kw("Trigonometric rules", r"trigonometric\s+rules?"),
)


_LIMIT_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Continuity", r"continuous", r"continuity"),
    _kw("L'Hopital's rule", r"l['\u2019]\s*hopital|l['\u2019]\s*h[o\u00f4]pital"),
    _kw("Squeeze theorem", r"squeeze\s+theorem|sandwich\s+theorem"),
    _kw("One-sided limits", r"one[-\s]sided|left[-\s]hand|right[-\s]hand"),
    _kw("Infinity limits", r"\\infty|infinity"),
    _kw("Indeterminate forms",
        r"indeterminate\s+form", r"\b0\s*/\s*0\b", r"\\infty\s*/\s*\\infty"),
)


_SERIES_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Convergence tests", r"convergence\s+test"),
    _kw("Ratio test", r"ratio\s+test"),
    _kw("Root test", r"root\s+test"),
    _kw("Power series", r"power\s+series"),
    _kw("Taylor series", r"\bTaylor\b"),
    _kw("Maclaurin series", r"\bMaclaurin\b"),
    _kw("Geometric series", r"geometric\s+series"),
)


_LA_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Row reduction", r"row\s+reduction|gaussian\s+elimination"),
    _kw("Determinants", r"determinant|\\det"),
    _kw("Eigenvalues", r"eigen\s*value"),
    _kw("Eigenvectors", r"eigen\s*vector"),
    _kw("Matrix rank", r"\brank\b"),
    _kw("Systems of equations", r"system\s+of\s+(linear\s+)?equations"),
)


_ODE_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Separable", r"separable"),
    _kw("Integrating factor", r"integrating\s+factor"),
    _kw("Characteristic equation", r"characteristic\s+equation"),
    _kw("Linear ODE", r"linear\s+ode|linear\s+differential"),
    _kw("Non-linear ODE", r"non[-\s]?linear\s+ode"),
)


_COMPLEX_ARITH_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Algebraic form", r"algebraic\s+form"),
    _kw("Complex coefficients", r"complex\s+coefficients?"),
    _kw("Complex plane", r"complex\s+plane"),
    _kw("Conjugate number", r"conjugate"),
    _kw("Equations involving complex numbers", r"equations?\s+involving\s+complex", r"complex\s+equations?"),
    _kw("Imaginary part", r"imaginary\s+part"),
    _kw("Operations with complex numbers", r"operations?\s+with\s+complex", r"complex\s+number\s+arithmetic"),
    _kw("Real part", r"real\s+part"),
    _kw("Rectangular form (Standard form)", r"rectangular\s+form", r"standard\s+form"),
)

_COMPLEX_PLANE_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Argand diagram", r"Argand"),
    _kw("Complex plane", r"complex\s+plane"),
    _kw("Complex plane rotation", r"complex\s+plane\s+rotation", r"rotation"),
    _kw("Modulus of a complex number", r"modulus", r"\|z"),
    _kw("Polar form", r"polar\s+form"),
    _kw("Principal argument", r"principal\s+argument", r"\bargument\b"),
    _kw("Regions in the complex plane", r"regions?\s+in\s+the\s+complex\s+plane", r"\blocus\b", r"shade\s+regions?"),
    _kw("Scaling", r"\bscaling\b"),
    _kw("Trigonometric form", r"trigonometric\s+form"),
)

_COMPLEX_POWER_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Complex exponentiation", r"complex\s+exponentiation"),
    _kw("Complex logarithm", r"complex\s+logarithm"),
    _kw("De Moivre formulas", r"de\s+moivre"),
    _kw("Euler's formula", r"Euler"),
    _kw("Exponential form", r"exponential\s+form"),
    _kw("Nth root", r"nth\s+root", r"\bn[-\s]?th\s+root"),
    _kw("Powers of complex numbers", r"powers?\s+of\s+complex"),
    _kw("Powers of i", r"powers?\s+of\s+i\b", r"\bi\^"),
    _kw("Roots of unity", r"roots?\s+of\s+unity"),
)

_OPT_CONSTRAINED_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Constrained optimization", r"constrained\s+optimization"),
    _kw("Feasible set", r"feasible\s+set"),
    _kw("Global optimum", r"global\s+optimum"),
    _kw("Interior point methods", r"interior\s+point"),
    _kw("KKT conditions", r"\bKKT\b", r"Karush"),
    _kw("Lagrange multipliers", r"Lagrange\s+multipliers?", r"\blambda\b"),
    _kw("Multi-objective optimization", r"multi[-\s]?objective"),
    _kw("Optimal solution", r"optimal\s+solution"),
    _kw("Optimization", r"\boptimization\b"),
    _kw("Pareto front", r"Pareto\s+front"),
    _kw("Sensitivity analysis", r"sensitivity\s+analysis"),
)

_OPT_UNCONSTRAINED_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Gradient descent", r"gradient\s+descent"),
    _kw("Hessian matrix", r"Hessian"),
    _kw("Jacobian matrix", r"Jacobian"),
    _kw("Newton's method", r"Newton"),
    _kw("Optimization", r"\boptimization\b"),
    _kw("Quasi-Newton methods", r"quasi[-\s]?Newton"),
    _kw("Steepest descent", r"steepest\s+descent"),
    _kw("Unconstrained optimization", r"unconstrained\s+optimization"),
)

_EXTREMA_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Concavity", r"concav"),
    _kw("Critical point", r"critical\s+point"),
    _kw("Decreasing function", r"decreasing\s+function"),
    _kw("First derivative test", r"first\s+derivative\s+test"),
    _kw("Global extremum", r"global\s+extrem"),
    _kw("Increasing function", r"increasing\s+function"),
    _kw("Inflection point", r"inflection\s+point"),
    _kw("Local maximum", r"local\s+maximum", r"relative\s+maximum"),
    _kw("Local minimum", r"local\s+minimum", r"relative\s+minimum"),
    _kw("Maximum", r"\bmaximum\b"),
    _kw("Maximizer", r"\bmaximizer\b"),
    _kw("Minimum", r"\bminimum\b"),
    _kw("Minimizer", r"\bminimizer\b"),
    _kw("Monotonic function", r"monotonic"),
    _kw("Saddle point", r"saddle\s+point"),
    _kw("Second derivative test", r"second\s+derivative\s+test"),
    _kw("Stationary point", r"stationary\s+point"),
)

_APPLICATION_DERIVATIVE_KEYWORDS: tuple[KeywordRule, ...] = (
    _kw("Acceleration", r"\bacceleration\b"),
    _kw("Curvature", r"\bcurvature\b"),
    _kw("Instantaneous rate of change", r"instantaneous\s+rate\s+of\s+change"),
    _kw("Marginal cost", r"marginal\s+cost"),
    _kw("Marginal revenue", r"marginal\s+revenue"),
    _kw("Velocity", r"\bvelocity\b"),
)


TAXONOMY: tuple[TopicRule, ...] = (
    TopicRule(
        name="Integration",
        subtopics=(
            SubtopicRule(
                name="Indefinite Integrals",
                patterns=(
                    r"\\int(?!_)(?!\\limits_)",
                    r"indefinite\s+integral",
                    r"antiderivative",
                    r"\+\s*C\b",
                ),
                keywords=_INT_TECHNIQUES,
            ),
            SubtopicRule(
                name="Definite Integrals",
                patterns=(
                    r"\\int_",
                    r"\\int\\limits_",
                    r"definite\s+integral",
                    r"evaluate\s+the\s+integral",
                ),
                keywords=_INT_TECHNIQUES,
            ),
            SubtopicRule(
                name="Improper Integrals",
                patterns=(
                    r"improper\s+integral",
                    r"\\int[^\n]*\\infty",
                ),
                keywords=_INT_TECHNIQUES,
            ),
            SubtopicRule(
                name="Double Integrals",
                patterns=(r"\\iint", r"double\s+integral"),
                keywords=_INT_TECHNIQUES + (
                    _kw("Polar coordinates", r"polar\s+coordinate"),
                    _kw("Change of order", r"change\s+(the\s+)?order"),
                ),
            ),
            SubtopicRule(
                name="Triple Integrals",
                patterns=(r"\\iiint", r"triple\s+integral"),
                keywords=_INT_TECHNIQUES + (
                    _kw("Cylindrical coordinates", r"cylindrical"),
                    _kw("Spherical coordinates", r"spherical"),
                ),
            ),
            SubtopicRule(
                name="Surface Integrals",
                patterns=(
                    r"surface\s+integral",
                    r"\\iint[^\n]*dS",
                    r"\bdS\b",
                ),
                keywords=_SURFACE_KEYWORDS,
            ),
            SubtopicRule(
                name="Line Integrals",
                patterns=(
                    r"line\s+integral",
                    r"\\oint",
                ),
                keywords=_SURFACE_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Differentiation",
        subtopics=(
            SubtopicRule(
                name="Derivatives",
                patterns=(
                    r"\\frac\{d",
                    r"\bderivative\b",
                    r"differentiate",
                    r"\bdy\s*/\s*dx\b",
                ),
                keywords=_DIFF_KEYWORDS,
            ),
            SubtopicRule(
                name="Derivative Applications",
                patterns=(
                    r"instantaneous\s+rate",
                    r"velocity",
                    r"acceleration",
                    r"marginal\s+(cost|revenue)",
                    r"curvature",
                ),
                keywords=_APPLICATION_DERIVATIVE_KEYWORDS,
            ),
            SubtopicRule(
                name="Derivative Rules",
                patterns=(
                    r"chain\s+rule",
                    r"product\s+rule",
                    r"quotient\s+rule",
                    r"power\s+rule",
                    r"sum\s*/\s*difference\s+rule",
                    r"trigonometric\s+rules?",
                    r"exponential\s+rule",
                    r"logarithmic\s+rule",
                ),
                keywords=_DIFF_KEYWORDS,
            ),
            SubtopicRule(
                name="Extrema and Concavity",
                patterns=(
                    r"second\s+derivative\s+test",
                    r"first\s+derivative\s+test",
                    r"critical\s+point",
                    r"local\s+(maximum|minimum)",
                    r"relative\s+extrema?",
                    r"concav",
                    r"inflection\s+point",
                ),
                keywords=_EXTREMA_KEYWORDS,
            ),
            SubtopicRule(
                name="Partial Derivatives",
                patterns=(
                    r"\\frac\{\\partial",
                    r"partial\s+derivative",
                ),
                keywords=_DIFF_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Complex Numbers",
        subtopics=(
            SubtopicRule(
                name="Complex Arithmetic",
                patterns=(
                    r"complex\s+number\s+arithmetic",
                    r"operations?\s+with\s+complex",
                    r"complex\s+conjugate",
                    r"real\s+and\s+imaginary",
                    r"standard\s+form",
                    r"rectangular\s+form",
                ),
                keywords=_COMPLEX_ARITH_KEYWORDS,
            ),
            SubtopicRule(
                name="Complex Plane and Loci",
                patterns=(
                    r"complex\s+plane",
                    r"Argand",
                    r"\blocus\b",
                    r"regions?\s+in\s+the\s+complex\s+plane",
                    r"modulus\s+of\s+(a\s+)?complex",
                    r"\|z",
                    r"principal\s+argument",
                ),
                keywords=_COMPLEX_PLANE_KEYWORDS,
            ),
            SubtopicRule(
                name="Complex Powers and Roots",
                patterns=(
                    r"powers?\s+of\s+i\b",
                    r"powers?\s+of\s+complex",
                    r"roots?\s+of\s+unity",
                    r"de\s+moivre",
                    r"Euler",
                    r"complex\s+exponent",
                    r"complex\s+logarithm",
                ),
                keywords=_COMPLEX_POWER_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Optimization",
        subtopics=(
            SubtopicRule(
                name="Constrained Optimization",
                patterns=(
                    r"constrained\s+optimization",
                    r"Lagrange\s+multipliers?",
                    r"\bKKT\b",
                    r"feasible\s+set",
                    r"constraint",
                    r"Karush",
                ),
                keywords=_OPT_CONSTRAINED_KEYWORDS + _EXTREMA_KEYWORDS,
            ),
            SubtopicRule(
                name="Extrema",
                patterns=(
                    r"local\s+(maximum|minimum)",
                    r"global\s+extrem",
                    r"\bmaximum\b",
                    r"\bminimum\b",
                    r"critical\s+point",
                    r"stationary\s+point",
                ),
                keywords=_EXTREMA_KEYWORDS,
            ),
            SubtopicRule(
                name="Numerical Optimization",
                patterns=(
                    r"gradient\s+descent",
                    r"Newton",
                    r"quasi[-\s]?Newton",
                    r"steepest\s+descent",
                    r"interior\s+point",
                ),
                keywords=_OPT_UNCONSTRAINED_KEYWORDS,
            ),
            SubtopicRule(
                name="Unconstrained Optimization",
                patterns=(r"unconstrained\s+optimization", r"Hessian", r"Jacobian"),
                keywords=_OPT_UNCONSTRAINED_KEYWORDS + _EXTREMA_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Limits and Continuity",
        subtopics=(
            SubtopicRule(
                name="Limits",
                patterns=(r"\\lim\b", r"\blimit\b"),
                keywords=_LIMIT_KEYWORDS,
            ),
            SubtopicRule(
                name="Continuity",
                patterns=(r"continuous", r"continuity"),
                keywords=_LIMIT_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Series",
        subtopics=(
            SubtopicRule(
                name="Numerical Series",
                patterns=(r"\\sum", r"\bseries\b"),
                keywords=_SERIES_KEYWORDS,
            ),
            SubtopicRule(
                name="Power Series",
                patterns=(r"power\s+series",),
                keywords=_SERIES_KEYWORDS,
            ),
            SubtopicRule(
                name="Taylor/Maclaurin",
                patterns=(r"\bTaylor\b", r"\bMaclaurin\b"),
                keywords=_SERIES_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Differential Equations",
        subtopics=(
            SubtopicRule(
                name="First-order ODEs",
                patterns=(
                    r"first[-\s]order",
                    r"\by'\b(?!\')",
                    r"dy\s*/\s*dx",
                ),
                keywords=_ODE_KEYWORDS,
            ),
            SubtopicRule(
                name="Higher-order ODEs",
                patterns=(
                    r"second[-\s]order",
                    r"y''",
                    r"higher[-\s]order\s+ode",
                ),
                keywords=_ODE_KEYWORDS,
            ),
            SubtopicRule(
                name="Partial Differential Equations",
                patterns=(r"\bPDE\b", r"partial\s+differential\s+equation"),
                keywords=_ODE_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Linear Algebra",
        subtopics=(
            SubtopicRule(
                name="Matrices",
                patterns=(r"\bmatrix\b", r"\bmatrices\b"),
                keywords=_LA_KEYWORDS,
            ),
            SubtopicRule(
                name="Determinants",
                patterns=(r"determinant", r"\\det"),
                keywords=_LA_KEYWORDS,
            ),
            SubtopicRule(
                name="Eigenvalues and Eigenvectors",
                patterns=(r"eigen\s*value", r"eigen\s*vector"),
                keywords=_LA_KEYWORDS,
            ),
            SubtopicRule(
                name="Systems of Equations",
                patterns=(r"system\s+of\s+(linear\s+)?equations",),
                keywords=_LA_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Vector Calculus",
        subtopics=(
            SubtopicRule(
                name="Divergence Theorem",
                patterns=(
                    r"divergence\s+theorem",
                    r"\\nabla\s*\\cdot",
                    r"\bGauss\b",
                ),
                keywords=_SURFACE_KEYWORDS,
            ),
            SubtopicRule(
                name="Stokes' Theorem",
                patterns=(r"Stokes", r"\\nabla\s*\\times", r"\bcurl\b"),
                keywords=_SURFACE_KEYWORDS,
            ),
            SubtopicRule(
                name="Flux",
                patterns=(r"\bflux\b", r"F\s*\\cdot\s*dS"),
                keywords=_SURFACE_KEYWORDS,
            ),
        ),
    ),
    TopicRule(
        name="Algebra",
        subtopics=(
            SubtopicRule(
                name="Equation Solving",
                patterns=(
                    r"solve\s+the\s+equation",
                    r"find\s+(?:the\s+)?roots",
                    r"\bquadratic\b",
                ),
                keywords=(
                    _kw("Quadratic formula", r"quadratic\s+formula"),
                    _kw("Factoring", r"\bfactor(?:ing|ise|ize)?\b"),
                    _kw("Rational roots", r"rational\s+root"),
                ),
            ),
            SubtopicRule(
                name="Inequalities",
                patterns=(r"inequalit(?:y|ies)",),
                keywords=(
                    _kw("Sign analysis", r"sign\s+analysis"),
                    _kw("Interval notation", r"interval\s+notation"),
                ),
            ),
        ),
    ),
)


def _count_hits(patterns: tuple[str, ...], text: str) -> int:
    """Return how many of ``patterns`` match somewhere in ``text`` (case-insensitive)."""
    if not text:
        return 0
    hits = 0
    for pat in patterns:
        try:
            if re.search(pat, text, re.IGNORECASE):
                hits += 1
        except re.error:
            continue
    return hits


def _best_subtopic(text: str) -> tuple[TopicRule, SubtopicRule, int] | None:
    """Find the (topic, subtopic) pair with the highest number of pattern hits."""
    best: tuple[TopicRule, SubtopicRule, int] | None = None
    for topic in TAXONOMY:
        for sub in topic.subtopics:
            hits = _count_hits(sub.patterns, text)
            if hits == 0:
                continue
            if best is None or hits > best[2]:
                best = (topic, sub, hits)
    return best


def classify_taxonomy(text: str, max_keywords: int = 5) -> dict:
    """Return ``{topic, subtopic, keywords}`` for the given problem text.

    ``keywords`` is capped at ``max_keywords`` entries (minimum two when the
    subtopic's pool provides enough matches). When no subtopic pattern fires
    ``topic`` and ``subtopic`` are empty strings, but the function still runs
    a keyword scan against the union of all known keyword rules so that the
    UI always shows something useful when possible.
    """
    text = text or ""
    picked = _best_subtopic(text)

    if picked is None:
        pool: list[KeywordRule] = []
        for topic in TAXONOMY:
            for sub in topic.subtopics:
                pool.extend(sub.keywords)
        seen: set[str] = set()
        unique_pool: list[KeywordRule] = []
        for rule in pool:
            if rule.name in seen:
                continue
            seen.add(rule.name)
            unique_pool.append(rule)
        keywords = [r.name for r in unique_pool if _count_hits(r.patterns, text)]
        return {
            "topic": "",
            "subtopic": "",
            "keywords": keywords[:max_keywords],
        }

    topic, sub, _ = picked
    keywords: list[tuple[str, int]] = []
    for rule in sub.keywords:
        hits = _count_hits(rule.patterns, text)
        if hits:
            keywords.append((rule.name, hits))

    keywords.sort(key=lambda kv: (-kv[1], kv[0].lower()))
    keyword_names = [name for name, _ in keywords[:max_keywords]]

    return {
        "topic": topic.name,
        "subtopic": sub.name,
        "keywords": keyword_names,
    }


def _find_subtopic(topic_name: str, subtopic_name: str) -> SubtopicRule | None:
    """Locate a subtopic rule by display name (falls back to ``None``)."""
    tn = (topic_name or "").strip().lower()
    sn = (subtopic_name or "").strip().lower()
    if not tn or not sn:
        return None
    for topic in TAXONOMY:
        if topic.name.lower() != tn:
            continue
        for sub in topic.subtopics:
            if sub.name.lower() == sn:
                return sub
    return None


def keywords_for_subtopic(topic_name: str, subtopic_name: str, text: str) -> list[str]:
    """Return keyword names from the given subtopic pool that match ``text``.

    Keeps the result ordered by pattern-hit count (descending) and falls back
    to case-insensitive name ordering on ties.
    """
    sub = _find_subtopic(topic_name, subtopic_name)
    if sub is None or not text:
        return []
    scored: list[tuple[str, int]] = []
    for rule in sub.keywords:
        hits = _count_hits(rule.patterns, text)
        if hits:
            scored.append((rule.name, hits))
    scored.sort(key=lambda kv: (-kv[1], kv[0].lower()))
    return [name for name, _ in scored]


_DEFINITE_CUES: tuple[str, ...] = (
    r"\\int_\{",
    r"\\int_[a-z0-9\\]",
    r"\\int\s*_",
    r"definite\s+integral",
    r"area\s+(under|between)",
    r"volume",
    r"washer",
    r"\bdisk\b",
    r"\bshell\b",
    r"revolution",
    r"rotate",
    r"solid\s+of",
    r"axis\s+of\s+rotation",
    r"\\big\|_",
    r"\\Big\|_",
    r"\bevaluate\b\s+the\s+(definite|following)\s+integral",
    r"compute\s+the\s+(definite|following)\s+integral",
    r"limits\s+of\s+integration",
    r"upper\s+limit",
    r"lower\s+limit",
)


def _has_definite_cue(text: str) -> bool:
    if not text:
        return False
    for pat in _DEFINITE_CUES:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


_TRIPLE_CUES: tuple[str, ...] = (r"\\iiint", r"∭", r"triple\s+integral")
_DOUBLE_CUES: tuple[str, ...] = (r"\\iint(?!t)", r"∬", r"double\s+integral")


def _has_any_cue(text: str, patterns: Iterable[str]) -> bool:
    if not text:
        return False
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def refine_subtopic(taxonomy: dict, *texts: str) -> dict:
    """Repair common subtopic drifts using cues found in supplied texts.

    Two corrections are applied (in order), both no-ops when their guard
    fails:

    1. ``Indefinite Integrals`` → ``Definite Integrals`` when any text
       contains a definite-integral cue (limits, "evaluate definite",
       volume / area / revolution words, ``\\int_a^b`` etc.).
    2. ``Triple Integrals`` / ``Double Integrals`` → ``Definite Integrals``
       when nothing in the texts actually mentions a triple / double
       integral. This catches the case where ``topic_from_keywords``
       had to break a tie between several Integration sub-pools whose
       keyword sets are identical (``_INT_TECHNIQUES``) and the chosen
       label is incompatible with the surface text. Without this guard
       a plain "volume of revolution" video can be mislabeled as
       Triple Integrals.

    Used by both the PDF and video flows so the subtopic does not drift to
    "Indefinite" just because the regex path missed a ``\\int_{a}^{b}``
    LaTeX detail or the upstream summary is in narrative English.
    Returns a *new* dict; the input is not mutated.
    """
    if not isinstance(taxonomy, dict):
        return taxonomy
    if taxonomy.get("topic") != "Integration":
        return taxonomy

    blob = "\n".join(t for t in texts if t)
    sub = taxonomy.get("subtopic")

    if sub == "Indefinite Integrals" and _has_definite_cue(blob):
        return {**taxonomy, "subtopic": "Definite Integrals"}

    if sub == "Triple Integrals" and not _has_any_cue(blob, _TRIPLE_CUES):
        target = "Definite Integrals" if _has_definite_cue(blob) else "Indefinite Integrals"
        return {**taxonomy, "subtopic": target}

    if sub == "Double Integrals" and not _has_any_cue(blob, _DOUBLE_CUES):
        target = "Definite Integrals" if _has_definite_cue(blob) else "Indefinite Integrals"
        return {**taxonomy, "subtopic": target}

    return taxonomy


def keyword_hit_counts(text: str, keyword_names: Iterable[str]) -> dict[str, int]:
    """Return ``{keyword_name: pattern_match_count}`` for each given keyword.

    Walks every ``KeywordRule`` in the taxonomy and looks up by display name
    (case-insensitive). Keywords that do not exist as a rule (e.g. names from
    the MathE pool that are not yet codified in ``taxonomy.py``) get a count
    of ``0`` rather than being missing — this keeps callers' lookup logic
    simple. Used by ``confidence.py`` as the rule_hits signal in the abstain
    decision.
    """
    text = text or ""
    rules_by_name: dict[str, tuple[str, ...]] = {}
    for topic in TAXONOMY:
        for sub in topic.subtopics:
            for rule in sub.keywords:
                key = rule.name.strip().lower()
                if key not in rules_by_name:
                    rules_by_name[key] = rule.patterns

    out: dict[str, int] = {}
    for raw in keyword_names or []:
        if not raw:
            continue
        name = str(raw)
        lc = name.strip().lower()
        patterns = rules_by_name.get(lc)
        out[name] = _count_hits(patterns, text) if patterns else 0
    return out


def merge_keywords(*lists: list[str]) -> list[str]:
    """Concatenate keyword lists preserving first-seen order and removing duplicates."""
    seen: set[str] = set()
    out: list[str] = []
    for items in lists:
        for name in items or []:
            if name in seen:
                continue
            seen.add(name)
            out.append(name)
    return out


def topic_from_keywords(keywords: list[str]) -> tuple[str, str] | None:
    """Infer ``(topic, subtopic)`` from a list of already-chosen keyword names.

    Finds the subtopic whose keyword pool best matches the given list (by
    count of matches). Useful as a fallback for sources whose text does not
    trigger the regex-based classifier — e.g. an English video summary that
    never uses the ``\\int`` LaTeX macro. Returns ``None`` when nothing matches.
    """
    if not keywords:
        return None

    ranked = [k.strip().lower() for k in keywords if k and k.strip()]
    if not ranked:
        return None
    wanted = set(ranked)

    candidates: list[tuple[int, int, TopicRule, SubtopicRule]] = []
    best_score = -1
    for topic in TAXONOMY:
        for sub in topic.subtopics:
            pool_lower = {r.name.lower() for r in sub.keywords}
            # Keyword order already carries relevance from the hybrid ranker.
            # Weight early hits more strongly so a topic matching the top item
            # wins over a broad topic that only ties by count.
            score = sum(max(1, len(ranked) - i) ** 2 for i, kw in enumerate(ranked) if kw in pool_lower)
            hits = len(wanted & pool_lower)
            if score == 0:
                continue
            if score > best_score:
                best_score = score
                candidates = [(score, hits, topic, sub)]
            elif score == best_score:
                candidates.append((score, hits, topic, sub))

    if not candidates:
        return None

    # Many integration subtopics share the same keyword pool (e.g. Definite,
    # Indefinite, Double, and Triple all reuse ``_INT_TECHNIQUES``), so ties
    # are extremely common. Tie-break order goes from most generic / most
    # likely default to most specific. "Triple Integrals" / "Double Integrals"
    # are intentionally last: they should only win when the regex stage
    # explicitly fired on ``\iiint`` / ``\iint`` (handled by
    # ``_best_subtopic``). Picking Triple here purely on a keyword tie made
    # plain calculus videos (volume of revolution, area between curves) be
    # mis-labeled as Triple Integrals.
    tie_order = (
        "Definite Integrals",
        "Indefinite Integrals",
        "Improper Integrals",
        "Surface Integrals",
        "Line Integrals",
        "Double Integrals",
        "Triple Integrals",
    )
    names = {sub.name for _, _, _, sub in candidates}
    for pref in tie_order:
        if pref in names:
            for _, _, topic, sub in candidates:
                if sub.name == pref:
                    return (topic.name, sub.name)
    _, _, topic, sub = candidates[0]
    return (topic.name, sub.name)
