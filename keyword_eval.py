"""
Keyword evaluation helper used by the web UI and by one-off scripts.

Two tasks are implemented to mirror the MathE question structure:

* **Task 1 (free suggestion):** ask the model for exactly five concise
  technique / concept keywords that describe the problem statement.
* **Task 2 (closed pool):** ask the model to pick, in order of relevance,
  the keywords from a fixed vocabulary that best fit the statement.

Both tasks run on Gemini (the pipeline's default text model) with
``temperature=0`` and a deterministic system prompt so results are easy to
reproduce. The callers supply the extracted problem text; this module
does not touch Layer 0/3.
"""

from __future__ import annotations

import time
from typing import Iterable

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL


# Default MathE keyword pool. Callers can override the list through
# ``evaluate_keywords(pool=...)`` so this default stays configuration-free.
DEFAULT_KEYWORD_POOL: tuple[str, ...] = (
    "Total area",
    "Riemann sum",
    "Left-endpoint approximation",
    "Right-endpoint approximation",
    "Midpoint rule",
    "Partition",
    "Integrable function",
    "Mean Value Theorem for Integrals",
    "Average value",
    "Additivity property",
    "Symmetry property",
    "Volume of a solid of revolution",
    "Center of mass of a 1D object",
    "Expected value (Mean)",
    "Fundamental theorem of Calculus",
    "Direct integrals",
    "Integration by parts",
    "Trigonometric substitution",
    "Partial fractions decomposition",
    "Trigonometric functions",
    "Area of a planar region",
    "Volume of revolution",
    "Substitution",
    "Area of a region between two or more curves",
    "Region decomposition",
    "Acceleration",
    "Algebraic form",
    "Argand diagram",
    "Chain rule",
    "Complex coefficients",
    "Complex exponentiation",
    "Complex logarithm",
    "Complex plane",
    "Complex plane rotation",
    "Concavity",
    "Conjugate number",
    "Constant rule",
    "Constrained optimization",
    "Continuity",
    "Critical point",
    "Curvature",
    "De Moivre formulas",
    "Decreasing function",
    "Derivative definition",
    "Differentiability",
    "Differential",
    "Equations involving complex numbers",
    "Euler's formula",
    "Exponential form",
    "Exponential rule",
    "Feasible set",
    "First derivative test",
    "Global extremum",
    "Global optimum",
    "Gradient descent",
    "Hessian matrix",
    "Higher-order derivatives",
    "Imaginary part",
    "Increasing function",
    "Inflection point",
    "Instantaneous rate of change",
    "Interior point methods",
    "Jacobian matrix",
    "KKT conditions",
    "Lagrange multipliers",
    "Leibniz notation",
    "L'Hopital's Rule",
    "Local maximum",
    "Local minimum",
    "Logarithmic rule",
    "Marginal cost",
    "Marginal revenue",
    "Maximizer",
    "Maximum",
    "Mean Value Theorem",
    "Minimizer",
    "Minimum",
    "Modulus of a complex number",
    "Monotonic function",
    "Multi-objective optimization",
    "Newton's method",
    "Normal line",
    "Nth root",
    "Operations with complex numbers",
    "Optimal solution",
    "Optimization",
    "Pareto front",
    "Polar form",
    "Power rule",
    "Powers of complex numbers",
    "Powers of i",
    "Principal argument",
    "Product rule",
    "Quasi-Newton methods",
    "Quotient rule",
    "Real part",
    "Rectangular form (Standard form)",
    "Regions in the complex plane",
    "Rolle's Theorem",
    "Roots of unity",
    "Saddle point",
    "Scaling",
    "Second derivative test",
    "Second order",
    "Sensitivity analysis",
    "Slope",
    "Smooth function",
    "Stationary point",
    "Steepest descent",
    "Sum rule",
    "Sum/Difference rule",
    "Tangent line",
    "Trigonometric form",
    "Trigonometric rules",
    "Unconstrained optimization",
    "Velocity",
)


TASK1_SYSTEM = (
    "You extract semantic tags from a single mathematics problem statement. "
    "Do NOT solve the problem. Read the statement carefully and output exactly "
    "five concise technique- or concept-level keywords in English. "
    "Keywords should be 2-4 words each, title case, no punctuation inside, "
    "no numbering. Output a single comma-separated line ordered from most to "
    "least relevant. Nothing else."
)

TASK2_SYSTEM = (
    "You classify a mathematics problem against a closed list of keywords. "
    "Do NOT add any keyword outside the list. Select only those keywords that "
    "are directly relevant to the problem. Order them from most to least "
    "relevant. Output a single comma-separated line (copy the keyword phrasing "
    "exactly as given in the list). If nothing fits, output only NONE."
)


def _client() -> genai.Client:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=GEMINI_API_KEY)


def _ask_line(client: genai.Client, model: str, system_prompt: str, user_prompt: str) -> str:
    """Single Gemini call returning a stripped single-line string.

    Gemini 2.5 "thinking" tokens count against ``max_output_tokens``; Pro can
    easily burn 2k+ tokens internally even for a short reply. Give Pro a wider
    budget so the visible answer never gets truncated.
    """
    budget = 8192 if "pro" in model.lower() else 2048
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=budget,
            temperature=0.0,
        ),
    )
    return (response.text or "").strip()


def _clean_list(text: str) -> list[str]:
    """Split a ``a, b, c`` reply into a tidy list, dropping empties and ``NONE``."""
    items: list[str] = []
    for piece in (text or "").split(","):
        s = piece.strip().strip(".").strip()
        if not s:
            continue
        if s.upper() == "NONE":
            continue
        items.append(s)
    return items


def evaluate_keywords(
    problem_text: str,
    *,
    pool: Iterable[str] | None = None,
    model: str | None = None,
) -> dict:
    """Run Task 1 and Task 2 for a single problem statement.

    Returns a dict with ``task1`` (list of five free-form keywords), ``task2``
    (subset of ``pool`` ordered by relevance), ``model_used`` and ``elapsed_s``.
    Raises ``RuntimeError`` if the Gemini client cannot be built.
    """
    text = (problem_text or "").strip()
    if not text:
        return {
            "task1": [],
            "task2": [],
            "model_used": "",
            "elapsed_s": 0.0,
            "error": "empty problem text",
        }

    pool_list = list(pool) if pool else list(DEFAULT_KEYWORD_POOL)
    model = (model or GEMINI_MODEL or "gemini-2.5-flash").strip()

    t0 = time.time()
    client = _client()

    task1_prompt = (
        "Problem statement (from OCR/VLM):\n"
        + text
        + "\n\nOutput exactly five keywords, comma-separated, ordered by relevance."
    )
    task2_prompt = (
        "Closed keyword list (use these phrasings only):\n"
        + ", ".join(pool_list)
        + "\n\nProblem statement (from OCR/VLM):\n"
        + text
        + "\n\nOutput the subset that best fits, ordered from most to least relevant, "
          "as a comma-separated line."
    )

    task1_raw = _ask_line(client, model, TASK1_SYSTEM, task1_prompt)
    task2_raw = _ask_line(client, model, TASK2_SYSTEM, task2_prompt)

    return {
        "task1": _clean_list(task1_raw),
        "task2": _clean_list(task2_raw),
        "task1_raw": task1_raw,
        "task2_raw": task2_raw,
        "pool": pool_list,
        "model_used": model,
        "elapsed_s": round(time.time() - t0, 1),
    }
