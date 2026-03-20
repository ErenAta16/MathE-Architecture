"""
Central paths, model names, and prompts. API keys come from `.env` via python-dotenv.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

BASE_DIR = Path(__file__).resolve().parent

WORK_DIR = BASE_DIR / "step_pipeline"
PDF_DIR = BASE_DIR / "Surface_Integration"
IMG_DIR = WORK_DIR / "images"
NOUGAT_OUT = WORK_DIR / "nougat_output"
RESULTS_DIR = WORK_DIR / "results"

for d in [WORK_DIR, PDF_DIR, IMG_DIR, NOUGAT_OUT, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# API keys — set in `.env`, never commit real values
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

NOUGAT_MODEL = "nougat"
NOUGAT_DPI = 400

# Default LLM IDs (override some via env)
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CLAUDE_MODEL = "claude-sonnet-4-20250514"
GPT_MODEL = "gpt-4o"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
VLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MAX_TOKENS = 8192
LLM_TEMPERATURE = 0.0

LLM_SYSTEM_PROMPT_SURFACE = """You are an expert mathematician specializing in multivariable calculus and surface integrals.
You must solve the given surface integral with absolute precision.

PROCEDURE — follow every step:

1. **PARSE**: Read the problem carefully. Identify the integrand f(x,y,z), the surface S, and any orientation.
   If the OCR text is garbled, reconstruct the most likely mathematical problem from context clues.

2. **PARAMETRIZE**: Choose the best parametrization for S.
   - Explicit surface z=g(x,y): use r(x,y) = <x, y, g(x,y)>
   - Sphere of radius R: use spherical coordinates r(θ,φ) = <R sinφ cosθ, R sinφ sinθ, R cosφ>
   - Cylinder: use cylindrical coordinates
   - Plane ax+by+cz=d: solve for one variable, project onto the other two
   State the parameter ranges explicitly.

3. **PARTIAL DERIVATIVES**: Compute r_u and r_v (or r_x and r_y).

4. **CROSS PRODUCT**: Compute r_u × r_v. Then compute |r_u × r_v| (the magnitude).
   For explicit z=g(x,y): |r_x × r_y| = sqrt(1 + (∂g/∂x)² + (∂g/∂y)²).

5. **SUBSTITUTE**: Write the double integral with the integrand expressed in parameters,
   multiplied by |r_u × r_v|, with correct bounds.

6. **EVALUATE**: Compute the integral step by step. Show every antiderivative.
   Be extremely careful with:
   - Trigonometric identities (sin²θ, cos²θ, double-angle formulas)
   - Power-reduction formulas
   - Integration bounds — do NOT drop limits
   - Coefficients — track every constant multiplier

7. **VERIFY**: Before stating the final answer, do a sanity check:
   - Does the answer have correct units/dimensions?
   - Is the sign correct?
   - For simple cases, does it match known results?
   - Recheck any step where you multiplied or divided by a constant.

8. **SIMPLIFY**: Reduce the answer to its SIMPLEST possible form:
   - Combine fractions, factor out common terms
   - Cancel common factors
   - Express in standard mathematical notation

9. **FINAL ANSWER**: State the exact symbolic result in SIMPLEST FORM inside \\boxed{}.
   Example: \\boxed{\\frac{4\\pi}{3}}

10. **SUMMARY**: After the boxed answer, write a brief analysis section with this exact format:
   SUMMARY:
   - Problem Type: (e.g. scalar surface integral, flux integral, divergence theorem, etc.)
   - Method Used: (e.g. spherical parametrization, direct computation, Stokes' theorem, etc.)
   - Surface: (e.g. unit sphere, paraboloid z=x²+y², plane 2x+3y+z=6, etc.)
   - Key Steps: (1-2 sentence description of the critical steps)
   - Difficulty: (Easy / Medium / Hard)
   - Domain: (e.g. Calculus III, Vector Calculus, etc.)

CRITICAL RULES:
- Use LaTeX for all math.
- The final answer MUST be exact, symbolic, and in SIMPLEST FORM (no decimals).
- The final answer MUST appear inside \\boxed{}.
- Double-check arithmetic at every step before proceeding.
- ALWAYS include the SUMMARY section after the answer."""

LLM_SYSTEM_PROMPT_GENERAL = """You are an expert mathematician. Solve the given mathematical problem with absolute precision.

PROCEDURE:

1. **PARSE**: Read the problem carefully. If the input comes from OCR and is garbled,
   reconstruct the most likely mathematical problem from context clues.
   Identify the type of problem: integral, derivative, limit, series, equation, etc.

2. **METHOD**: Choose the best solution method:
   - **Integrals**: substitution, integration by parts, partial fractions, trig substitution, etc.
   - **Derivatives**: chain rule, product rule, quotient rule, implicit differentiation, etc.
   - **Limits**: L'Hôpital's rule, squeeze theorem, Taylor expansion, etc.
   - **Series**: convergence tests, power series, Taylor/Maclaurin series, etc.
   - **Equations**: algebraic manipulation, factoring, quadratic formula, etc.
   - **Linear Algebra**: row reduction, eigenvalues, determinants, etc.
   - **Differential Equations**: separation of variables, integrating factor, characteristic eq, etc.

3. **SOLVE**: Work through the solution step by step. Show every key step.
   Be extremely careful with:
   - Signs and coefficients
   - Trig identities and substitution back-conversions
   - Integration constants (+ C for indefinite integrals)
   - Domain restrictions

4. **SIMPLIFY**: Reduce the answer to its SIMPLEST possible form:
   - Combine fractions under a common denominator
   - Factor out common terms
   - Use standard mathematical notation (prefer arcsin over sin^{-1})
   - Cancel common factors in numerator and denominator
   - Simplify nested radicals if possible

5. **VERIFY**: Before stating the final answer, do a sanity check:
   - Differentiate an antiderivative to check it matches the integrand
   - Plug the answer back into the equation if applicable
   - Check dimensional consistency
   - Verify edge cases

6. **FINAL ANSWER**: State the exact symbolic result in SIMPLEST FORM inside \\boxed{}.
   For indefinite integrals, include + C.
   Example: \\boxed{\\frac{1}{16}\\arcsin(2x) - \\frac{x}{8}\\sqrt{1-4x^2} + C}

7. **SUMMARY**: After the boxed answer, write a brief analysis section with this exact format:
   SUMMARY:
   - Problem Type: (e.g. indefinite integral, definite integral, derivative, limit, etc.)
   - Method Used: (e.g. trigonometric substitution, integration by parts, etc.)
   - Key Steps: (1-2 sentence description of the critical steps)
   - Difficulty: (Easy / Medium / Hard)
   - Domain: (e.g. Calculus I, Calculus II, Linear Algebra, etc.)

CRITICAL RULES:
- Use LaTeX for all math.
- The final answer MUST be exact, symbolic, and in SIMPLEST FORM (no decimals).
- The final answer MUST appear inside \\boxed{}.
- Double-check arithmetic at every step before proceeding.
- ALWAYS include the SUMMARY section after the answer."""

LLM_SYSTEM_PROMPT = LLM_SYSTEM_PROMPT_SURFACE

# Reference answers for SymPy verification in L6.
# Sources: "At the end you should get:" hints in PDFs, manual checks, LLM consensus.
from sympy import pi, sqrt, S, E, Symbol

a = Symbol('a', positive=True)

KNOWN_ANSWERS = {
    # --- Scalar surface integrals ---
    "si1": 4 * pi / 3,
    "si2": 13 * sqrt(2) / 3,
    "si3": pi * (sqrt(2) + S(3) / 2),
    "si4": 11 * sqrt(14),
    "si5": sqrt(2) / 10,
    "si6": 2 * (2 * sqrt(2) - 1) / 3,
    "si7": sqrt(2) * pi,
    "si8": 171 * sqrt(14),
    "si9": S(4),
    "si10": sqrt(21) / 3,
    "si11": 4 * (9 * sqrt(3) + 4 * sqrt(2) - 2) / 105,
    "si12": 364 * sqrt(2) * pi / 3,
    "si13": 13 * sqrt(2) / 12,
    "si14": pi * (391 * sqrt(17) + 1) / 60,
    "si15": (S(32) / 3 - 6 * sqrt(3)) * pi,
    "si16": 16 * pi,
    "si17": S(0),
    "si18": S(12),
    "si22": S(4),
    "si23": pi,
    "si24": S(713) / 180,
    "si34": sqrt(3) / 24,
    "si35": 5 * sqrt(5) / 48 + S(1) / 240,
    "si37": (17 * sqrt(17) - 1) / 4,
    "si38": S(24),
    "si39": 2 * a**4 * pi / 3,
    "si40": 3 * sqrt(3),
    "si41": 2 * pi * sqrt(2) / 3,
    "si42": 11 * pi / 12,
    "si43": S(0),
    # --- Flux integrals ---
    "si19": 241 * pi,
    "si20": 4 * pi / 3,
    "si21": pi / 2,
    "si25": S(-1712) * pi / 15,
    "si26": -4 * pi / 3,
    "si27": S(0),
    "si28": S(0),
    "si29": 1 - E,
    "si30": S(48),
    "si31": 4 * pi,
    "si32": 2 * pi + S(8) / 3,
    "si33": S(-1) / 6,
    "si44": S(-32),
    "si45": S(-4) / 3,
    "si46": 4 * pi * a**3,
    "si47": 4 * pi * a**3,
    "si48": 13 * a**4 / 6,
    "si49": 2 * pi * a,
    # --- Volume / divergence–theorem style ---
    "si50": 2 * pi / 3,
    "si51": 4 * pi / 3,
    "si52(r)": S(184) / 35,
    "si53": S(9) / 2,
    "si55": 9 * pi / 2,
    "si56": 384 * pi / 5,
    "si57": S(0),
    "si59": 32 * pi / 3,
    "si60": 2 * pi / 3,
    "si61": S(2),
}
