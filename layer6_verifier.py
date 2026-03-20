"""
Layer 6 — parse the LLM’s final line and compare to `KNOWN_ANSWERS` with SymPy when possible.
"""

import re
from sympy import symbols, integrate, simplify, pi, sqrt, sin, cos, S, Symbol, sympify

from latex_parser import parse_latex_to_value, parse_latex_to_expr, latex_to_sympy
from config import KNOWN_ANSWERS

phi, theta, r, u, v, x, y, z, a = symbols("phi theta r u v x y z a")


class Layer6_SymPyVerifier:
    """SymPy checks against `KNOWN_ANSWERS`; standalone integral tests in `run_standalone_tests`."""

    def __init__(self):
        self.results = []

    def verify(self, name: str, description: str, integral_fn, expected) -> bool:
        """Manual test helper: evaluate `integral_fn` and compare to `expected`."""
        print(f"\n  {'─'*50}")
        print(f"  {name}: {description}")
        print(f"  {'─'*50}")

        try:
            result = integral_fn()
            simplified = simplify(result)

            try:
                num_r = float(simplified.evalf())
                num_e = float(expected.evalf()) if hasattr(expected, "evalf") else float(expected)
                match = abs(num_r - num_e) < 1e-6
            except Exception:
                match = simplify(simplified - expected) == 0
                num_r = str(simplified)
                num_e = str(expected)

            status = "[OK]" if match else "[FAIL]"
            print(f"    SymPy result:   {simplified}")
            print(f"    Expected:       {expected}")
            print(f"    {status}")

            self.results.append({
                "name": name,
                "success": match,
                "result": str(simplified),
                "expected": str(expected),
            })
            return match

        except Exception as e:
            print(f"    [ERROR] {e}")
            self.results.append({
                "name": name,
                "success": False,
                "result": f"ERR: {e}",
                "expected": str(expected),
            })
            return False

    @staticmethod
    def _extract_boxed(text: str) -> list[str]:
        """Collect every `\\boxed{...}` substring, respecting nested braces."""
        results = []
        start = 0
        while True:
            idx = text.find("\\boxed{", start)
            if idx == -1:
                break
            brace_start = idx + 6
            depth = 0
            i = brace_start
            while i < len(text):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        results.append(text[brace_start + 1:i])
                        break
                i += 1
            start = i + 1
        return results

    @staticmethod
    def _clean_boxed_content(text: str) -> str:
        """If the model chained equals signs, take the last segment only."""
        if "=" in text:
            parts = text.split("=")
            return parts[-1].strip()
        return text

    def _extract_final_answer(self, llm_solution: str) -> str:
        """Best-effort final answer from tags, \\boxed{}, or trailing math."""
        work = llm_solution.split("SUMMARY")[0] if "SUMMARY" in llm_solution else llm_solution

        # Strategy 1: FINAL_ANSWER: tag
        fa_match = re.search(r"FINAL_ANSWER:\s*(.+)", work)
        if fa_match:
            text = fa_match.group(1).strip().strip("$").strip()
            if text:
                return self._clean_boxed_content(text)

        # Strategy 2: \boxed{} (most reliable)
        boxed = self._extract_boxed(work)
        if boxed:
            return self._clean_boxed_content(boxed[-1].strip())

        # Strategy 3: Explicit answer phrases (multiline scan)
        phrase_patterns = [
            r"[Tt]he\s+final\s+answer\s+is[:\s]*\$*([^$\n]+?)\$*\s*$",
            r"[Ff]inal\s+[Aa]nswer[:\s]*\$*([^$\n]+?)\$*\s*$",
            r"\*\*(?:Answer|Result|Final Answer)[:\s]*\*\*\s*\$*([^$\n]+?)\$*\s*$",
        ]
        for pat in phrase_patterns:
            m = re.search(pat, work, re.MULTILINE)
            if m:
                text = m.group(1).strip().strip("$").strip()
                if text and len(text) < 120 and re.search(r'[0-9\\]', text):
                    return self._clean_boxed_content(text)

        # Strategy 4: "the flux/integral/answer/result is $...$" (inline math)
        is_patterns = [
            r"(?:flux|integral|answer|result|value)\s+(?:is|equals?)\s+\$([^$]+)\$",
            r"(?:flux|integral|answer|result|value)\s+(?:is|equals?)\s+\$\$([^$]+)\$\$",
            r"(?:across|over|of)\s+\$?[^$]*\$?\s+is\s+\$([^$]+)\$",
        ]
        for pat in is_patterns:
            matches = re.findall(pat, work, re.IGNORECASE)
            if matches:
                candidate = matches[-1].strip()
                if len(candidate) > 2 and re.search(r'[0-9\\]', candidate):
                    return self._clean_boxed_content(candidate)

        # Strategy 5: Last display math ($$...$$) that looks like a result
        display_blocks = re.findall(r'\$\$\s*(.+?)\s*\$\$', work, re.DOTALL)
        if display_blocks:
            for block in reversed(display_blocks):
                block = block.strip()
                if "boxed" in block:
                    inner = self._extract_boxed(block)
                    if inner:
                        return self._clean_boxed_content(inner[-1].strip())
                cleaned = self._clean_boxed_content(block)
                if 2 < len(cleaned) < 80 and re.search(r'[0-9]', cleaned):
                    if not re.search(r'(\\partial|\\nabla|\\int|\\iint|\\iiint)', cleaned):
                        return cleaned

        # Strategy 6: Last inline math ($...$) containing a numeric expression
        inline_blocks = re.findall(r'(?<!\$)\$([^$]+)\$(?!\$)', work)
        if inline_blocks:
            for block in reversed(inline_blocks):
                block = block.strip()
                if len(block) > 60 or len(block) < 2:
                    continue
                if re.search(r'\\(?:frac|pi|sqrt)', block) or re.search(r'\d', block):
                    if not re.search(r'(\\mathbf|\\text|\\partial|\\nabla)', block):
                        return self._clean_boxed_content(block)

        # Strategy 7: "= value" at end of a line
        eq_end = re.findall(r'=\s*\$*([^$=\n]{2,60}?)\$*\s*$', work, re.MULTILINE)
        if eq_end:
            for candidate in reversed(eq_end):
                candidate = candidate.strip().strip("$").strip()
                if re.search(r'[0-9]', candidate) and len(candidate) < 60:
                    if not re.search(r'\\(?:partial|nabla|int)', candidate):
                        return self._clean_boxed_content(candidate)

        return ""

    def _is_parametric(self, expr) -> bool:
        """True if the SymPy object still depends on a symbol (not a single number)."""
        return bool(expr.free_symbols - {S.One})

    def _symbolic_match(self, llm_text: str, known) -> dict | None:
        """Direct symbolic subtract-and-simplify vs the reference expression."""
        try:
            llm_expr = parse_latex_to_expr(llm_text)
            if llm_expr is not None:
                diff = simplify(llm_expr - known)
                if diff == 0:
                    return {"status": "match", "method": "symbolic"}
        except Exception:
            pass
        return None

    def verify_llm_answer(self, fname: str, llm_solution: str) -> dict:
        """Return match / mismatch / parse_error against `KNOWN_ANSWERS[fname]`."""
        known = KNOWN_ANSWERS.get(fname)
        if known is None:
            return {"status": "skip", "reason": "no_reference_answer"}

        is_param = self._is_parametric(known)

        fa_text = self._extract_final_answer(llm_solution)

        if fa_text:
            if not is_param:
                llm_num = parse_latex_to_value(fa_text)
                if llm_num is not None:
                    known_num = float(known.evalf())
                    error = abs(llm_num - known_num)
                    match = error < 0.01
                    return {
                        "status": "match" if match else "mismatch",
                        "known": str(known),
                        "known_numeric": known_num,
                        "llm_answer": fa_text,
                        "llm_numeric": llm_num,
                        "error": error,
                        "method": "numeric_comparison",
                    }

            sym_result = self._symbolic_match(fa_text, known)
            if sym_result:
                return {
                    **sym_result,
                    "known": str(known),
                    "llm_answer": fa_text,
                }

        if not is_param:
            known_num = float(known.evalf())
            answer_indicators = [str(known), f"{known_num:.4f}"]
            if "pi" in str(known):
                answer_indicators.append(str(known).replace("pi", "\\pi"))
            found_in_text = any(ind in llm_solution for ind in answer_indicators if ind)
            if found_in_text:
                return {
                    "status": "match",
                    "known": str(known),
                    "known_numeric": known_num,
                    "method": "string_search",
                }

        if fa_text:
            return {
                "status": "parse_error",
                "known": str(known),
                "llm_answer": fa_text,
                "reason": "could_not_parse_llm_answer",
            }

        return {
            "status": "no_answer",
            "reason": "unrecognized_answer_format",
            "known": str(known),
        }

    def summary(self):
        """Print pass/fail for the manual integral regression list."""
        s = sum(1 for r in self.results if r["success"])
        t = len(self.results)
        print(f"\n  {'='*50}")
        print(f"  LAYER 6 SUMMARY: {s}/{t} ({100 * s // t if t else 0}%)")
        print(f"  {'='*50}")
        for r in self.results:
            icon = "[OK]" if r["success"] else "[FAIL]"
            print(f"    {icon} {r['name']}")
        return s, t


def run_standalone_tests():
    """Nine small surface integrals checked only with SymPy (no LLM)."""
    verifier = Layer6_SymPyVerifier()

    print("=" * 58)
    print("  LAYER 6: SymPy regression")
    print("=" * 58)

    verifier.verify(
        "si1", "∬ x² dS — unit sphere",
        lambda: integrate(integrate(
            sin(phi) ** 2 * cos(theta) ** 2 * sin(phi),
            (phi, 0, pi)), (theta, 0, 2 * pi)),
        4 * pi / 3,
    )

    verifier.verify(
        "si2", "∬ y dS — z=x+y², [0,1]x[0,2]",
        lambda: integrate(integrate(
            y * sqrt(2 + 4 * y ** 2),
            (y, 0, 2)), (x, 0, 1)),
        13 * sqrt(2) / 3,
    )

    verifier.verify(
        "si4", "∬ (x+y+z) dS — parallelogram",
        lambda: integrate(integrate(
            (4 * u + v + 1) * sqrt(14),
            (u, 0, 2)), (v, 0, 1)),
        11 * sqrt(14),
    )

    verifier.verify(
        "si5", "∬ xyz dS — cone",
        lambda: integrate(integrate(
            u ** 3 * sin(v) * cos(v) * sqrt(2) * u,
            (v, 0, pi / 2)), (u, 0, 1)),
        sqrt(2) / 10,
    )

    verifier.verify(
        "si8", "∬ x²yz dS — z=1+2x+3y, [0,3]x[0,2]",
        lambda: integrate(integrate(
            x ** 2 * y * (1 + 2 * x + 3 * y) * sqrt(14),
            (y, 0, 2)), (x, 0, 3)),
        171 * sqrt(14),
    )

    verifier.verify(
        "si9", "∬ xz dS — plane 2x+2y+z=4, first octant",
        lambda: integrate(integrate(
            x * (4 - 2 * x - 2 * y) * 3,
            (y, 0, 2 - x)), (x, 0, 2)),
        S(4),
    )

    verifier.verify(
        "si16", "∬ (x²z+y²z) dS — upper hemisphere R=2",
        lambda: integrate(integrate(
            4 * sin(phi) ** 2 * 2 * cos(phi) * 4 * sin(phi),
            (phi, 0, pi / 2)), (theta, 0, 2 * pi)),
        16 * pi,
    )

    verifier.verify(
        "si20", "Flux F=zi+yj+xk — unit sphere",
        lambda: integrate(integrate(
            2 * sin(phi) ** 2 * cos(theta) * cos(phi) + sin(phi) ** 3 * sin(theta) ** 2,
            (phi, 0, pi)), (theta, 0, 2 * pi)),
        4 * pi / 3,
    )

    verifier.verify(
        "si39", "∬ yz dS — plane x+y+z=1, first octant",
        lambda: integrate(integrate(
            y * (1 - x - y) * sqrt(3),
            (y, 0, 1 - x)), (x, 0, 1)),
        sqrt(3) / 24,
    )

    return verifier.summary()


if __name__ == "__main__":
    run_standalone_tests()
