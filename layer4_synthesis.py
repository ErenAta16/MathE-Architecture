"""
Layer 4 — merge L0 text + L2 Nougat + L3 VLM into one LLM prompt.
Picks a template from quality scores and `profile["domain"]`.
Token budget keeps the final prompt under MAX_PROMPT_CHARS.
"""

MAX_PROMPT_CHARS = 24_000  # ~6k tokens at 4 chars/token


class Layer4_Synthesis:
    """Build the final user message sent to the solver LLM."""

    def synthesize(self, raw_text: str, nougat_latex: str, nougat_score: int,
                   vlm_latex: str, vlm_score: int,
                   profile: dict = None, md_text: str = "") -> dict:
        """Prefer both OCR paths when decent; else fall back to raw PyMuPDF text.
        `md_text` is the PyMuPDF4LLM structured markdown (richer than raw_text)."""
        nougat_good = nougat_score >= 2
        vlm_good = vlm_score >= 2

        base_text = md_text if len(md_text) > len(raw_text) else raw_text

        if nougat_good and vlm_good:
            source = "nougat+vlm+raw"
            prompt = self._build_triple_source(base_text, nougat_latex, vlm_latex, profile)
        elif nougat_good:
            source = "nougat+raw"
            prompt = self._build_nougat_primary(base_text, nougat_latex, profile)
        elif vlm_good:
            source = "vlm+raw"
            prompt = self._build_vlm_primary(base_text, vlm_latex, profile)
        else:
            source = "raw_fallback"
            prompt = self._build_raw_fallback(base_text, profile)

        prompt = self._enforce_budget(prompt)

        domain = profile.get("domain", "general_math") if profile else "general_math"

        return {
            "prompt": prompt,
            "source": source,
            "domain": domain,
            "nougat_score": nougat_score,
            "vlm_score": vlm_score,
            "prompt_chars": len(prompt),
        }

    @staticmethod
    def _enforce_budget(prompt: str) -> str:
        """Hard trim to MAX_PROMPT_CHARS keeping the header and tail intact."""
        if len(prompt) <= MAX_PROMPT_CHARS:
            return prompt
        head = prompt[:2000]
        tail = prompt[-(MAX_PROMPT_CHARS - 2200):]
        return head + "\n\n[... content trimmed for token budget ...]\n\n" + tail

    @staticmethod
    def _secondary_hints_line(profile: dict | None) -> str:
        if not profile:
            return ""
        sec = profile.get("secondary_categories") or []
        if not sec:
            return ""
        joined = ", ".join(str(c) for c in sec[:5])
        return f"Additional topic signals (secondary): {joined}.\n"

    def _problem_header(self, profile: dict) -> str:
        """Pick the opening line of the user prompt from `domain` + `category`."""
        if not profile:
            return "Solve the following mathematical problem.\n"

        domain = profile.get("domain", "general_math")
        cat = profile.get("category", "unknown")

        if domain == "surface_integral":
            header = "Solve the following surface integral problem.\n"
            hints = {
                "scalar_surface_integral": (
                    "HINT: This is a SCALAR surface integral (∬ f dS). "
                    "Compute |r_u × r_v| for the surface element.\n"
                ),
                "flux_integral": (
                    "HINT: This is a FLUX integral (∬ F · dS). "
                    "Compute r_u × r_v (NOT its magnitude). Pay attention to orientation.\n"
                ),
                "divergence_theorem": (
                    "HINT: This likely uses the DIVERGENCE THEOREM. "
                    "Convert the surface integral to a volume integral using ∭ div(F) dV.\n"
                ),
                "stokes_theorem": (
                    "HINT: This likely uses STOKES' THEOREM. "
                    "Convert between line integral and surface integral of curl(F).\n"
                ),
            }
            return header + hints.get(cat, "") + self._secondary_hints_line(profile)

        cat_names = {
            "indefinite_integral": "indefinite integral",
            "definite_integral": "definite integral",
            "double_integral": "double integral",
            "triple_integral": "triple integral",
            "derivative": "derivative / differentiation",
            "limit": "limit",
            "series": "series / summation",
            "differential_equation": "differential equation",
            "linear_algebra": "linear algebra",
            "equation": "equation",
        }
        type_name = cat_names.get(cat, "mathematical")
        return f"Solve the following {type_name} problem.\n" + self._secondary_hints_line(profile)

    @staticmethod
    def _detect_conflicts(nougat: str, vlm: str) -> str:
        """Find exponent/sign/symbol mismatches between two OCR sources."""
        import re as _re
        conflicts = []
        # Compare exponents like x^2 vs x^3
        n_exps = set(_re.findall(r'[a-zA-Z]\^[\{]?(\d+)', nougat))
        v_exps = set(_re.findall(r'[a-zA-Z]\^[\{]?(\d+)', vlm))
        if n_exps != v_exps and n_exps and v_exps:
            conflicts.append(f"Exponent mismatch: Nougat has {n_exps}, VLM has {v_exps}")
        # Compare trig functions present
        trig = ["sin", "cos", "tan"]
        for fn in trig:
            n_has = f"\\{fn}" in nougat
            v_has = f"\\{fn}" in vlm
            if n_has != v_has:
                conflicts.append(f"Trig conflict: \\{fn} in {'Nougat' if n_has else 'VLM'} but not the other")
        if not conflicts:
            return ""
        return "CONFLICTS DETECTED between OCR sources:\n" + "\n".join(f"  - {c}" for c in conflicts) + "\nResolve these carefully using context.\n\n"

    def _build_triple_source(self, raw: str, nougat: str, vlm: str,
                              profile: dict) -> str:
        header = self._problem_header(profile)
        conflict_note = self._detect_conflicts(nougat, vlm)
        return (
            header + "\n"
            "Two independent OCR systems extracted the content. Cross-reference them.\n\n"
            + conflict_note +
            "--- SOURCE 1: Nougat OCR (LaTeX) ---\n" + nougat + "\n\n"
            "--- SOURCE 2: VLM Vision (LaTeX) ---\n" + vlm + "\n\n"
            "--- SOURCE 3: Raw text (plain, for verification) ---\n" + raw + "\n\n"
            "Use Sources 1 and 2 as primary mathematical references. "
            "Use Source 3 to verify variable names and bounds."
        )

    def _build_nougat_primary(self, raw: str, nougat: str,
                                profile: dict) -> str:
        header = self._problem_header(profile)
        return (
            header + "\n"
            "--- OCR LaTeX extraction (primary source) ---\n" + nougat + "\n\n"
            "--- Raw text extraction (supplementary) ---\n" + raw + "\n\n"
            "Use the LaTeX as primary source. Cross-reference with raw text if anything is unclear."
        )

    def _build_vlm_primary(self, raw: str, vlm: str,
                             profile: dict) -> str:
        header = self._problem_header(profile)
        return (
            header + "\n"
            "--- VLM Vision extraction (primary source) ---\n" + vlm + "\n\n"
            "--- Raw text extraction (supplementary) ---\n" + raw + "\n\n"
            "The VLM extracted content directly from the PDF image. "
            "Use it as primary source, cross-reference with raw text."
        )

    def _build_raw_fallback(self, raw: str, profile: dict) -> str:
        header = self._problem_header(profile)
        return (
            header + "\n"
            "Both OCR systems failed. Only raw text is available.\n"
            "Mathematical notation may be garbled (e.g. 'ZZ' means double integral, "
            "superscripts may be missing).\n\n"
            "Carefully reconstruct the mathematical problem, then solve it step by step.\n\n"
            "--- Raw text ---\n" + raw
        )
