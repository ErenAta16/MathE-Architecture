"""Transcription prompt profiles for MathE pilot videos."""

from __future__ import annotations

MATH_VOCAB_HINT = (
    "Preserve exact mathematical terms when spoken, including: imaginary unit, "
    "powers of i, complex number, quotient rule, derivative, partial derivative, "
    "critical point, local maximum, local minimum, saddle point, Hessian matrix, "
    "second derivative test, second differential, gradient, local extrema, "
    "factorization, square root."
)

PROMPT_VERSION = "mathe-transcribe-v4-readable"

READABLE_RULES = (
    "Rules for human-readable output:\n"
    "1. Do not summarize, paraphrase, skip steps, or merge sentences.\n"
    "2. Keep discourse markers when spoken (e.g. yeah, let's write, try it out).\n"
    "3. Use lowercase i for the imaginary unit (never capital I).\n"
    "4. Do NOT use LaTeX, dollar signs, backslashes, or markup of any kind.\n"
    "5. Write every formula in plain English as a person would read it aloud, "
    "for example: 'i squared equals negative one', "
    "'two y times the quantity two y squared minus one equals zero', "
    "'f maps from R squared to R', 'dy over dx'.\n"
    "6. You may use simple ASCII when it helps clarity: +, -, =, /, parentheses, "
    "2x+1, (0,0), M11 — but prefer words over symbols when the speaker uses words.\n"
    "7. If the speaker says saddle point, quotient rule, or second derivative test, "
    "write that exact phrase.\n"
    "8. Do not describe visuals unless the speaker explicitly narrates them.\n"
    "9. If audio is unintelligible, write [inaudible] for that span only."
)

LATEX_RULES = (
    "Rules:\n"
    "1. Do not summarize, paraphrase, skip steps, or merge sentences.\n"
    "2. Keep discourse markers when spoken (e.g. yeah, let's write, try it out).\n"
    "3. Use lowercase i for the imaginary unit (never capital I).\n"
    "4. Write formulas in inline LaTeX inside $...$ when the speaker states math "
    "(e.g. $2y(2y^2-1)=0$, $f:\\mathbb{R}^2\\to\\mathbb{R}$), but keep all "
    "surrounding spoken words in plain English.\n"
    "5. If the speaker says saddle point, quotient rule, or second derivative test, "
    "write that exact phrase.\n"
    "6. Do not describe visuals unless the speaker explicitly narrates them.\n"
    "7. If audio is unintelligible, write [inaudible] for that span only."
)

PROFILES: dict[str, dict[str, str]] = {
    "latex": {
        "version": "mathe-transcribe-v3-final",
        "system": (
            "You are an expert speech-to-text transcriber for university mathematics "
            "education videos in English. "
            "Your task is a complete verbatim transcript of every word the instructor "
            "speaks aloud from the first sentence to the last. "
            f"{MATH_VOCAB_HINT} "
            f"{LATEX_RULES}"
        ),
        "user": (
            "Transcribe the full spoken narration of this mathematics lecture video, "
            "from beginning to end. "
            "Return plain text only (inline LaTeX inside $...$ is allowed). "
            "Use short paragraphs when the topic shifts. "
            "Include every step, example, and closing remark the instructor speaks."
        ),
    },
    "readable": {
        "version": "mathe-transcribe-v4-readable",
        "system": (
            "You are an expert speech-to-text transcriber for university mathematics "
            "education videos in English. "
            "Your task is a complete verbatim transcript that any reader can follow "
            "in a Word document or email — no mathematical markup. "
            f"{MATH_VOCAB_HINT} "
            f"{READABLE_RULES}"
        ),
        "user": (
            "Transcribe the full spoken narration of this mathematics lecture video, "
            "from beginning to end. "
            "Return plain ASCII text only: no LaTeX, no dollar signs, no special markup. "
            "Write formulas the way the instructor says them, in everyday English. "
            "Use short paragraphs when the topic shifts. "
            "Include every step, example, and closing remark the instructor speaks."
        ),
    },
}

WHISPER_INITIAL_PROMPT = (
    "University mathematics lecture. Terms: imaginary unit i, powers of i, "
    "quotient rule, derivative, partial derivative, critical point, saddle point, "
    "local maximum, local minimum, Hessian, second derivative test, extrema."
)


def get_prompt_profile(name: str = "readable") -> tuple[str, str, str]:
    """Return (system_instruction, user_prompt, prompt_version)."""
    profile = PROFILES.get(name) or PROFILES["readable"]
    return profile["system"], profile["user"], profile["version"]


SYSTEM_INSTRUCTION, USER_PROMPT, PROMPT_VERSION = get_prompt_profile("readable")
