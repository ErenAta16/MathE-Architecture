"""
Layer 5 — LLM solver (Groq, Gemini, Claude, or OpenAI-compatible).
Picks the first provider that has a working API key unless `force_provider` is set.
"""

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_BASE_URL,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    CLAUDE_MODEL,
    GPT_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LLM_SYSTEM_PROMPT,
)


class Layer5_LLMSolver:
    """Thin wrapper: send the built prompt to one LLM and return raw text."""

    def __init__(self, force_provider: str | None = None):
        self.provider = None
        self.client = None
        self.model_name = None
        self._setup(force_provider)

    def _setup(self, force_provider: str | None = None):
        """Wire up one client. If `force_provider` is set, only that backend is tried."""
        providers = {
            "groq": self._init_groq,
            "gemini": self._init_gemini,
            "claude": self._init_claude,
            "openai": self._init_openai,
        }

        if force_provider:
            if force_provider in providers:
                providers[force_provider]()
            else:
                print(f"  [!] Unknown LLM provider: {force_provider}")
            return

        for name, init_fn in providers.items():
            if init_fn():
                return
        print("  [!] No LLM API key found in the environment.")

    def _init_groq(self) -> bool:
        if not GROQ_API_KEY:
            return False
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)
            self.provider = "groq"
            self.model_name = GROQ_MODEL
            print(f"  LLM: Groq ({GROQ_MODEL})")
            return True
        except ImportError:
            print("  [!] `openai` package missing. pip install openai")
            return False

    def _init_gemini(self) -> bool:
        if not GEMINI_API_KEY:
            return False
        try:
            from google import genai
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            self.provider = "gemini"
            self.model_name = GEMINI_MODEL
            print(f"  LLM: Gemini ({GEMINI_MODEL})")
            return True
        except ImportError:
            print("  [!] google-genai missing. pip install google-genai")
            return False

    def _init_claude(self) -> bool:
        if not ANTHROPIC_API_KEY:
            return False
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.provider = "claude"
            self.model_name = CLAUDE_MODEL
            print(f"  LLM: Claude ({CLAUDE_MODEL})")
            return True
        except ImportError:
            print("  [!] anthropic package missing. pip install anthropic")
            return False

    def _init_openai(self) -> bool:
        if not OPENAI_API_KEY:
            return False
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=OPENAI_API_KEY)
            self.provider = "openai"
            self.model_name = GPT_MODEL
            print(f"  LLM: GPT-4o ({GPT_MODEL})")
            return True
        except ImportError:
            print("  [!] `openai` package missing. pip install openai")
            return False

    @property
    def is_available(self) -> bool:
        return self.client is not None

    def solve(self, problem_latex: str, system_prompt: str | None = None) -> str:
        """Run the model on `problem_latex`; optional `system_prompt` overrides the default."""
        if not self.is_available:
            raise RuntimeError("No LLM client configured. Check your API keys in .env")

        prompt = system_prompt or LLM_SYSTEM_PROMPT

        if self.provider in ("groq", "openai"):
            return self._solve_openai_compat(problem_latex, prompt)
        elif self.provider == "gemini":
            return self._solve_gemini(problem_latex, prompt)
        elif self.provider == "claude":
            return self._solve_claude(problem_latex, prompt)
        else:
            raise RuntimeError(f"Unknown LLM provider: {self.provider}")

    def _solve_openai_compat(self, problem_latex: str, system_prompt: str) -> str:
        """Chat completions API (shared by Groq and OpenAI)."""
        response = self.client.chat.completions.create(
            model=self.model_name,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem_latex},
            ],
        )
        return response.choices[0].message.content or ""

    def _solve_gemini(self, problem_latex: str, system_prompt: str) -> str:
        """Gemini generate_content; if there is no \\boxed{}, ask once more for a boxed line only."""
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=problem_latex,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            ),
        )
        text = response.text or ""

        if "\\boxed" not in text:
            try:
                followup = self.client.models.generate_content(
                    model=self.model_name,
                    contents=(
                        f"Based on the following solution, state ONLY the final answer "
                        f"inside \\boxed{{}}. Nothing else.\n\nSolution:\n{text[-1500:]}"
                    ),
                    config=types.GenerateContentConfig(
                        max_output_tokens=256,
                        temperature=0.0,
                    ),
                )
                boxed_text = (followup.text or "").strip()
                if "\\boxed" in boxed_text:
                    text = text + "\n\n" + boxed_text
            except Exception:
                pass

        return text

    def _solve_claude(self, problem_latex: str, system_prompt: str) -> str:
        response = self.client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": problem_latex,
            }],
        )
        return response.content[0].text
