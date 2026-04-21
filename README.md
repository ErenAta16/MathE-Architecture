# STEP — runnable app

This directory is the runnable slice of the project (application code and templates). Pipeline logs and comments are mostly **English**; a few CLI and web strings stay **Turkish** for local use (`run.py --check`, `web_app.py` banner). The **Project report** below is intentionally repeated in **English, Portuguese, and Turkish**. Quick start is at the end.

---

## Project report

### English

**Project:** Automated mathematical problem-solving system  
**Environment:** Windows 10, Python 3.11.9, NVIDIA GeForce RTX 3050 Ti (4.3 GB VRAM), CUDA 12.4

---

#### Scope

The first version of the system was developed as a Google Colab notebook. That single file combined PDF handling, OCR, LLM calls, and verification. The first step was to refactor this monolith into a modular local Python project. Each stage became an independent Python module, a central configuration file was added, and a command-line interface was introduced. That refactor made it possible to test, tune, and replace each layer on its own.

---

#### Pipeline architecture

The pipeline has seven layers. **Layer 0 (PDF ingestion)** uses PyMuPDF to extract raw text, metadata, and high-resolution page images from PDFs. As a C-based library, PyMuPDF completes this step in about 0.02 s on average. It ran without failure on all 58 PDFs. Because mathematical symbols (integral signs, superscripts, Greek letters) are often garbled in extracted text, a quality check with seven checkpoints evaluates the text. Image rasterization started at 300 DPI and was raised to 400 DPI to improve OCR performance.

**Layer 1 (Profiling)** interprets PDF content with a custom heuristic regex classifier. It pulls relevant terms from a pool of mathematical keywords, assigns a **primary category** (surface-integral family: scalar/flux/divergence/Stokes, plus general-math types such as integrals, derivative, limit, series, ODE, linear algebra, etc.), optional **secondary** signals, and surface type when applicable (sphere, paraboloid, cylinder, cone, plane, hemisphere, torus). A typical PDF yields several keywords in milliseconds. The derived **domain** is either `surface_integral` or `general_math`, which selects the Layer 5 system prompt in `config.get_system_prompt`.

**Layer 2 (OCR)** uses Meta’s Nougat-OCR. Nougat combines a Swin Transformer and mBART in an encoder–decoder setup and maps pixels directly to LaTeX—a different approach from engines like Tesseract that understand structures such as `\frac{}{}`, `\int`, and `\sqrt{}`. Integrating Nougat locally was one of the hardest technical steps. It depended on Albumentations, which conflicted with the installed PyTorch/transformers stack; that was fixed with a fake-module shim `_setup_albumentations_bypass()`. `generate()` validation changes in newer transformers were also monkey-patched. Even so, Nougat succeeded on only 42 of 58 PDFs (72.4%): 17 PDFs produced empty output or hit a `[repetition]` loop. Average time was ~10.7 s per PDF (~620.9 s total). Those gaps motivated **Layer 3**.

**Layer 3 (VLM)** is one of the most important additions. In the current setup, VLM page extraction runs on Gemini vision calls (with optional bounded parallel workers). Each page is sent to the API and the model returns LaTeX read from the image. VLM succeeded on **all** 58 PDFs (100%) and alone recovered the 16 PDFs where Nougat failed. Average time was ~2.4 s per PDF—about **4.5× faster** than local GPU Nougat on this hardware.

Prompt design matters: the VLM is instructed to **read only, not solve**—an eight-rule system prompt blocks solution steps, forbids `\boxed{}` answers, and stresses reading every symbol carefully. **Dual-pass** VLM runs (when the first pass does not already hit the quality rubric maximum) pick the better output by score and length. `clean_output()` strips lines that look like reasoning (“we get”, “substituting”, “therefore”) while keeping the problem statement.

**Layer 4 (Synthesis)** merges prior outputs into a single LLM prompt. It chooses among four strategies: triple source (Nougat + VLM + raw text, 42 PDFs), VLM-primary (VLM + raw, 16 PDFs), Nougat-primary, or raw fallback. Layer-1 classification adds targeted hints (e.g. for flux problems, notes on **r_u × r_v** and **|r_u × r_v|** in the surface element). Phase 6 adds general-math support: surface problems use an eight-step procedure; other math uses a more flexible prompt. Average prompt length is ~3200 characters; synthesis time is negligible.

**Layer 5 (LLM solver)** is the mathematical reasoner. Four LLMs were tried. **Claude Sonnet 4** was dropped due to latency and unreliable quality. **GPT-4o** reached ~88% in limited tests but full evaluation stopped for credit limits. **Gemini 2.5 Flash** (with “thinking”) achieved **100%** on the tested set but is slower (~15–35 s). The current setup uses **Gemini as primary** and **Together.ai Llama 3.3 70B** as fallback.

**Layer 6 (answer display)** parses the LLM reply to extract one **final line** for the UI (`\boxed{}`, `FINAL_ANSWER:`, tail heuristics). The class name still says “SymPyVerifier” for historical reasons; **reference-answer SymPy checks are not in this layer**. **Consensus / numeric agreement** between repeated LLM attempts uses `latex_parser.parse_latex_to_value` inside `run.STEPSolver._solve_with_consensus`. The LaTeX normalizer behind that path evolved across versions (fractions, roots, powers, implicit multiplication).

---

#### Issues encountered and fixes

**Nougat:** (1) Albumentations vs PyTorch → fake-module injection. (2) `generate()` validation on new transformers → monkey-patch. (3) `[repetition]` loops on 17 PDFs → partial output salvage plus VLM fallback.

**VLM:** The worst bug was `clean_output()` **deleting the problem statement**. Revising the kill-phrase list increased retained length ~570% and restored correct LLM answers.

**Answer extraction:** Gemini sometimes omits `\boxed{}` or answers in free-form LaTeX—addressed with a seven-stage extraction strategy and a follow-up API call.

---

#### Runtime and reliability updates

**Layers 0 and 2 — raster reuse:** When `dpi` is not passed in, Layer 0 rasterizes page images at **`NOUGAT_DPI`** from config so output matches what Nougat expects. Each folder of `page_*.png` files carries a small **`.step_raster_meta`** file (DPI on the first line, SHA-256 of the PDF on the second). Layer 2 only reuses those PNGs when that sidecar, the on-disk page names, and the current PDF all agree; otherwise it renders again and refreshes the sidecar. **`page_*.png` paths are sorted numerically** so order stays correct past page nine.

**Layer 3 — VLM concurrency:** Per-page vision calls can run through a **bounded worker pool**. Set **`STEP_VLM_PAGE_WORKERS`** if you need fewer parallel requests (rate limits) or fully serial calls (`1`).

**Web UI:** Several solves at once are limited with a **semaphore**; the cap is **`STEP_WEB_MAX_CONCURRENT_SOLVES`** (defaults to 2). Uploading the same sanitized filename again while the first job is still running returns **HTTP 409**. Lower the cap when Nougat-heavy GPU work risks contention.

**Layer 6 — answer cleanup:** Extracting a short final string no longer splits on `=` blindly. The code tracks **brace depth** and only uses the last top-level equals, so groups such as `\text{...}` and similar LaTeX do not get chopped apart.

**LLM provider — Groq replaced by Together.ai:** Groq’s API became unreliable during provider-side outages for our workloads, and their vision offering did not match what we needed for stable image-to-LaTeX in production. The stack now uses **Together.ai** (`TOGETHER_API_KEY`, `TOGETHER_MODEL`) as the OpenAI-compatible **chat** fallback next to **Gemini**. Vision in normal operation stays on **Gemini**; Together vision IDs remain optional for experiments and carry automatic model fallback when a chosen serverless vision model is unavailable.

**Layer 3 — handwriting / export consistency:** Page images are **normalized before each VLM call** (grayscale, autocontrast, light sharpen, fixed long-edge scale). Two PDFs of the same handwritten problem but exported at different pixel dimensions used to yield different extracted integrals; this preprocessing narrows that gap.

**Solver (`run.py`) — reliability and definite integrals:** Consensus handling fixes a `KeyError` on partial attempt records, switches to Together faster after transient errors when a usable answer already exists, and skips the LLM when no readable math remains after OCR/VLM. For **definite integrals** the pipeline can run **one** control LLM pass, then **SymPy** evaluates the integral from the parsed prompt and **overrides the displayed final** (often rewritten via `log` instead of `acosh` for readability). Intermediate-looking finals trigger a **bounded-time** refine call so the web UI does not block indefinitely.

**Gemini text — optional fallback model:** Set **`GEMINI_FALLBACK_MODEL`** (for example `gemini-2.5-flash`) so a failed primary `GEMINI_MODEL` call retries once on another Gemini ID.

**Web UI:** The layout **prioritizes profiling and the answer** over the PDF column. A compact **Simple explanation** strip summarizes problem type, method, and key idea; full solution remains expanded by default for readability. Optional **`STEP_WEB_USE_NOUGAT`** / **`STEP_WEB_USE_VLM`** toggle pipeline stages from the Flask worker.

**Adaptive rasterization:** When Nougat is disabled the pipeline rasterizes pages at **`VLM_ONLY_DPI`** (220 by default, override with `STEP_VLM_ONLY_DPI`) instead of 400 DPI. The VLM normalizes images internally anyway, so the high DPI was wasted work on the common VLM-only path. PyMuPDF4LLM markdown extraction is also gated on a thin native text layer, avoiding a second PDF open otherwise.

**Layer 3 — targeted retries and on-disk cache:** Pass 2 only re-runs pages whose individual output is empty or too short rather than repeating the whole document. The cleaned VLM output is persisted under **`step_pipeline/vlm_output/<pdf>/<pdf>.vlm.mmd`** with a SHA-256 fingerprint of all page PNGs; identical pixel input short-circuits the VLM stage entirely on later runs.

**Layer 5 — skip the extra boxed call:** When a response lacks `\boxed{}` but Layer 6 can already extract a final line (via `FINAL_ANSWER:`, display math, or tail heuristics), the follow-up LLM call is skipped, saving one round trip per solve in the common case.

**Gemini 2.5 Pro speed profile:** With `gemini-2.5-pro` as the primary, consensus runs a single strong attempt instead of stacking two long calls. Other providers keep the previous consensus budget.

**Shared solver in the web app:** A single process-wide `STEPSolver` is now built lazily and reused for every upload, so Gemini/Together clients and the Layer 3 cache persist across requests. The in-memory result cache is protected by a dedicated lock so concurrent solves do not race on the `OrderedDict`.

**Classification taxonomy (MathE-style):** A new `taxonomy.py` maps each problem to a **Topic** (Integration, Differentiation, Limits and Continuity, Series, Differential Equations, Linear Algebra, Vector Calculus, Algebra), a **Subtopic** (Indefinite Integrals, Definite Integrals, Surface Integrals, …), and up to five controlled **Keywords** (Substitution, Integration by parts, Power rule, Sum rule, Logarithmic integration, …). After the LLM finishes, its own solution and summary are rescanned against the chosen subtopic pool so the keywords reflect techniques the model actually used. Exposed in the API as `result.taxonomy = {topic, subtopic, keywords}`.

**User question on upload:** `/upload` accepts an optional **`user_query`** (up to 600 characters) which is appended to the Layer 4 prompt under an explicit marker and echoed back in the result. Its digest is part of the cache key so different notes produce different runs.

**Follow-up chat:** `/followup` is a synchronous JSON endpoint that calls `STEPSolver.ask_followup` with the task's stored prompt, prior solution, and system prompt. Each call is a single LLM pass (no consensus) and returns `{solution, final_answer, elapsed_s, model_used}`. The web UI turns the composer into a follow-up chat once a solve completes, appending each question and reply as its own card.

**Web UI — auto-solve and answer-first layout:** Uploading a PDF now starts the solver immediately; the composer is inactive until the solve completes and then switches to follow-up mode. The right panel leads with the answer, followed by a dedicated **Classification** card (Topic, Subtopic, Keywords), an echo of the user's instruction when provided, the model summary, and a collapsible **Pipeline details** table for the technical fields that used to clutter the profiling card. Full-solution rendering strips trailing `\boxed{...}` lines (the answer is already shown on top) and wraps any remaining bare boxed blocks in display-math delimiters so MathJax typesets them.

---

#### Technology choices

**PyMuPDF** — fast, reliable PDF text and raster output. **Nougat** — pixels to LaTeX for academic layouts; on its own it missed too many of our PDFs. **Gemini 2.5 Flash** — primary text solver and active VLM path. **Together.ai Llama 3.3 70B** — quick fallback when Gemini is busy or errors. **SymPy** (via `latex_parser`) — turns short LaTeX fragments into numbers so repeated solver attempts can agree. **Flask** — small web UI with SSE. **MathJax 3** — render LaTeX in the browser.

---

### Português (Portugal / Brasil)

**Projeto:** Sistema automático de resolução de problemas de matemática  
**Ambiente:** Windows 10, Python 3.11.9, NVIDIA GeForce RTX 3050 Ti (4.3 GB VRAM), CUDA 12.4

---

#### Âmbito

A primeira versão foi um notebook Google Colab monolítico com PDF, OCR, LLM e verificação no mesmo ficheiro. O primeiro passo foi modularizar num projeto Python local: cada etapa tornou-se um módulo, criou-se configuração central e uma CLI, permitindo testar e otimizar cada camada separadamente.

---

#### Arquitetura do pipeline

O pipeline tem **sete camadas**. **Camada 0** (PyMuPDF) extrai texto, metadados e imagens de página (~0,02 s em média; 58/58 PDFs sem falha). O texto passa por **sete verificações de qualidade**; a resolução das imagens subiu de 300 para **400 DPI**.

**Camada 1** usa um classificador heurístico com regex: palavras-chave, categoria **primária** (família de integrais de superfície ou tipos de matemática geral), sinais **secundários** opcionais e tipo de superfície quando aplicável. O **domínio** derivado (`surface_integral` ou `general_math`) escolhe o *system prompt* da Camada 5.

**Camada 2 (Nougat-OCR):** arquitetura Swin + mBART, LaTeX a partir de pixeis. Integração difícil: conflito com Albumentations → `_setup_albumentations_bypass()`; validação do `generate()` → monkey-patch. Sucesso em **42/58 PDFs (72,4%)**; 17 com saída vazia ou `[repetition]`; ~10,7 s/PDF. **Camada 3 (VLM)** — no setup atual, a extração visual usa Gemini; **100%** nos 58 PDFs; ~2,4 s/PDF (~4,5× mais rápido que Nougat local neste hardware). Prompt “só ler, não resolver”; **duas passagens** quando a primeira não atinge o máximo da rubrica; `clean_output()` remove passos de solução.

**Camada 4** funde fontes (tripla, VLM+prioridade, Nougat+prioridade, só texto) e *hints* por classe de problema. **Camada 5:** testados Claude, Together Llama 3.3 70B, GPT-4o, Gemini 2.5 Flash; **Gemini primário**, **Together fallback**. **Camada 6** extrai uma linha final para a UI (`\boxed{}`, `FINAL_ANSWER:`, heurísticas no fim do texto); o nome da classe ainda remete a SymPy por histórico. A **comparação numérica entre tentativas** do solver usa `latex_parser` em `run.STEPSolver._solve_with_consensus`.

---

#### Problemas e soluções

Nougat: módulo falso + *patch* do `generate` + VLM quando há `[repetition]`. VLM: `clean_output()` apagava o enunciado — lista de frases corrigida. Extração: Gemini sem `\boxed{}` consistente — extração em sete níveis + *follow-up*.

---

#### Atualizações de execução e fiabilidade

**Camadas 0 e 2 — reutilização de raster:** Sem argumento `dpi`, a camada 0 rasteriza à **`NOUGAT_DPI`** definida em configuração, alinhando com o Nougat. Cada pasta de `page_*.png` inclui **`.step_raster_meta`** (linha 1: DPI; linha 2: SHA-256 do PDF). A camada 2 só reutiliza esses PNGs quando o *sidecar*, os nomes das páginas e o ficheiro PDF atual coincidem; caso contrário volta a rasterizar e atualiza o *sidecar*. **`page_*.png`** é ordenado **numericamente** para manter a ordem correta após a página 9.

**Camada 3 — paralelismo VLM:** Chamadas por página podem usar um **conjunto limitado de *workers***. **`STEP_VLM_PAGE_WORKERS`** controla o paralelismo (use `1` para chamadas totalmente sérias ou para limitar pedidos por minuto).

**Interface web:** Várias resoluções em simultâneo ficam limitadas por um **semáforo** (**`STEP_WEB_MAX_CONCURRENT_SOLVES`**, por omissão 2). Um segundo *upload* com o **mesmo nome sanitizado** enquanto a primeira tarefa corre devolve **HTTP 409**. Reduza o limite quando o Nougat e a GPU estiverem sob pressão.

**Camada 6 — limpeza da resposta:** A extração deixa de partir a cadeia em `=` de forma ingénua. O código respeita a **profundidade de chavetas** e só considera o último `=` ao nível superior, evitando cortar texto dentro de `\text{...}` e construções semelhantes.

**Fornecedor LLM — Groq substituído por Together.ai:** A API do Groq tornou-se instável durante incidentes do lado do fornecedor para a nossa carga, e a oferta de visão não correspondia ao que precisávamos para LaTeX estável a partir de imagem. A configuração aponta agora para **Together.ai** (`TOGETHER_API_KEY`, `TOGETHER_MODEL`) como *fallback* de **chat** compatível com OpenAI, junto do **Gemini**. O VLM em operação normal permanece no **Gemini**; identificadores de visão na Together continuam opcionais e podem fazer *fallback* automático de modelo quando o escolhido não está disponível em modo *serverless*.

**Camada 3 — escrita manual / consistência de exportação:** As imagens de página são **normalizadas antes de cada pedido VLM** (tons de cinzento, autocontraste, suavização de nitidez, escala fixa no lado maior). Dois PDFs do mesmo problema manuscrito mas exportados com dimensões diferentes davam integrais lidos diferentes; este pré-processamento reduz essa dispersão.

**Solver (`run.py`) — fiabilidade e integrais definidos:** O consenso corrige `KeyError` em registos de tentativas incompletos, muda mais depressa para a Together após erros transitórios quando já existe uma resposta utilizável, e evita o LLM quando não há matemática legível após OCR/VLM. Para **integrais definidos** pode correr **uma** passagem de controlo no LLM e depois o **SymPy** avalia o integral a partir do *prompt* analisado e **substitui o resultado final mostrado** (muitas vezes reescrito com `log` em vez de `acosh`). Respostas com aspeto de passo intermédio disparam um *refine* com **limite de tempo** para a UI web não ficar bloqueada.

**Texto Gemini — modelo de recurso opcional:** Defina **`GEMINI_FALLBACK_MODEL`** (por exemplo `gemini-2.5-flash`) para repetir o mesmo conteúdo noutro ID Gemini se a chamada principal falhar.

**Interface web:** O *layout* **dá prioridade ao perfil e à resposta** em relação à coluna do PDF. Uma faixa **Simple explanation** resume tipo de problema, método e ideia-chave; a solução completa fica expandida por defeito. **`STEP_WEB_USE_NOUGAT`** / **`STEP_WEB_USE_VLM`** ativam ou desativam etapas no *worker* Flask.

**Rasterização adaptativa:** Quando o Nougat está desativado, as páginas são rasterizadas a **`VLM_ONLY_DPI`** (220 por defeito, ajustável com `STEP_VLM_ONLY_DPI`) em vez de 400 DPI. O VLM normaliza as imagens internamente, por isso o DPI alto era trabalho desperdiçado no caminho habitual só com VLM. A extração em Markdown via PyMuPDF4LLM também passa a correr apenas quando o texto nativo é escasso, evitando uma segunda abertura do PDF nos restantes casos.

**Camada 3 — reexecuções dirigidas e cache em disco:** A segunda passagem repete apenas as páginas cuja saída individual esteja vazia ou demasiado curta, em vez de voltar a processar todo o documento. O resultado limpo é guardado em **`step_pipeline/vlm_output/<pdf>/<pdf>.vlm.mmd`** com um *fingerprint* SHA-256 de todos os PNG de página; entradas idênticas em pixel saltam por completo a etapa VLM em execuções seguintes.

**Camada 5 — evitar a chamada extra do *boxed*:** Se a resposta não trouxer `\boxed{}` mas a Camada 6 já conseguir extrair a linha final (`FINAL_ANSWER:`, blocos de matemática no fim, heurísticas de cauda), a chamada adicional ao LLM deixa de ser feita, poupando uma ida ao modelo em cada resolução.

**Perfil de velocidade para Gemini 2.5 Pro:** Com `gemini-2.5-pro` como primário, o consenso executa apenas uma tentativa forte em vez de empilhar duas chamadas longas. Os restantes fornecedores mantêm o orçamento de consenso anterior.

**Solver partilhado na aplicação web:** Agora é construído um único `STEPSolver` por processo e reutilizado em cada *upload*, pelo que os clientes Gemini/Together e a cache da Camada 3 persistem entre pedidos. A cache em memória de resultados está protegida por um *lock* dedicado para que resoluções concorrentes não causem corrida no `OrderedDict`.

**Taxonomia de classificação (estilo MathE):** Um novo módulo `taxonomy.py` mapeia cada problema para um **Tópico** (Integração, Derivação, Limites e Continuidade, Séries, Equações Diferenciais, Álgebra Linear, Cálculo Vetorial, Álgebra), um **Subtópico** (Integrais Indefinidos, Integrais Definidos, Integrais de Superfície, …) e até cinco **Palavras-chave** controladas (Substituição, Integração por partes, Regra da potência, Regra da soma, Integração logarítmica, …). Depois do LLM responder, a sua solução e resumo são re-analisados contra o conjunto do subtópico escolhido, para que as palavras-chave reflitam as técnicas realmente usadas. Exposto na API como `result.taxonomy = {topic, subtopic, keywords}`.

**Pergunta do utilizador no *upload*:** O endpoint `/upload` aceita um campo opcional **`user_query`** (até 600 caracteres), que é acrescentado ao *prompt* da Camada 4 com um marcador explícito e devolvido no resultado. O *digest* da pergunta entra na chave da cache, pelo que perguntas diferentes produzem execuções diferentes.

**Conversa de seguimento:** `/followup` é um endpoint JSON síncrono que chama `STEPSolver.ask_followup` com o *prompt* original, a solução anterior e o *system prompt* guardados para a tarefa. Cada chamada é uma só passagem ao LLM (sem consenso) e devolve `{solution, final_answer, elapsed_s, model_used}`. A interface transforma o *composer* num *chat* de seguimento assim que uma resolução termina, adicionando cada pergunta e resposta como um cartão próprio.

**Interface web — resolução automática e foco na resposta:** O *upload* de um PDF arranca imediatamente o *solver*; o *composer* fica inativo até o processo terminar e só depois entra em modo de seguimento. O painel direito passa a começar pela resposta, seguida de um cartão dedicado à **Classificação** (Tópico, Subtópico, Palavras-chave), um eco da instrução do utilizador (quando existe), o resumo do modelo e uma tabela **Pipeline details** recolhível para os campos técnicos que antes sobrecarregavam o cartão de perfil. A renderização da solução completa remove uma linha final isolada com `\boxed{...}` (a resposta já está destacada em cima) e envolve os restantes blocos `\boxed{}` soltos em delimitadores de matemática em bloco para o MathJax os compor corretamente.

---

#### Escolhas tecnológicas

PyMuPDF pela velocidade; Nougat para LaTeX a partir de imagens; Gemini como solver principal e caminho VLM ativo; Together Llama 3.3 70B como *fallback*; SymPy (via `latex_parser`) para alinhar respostas numéricas entre tentativas; Flask + SSE; MathJax 3 no browser.

---

### Türkçe

**Proje:** Matematik problemi otomatik çözüm sistemi  
**Ortam:** Windows 10, Python 3.11.9, NVIDIA GeForce RTX 3050 Ti (4,3 GB VRAM), CUDA 12.4

---

#### Kapsam

Sistemin ilk versiyonu bir Google Colab notu olarak geliştirilmiştir; PDF işleme, OCR, LLM ve doğrulama tek dosyadaydı. İlk adım olarak bu monolitik yapı modüler bir yerel Python projesine dönüştürülmüştür. Her katman bağımsız modül, merkezi yapılandırma ve komut satırı arayüzü ile katmanların ayrı ayrı test ve iyileştirilmesi mümkün hale gelmiştir.

---

#### Pipeline mimarisi

**Katman 0 (PDF ingestion):** PyMuPDF ile ham metin, metadata ve yüksek çözünürlüklü görseller; ortalama ~0,02 s; 58 PDF’de hatasız çalışma. Metin 7 kontrol noktasından geçer; görüntü DPI başlangıçta 300, OCR için **400**’e çıkarılmıştır.

**Katman 1 (Profiling):** Heuristik regex; anahtar kelimeler, **birincil** kategori (yüzey integrali ailesi veya genel matematik türleri), isteğe bağlı **ikincil** sinyaller ve uygunsa yüzey tipi. Türetilen **domain** (`surface_integral` / `general_math`) Katman 5 sistem promptunu seçer.

**Katman 2 (Nougat-OCR):** Swin Transformer + mBART; pikselden LaTeX. Albumentations çakışması → `_setup_albumentations_bypass()`; transformers `generate()` doğrulaması → monkey-patch. **42/58 PDF (%72,4)** başarı; 17 PDF’de boş çıktı veya `[repetition]`; ortalama ~10,7 s/PDF. **Katman 3 (VLM):** mevcut kurulumda görsel çıkarım Gemini ile yapılır; **58/58 PDF (%100)**; Nougat’ın düştüğü 16 PDF’yi kurtarır; ortalama ~2,4 s/PDF (Nougat’a göre ~4,5× daha hızlı). Prompt: “sadece oku, çözme”; birinci geçiş zaten tam puan değilse **ikinci VLM geçişi**; `clean_output()` çözüm cümlelerini budar.

**Katman 4 (Synthesis):** Nougat + VLM + ham metin stratejileri; akı gibi problemlerde otomatik ipuçları; yüzey integrali için 8 adımlı prosedür, genel matematik için esnek prompt; ortalama ~3200 karakter.

**Katman 5 (LLM):** Claude (gecikme/kalite) → Together Llama 3.3 70B (hızlı yedek) → GPT-4o (sınırlı test) → **Gemini 2.5 Flash** (%100 test seti, daha yavaş). **Birincil: Gemini, yedek: Together.**

**Katman 6:** LLM çıktısından arayüz için tek bir **final satır** çıkarımı (`\boxed{}`, `FINAL_ANSWER:`, kuyruk heuristikleri). Sınıf adı geçmişten “SymPyVerifier” olsa da bu katmanda referans cevap doğrulaması yok. **Yinelemeli denemelerde sayısal örtüşme** `run.STEPSolver._solve_with_consensus` içinde `latex_parser.parse_latex_to_value` ile yapılır.

---

#### Hatalar ve çözümler

Nougat: sahte modül + `generate` yaması + VLM yedek. VLM: `clean_output()` en başta problem metnini siliyordu — kill-phrase revizyonu; çıktı uzunluğu ~%570 arttı. Cevap çıkarma: Gemini `\boxed{}` tutarsızlığı — 7 kademeli strateji + takip çağrısı.

---

#### Çalışma zamanı ve güvenilirlik güncellemeleri

**Katman 0 / 2 — raster yeniden kullanım:** `dpi` verilmezse Katman 0, yapılandırmadaki **`NOUGAT_DPI`** ile raster üretir; Nougat ile uyum korunur. Her `page_*.png` klasöründe **`.step_raster_meta`** bulunur (1. satır: DPI, 2. satır: PDF SHA-256). Katman 2, bu dosya ve sayfa adları güncel PDF ile örtüşmedikçe PNG’leri **yeniden çizer**; örtüşürse **yeniden rasterize etmeden** kullanır. **`page_*.png`** listesi **sayısal** sıralanır; 9. sayfanın üzerindeki PDF’lerde sıra bozulmaz.

**Katman 3 — VLM eşzamanlılığı:** Sayfa başına API çağrıları **sınırlı iş parçacığı havuzu** ile yürütülebilir. Yoğun RPM veya tam seri çalışma için **`STEP_VLM_PAGE_WORKERS`** (ör. `1`) ayarlanabilir.

**Web arayüzü:** Eşzamanlı çözüm sayısı **`STEP_WEB_MAX_CONCURRENT_SOLVES`** ile sınırlanır (varsayılan 2). Bitmemiş bir görev varken **aynı güvenli dosya adıyla** ikinci yükleme **HTTP 409** döner. Nougat ve GPU yükü yüksekse limit düşürülebilir.

**Katman 6 — cevap temizliği:** Final metin çıkarımında düz `split('=')` kullanılmaz; **süslü parantez derinliği** ile yalnızca üst düzeydeki son `=` dikkate alınır; `\text{...}` gibi yapılar budanmaz.

**LLM sağlayıcı — Groq yerine Together.ai:** Groq API’si sağlayıcı tarafı kesintilerinde iş yükümüz için güvenilmez hale geldi; görüntüden LaTeX için ihtiyaç duyduğumuz görsel yol da üretim beklentilerimize uymadı. Yapılandırma artık **Together.ai** (`TOGETHER_API_KEY`, `TOGETHER_MODEL`) ile OpenAI uyumlu **sohbet** yedeğini **Gemini** yanında kullanıyor. Normal çalışmada VLM **Gemini** üzerinde kalır; Together görsel model kimlikleri deneysel kalır ve seçilen *serverless* model yoksa otomatik model yedeğine düşer.

**Katman 3 — el yazısı / dışa aktarım tutarlılığı:** Her VLM çağrısından önce sayfa görselleri **normalize edilir** (gri tonlama, otokontrast, hafif keskinleştirme, uzun kenar sabit ölçek). Aynı el yazısı sorununun farklı piksel boyutlarında export edilmiş iki PDF’si farklı integral okuması üretebiliyordu; bu ön işleme sapmayı azaltır.

**Çözücü (`run.py`) — güvenilirlik ve belirli integraller:** Konsensüs, eksik deneme kayıtlarında `KeyError` riskini giderir; geçici hatalardan sonra zaten geçerli bir cevap varken Together’a daha hızlı geçer; OCR/VLM sonrası okunabilir matematik yoksa LLM atlanır. **Belirli integral** ve sentezlenen metinden ayrıştırılabiliyorsa **tek kontrol** LLM turu çalıştırılır, ardından **SymPy** integrali hesaplar ve **gösterilen finali** günceller (çoğu zaman `acosh` yerine daha okunaklı `log` biçimi). Ara adım görünümlü sonuçlarda **süre sınırlı** *refine* çağrısı ile web arayüzünün kilitlenmesi engellenir.

**Gemini metin — isteğe bağlı yedek model:** Birincil `GEMINI_MODEL` çağrısı düşerse aynı içeriği başka bir kimlikte denemek için **`GEMINI_FALLBACK_MODEL`** (ör. `gemini-2.5-flash`) ayarlanabilir.

**Web arayüzü:** Yerleşim **profilleme ve cevabı** PDF sütununa göre öne alır. **Simple explanation** şeridi problem türü, yöntem ve ana fikri özetler; tam çözüm okunabilirlik için varsayılan olarak açıktır. **`STEP_WEB_USE_NOUGAT`** / **`STEP_WEB_USE_VLM`** Flask işçisinde katmanları açıp kapatır.

**Uyarlanabilir rasterize:** Nougat kapalı olduğunda sayfalar 400 DPI yerine **`VLM_ONLY_DPI`** (varsayılan 220, `STEP_VLM_ONLY_DPI` ile değiştirilebilir) ile raster edilir. VLM görüntüyü zaten iç ölçeklendirmeden geçirdiği için, yalnızca VLM kullanılan akışta 400 DPI gereksizdi. PyMuPDF4LLM markdown çıkarımı da yalnızca yerel metin katmanı zayıfsa çağrılır; aksi halde ikinci PDF açılışı ortadan kalkar.

**Katman 3 — hedefli yeniden deneme ve disk cache:** İkinci geçiş tüm sayfaları değil, çıktısı boş veya çok kısa olan sayfaları yeniden işler. Temizlenmiş VLM çıktısı **`step_pipeline/vlm_output/<pdf>/<pdf>.vlm.mmd`** altında, tüm sayfa PNG’lerinin SHA-256 parmak iziyle saklanır; aynı piksel girdi geldiğinde VLM aşaması tümüyle atlanır.

**Katman 5 — gereksiz boxed turunu atla:** Cevap `\boxed{}` içermese bile Katman 6 son satırı (`FINAL_ANSWER:`, display math, kuyruk heuristikleri) çıkarabiliyorsa ek LLM çağrısı yapılmaz; her çözümde bir model turu tasarrufu sağlanır.

**Gemini 2.5 Pro hız profili:** Birincil `gemini-2.5-pro` olduğunda konsensüs iki uzun çağrı yerine tek güçlü deneme ile koşar. Diğer sağlayıcılarda önceki konsensüs bütçesi korunur.

**Web arayüzünde paylaşılan çözücü:** Artık süreç başına tek bir `STEPSolver` gerektiğinde kurulur ve her yükleme için yeniden kullanılır; Gemini/Together istemcileri ve Katman 3 kaynakları istekler arasında kalır. Bellek içindeki sonuç cache’i özel bir kilitle korunur, böylece eşzamanlı çözümler `OrderedDict` üzerinde yarışmaz.

**Sınıflandırma taksonomisi (MathE tarzı):** Yeni `taxonomy.py` modülü her problemi bir **Topic**’e (Integration, Differentiation, Limits and Continuity, Series, Differential Equations, Linear Algebra, Vector Calculus, Algebra), bir **Subtopic**’e (Indefinite Integrals, Definite Integrals, Surface Integrals, …) ve en fazla beş kontrollü **Keyword**’e (Substitution, Integration by parts, Power rule, Sum rule, Logarithmic integration, …) eşler. LLM cevabı geldikten sonra kendi çözümü ve özeti seçilen alt başlığın havuzuyla yeniden taranır; böylece anahtar kelimeler modelin gerçekten kullandığı teknikleri yansıtır. API’de `result.taxonomy = {topic, subtopic, keywords}` olarak sunulur.

**Yüklemede kullanıcı sorusu:** `/upload` isteğe bağlı bir **`user_query`** alanı kabul eder (en fazla 600 karakter). Bu metin, Katman 4 promptunun sonuna açık bir etiketle eklenir ve cevapla birlikte geri döner. Metnin özeti cache anahtarına dahil edilir; farklı talimatlar farklı çalıştırmalar üretir.

**Takip sohbeti:** `/followup` senkron bir JSON uç noktasıdır; görevin saklanan orijinal promptunu, önceki çözümünü ve sistem promptunu kullanarak `STEPSolver.ask_followup`’i çağırır. Her çağrı tek LLM turudur (konsensüs yok) ve `{solution, final_answer, elapsed_s, model_used}` döner. Web arayüzü, bir çözüm tamamlandığında composer’ı follow-up sohbetine çevirir ve her soru/cevap çiftini kendi kartı olarak ekler.

**Web arayüzü — otomatik çözüm ve cevap öncelikli düzen:** PDF yükler yüklemez çözüm başlar; composer, çözüm tamamlanana kadar pasif kalır ve ardından takip moduna geçer. Sağ panel önce cevabı, ardından MathE sorusuna benzeyen bir **Classification** kartını (Topic, Subtopic, Keywords), varsa kullanıcının talimatını yansıtan kartı, model özetini ve teknik alanların toplandığı katlanabilir **Pipeline details** tablosunu gösterir. Tam çözüm render’ında sonda yalnız başına duran `\boxed{...}` satırı (cevap yukarıda büyük panelde zaten görünüyor) kaldırılır; gövdede kalan `\boxed{}` blokları ise display-math delimiter’larına sarılarak MathJax tarafından düzgünce dizilir.

---

#### Teknoloji seçimleri

PyMuPDF (hız); Nougat (görüntüden LaTeX); Gemini (ana metin çözücü ve aktif VLM yolu); Together Llama 3.3 70B (yedek); SymPy tabanlı `latex_parser` (denemeler arası sayısal karşılaştırma); Flask + SSE; MathJax 3.

---

## Run this project locally

### Setup

```bash
cd Step_Project
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env
# or: cp .env.example .env
```

Put at least `GEMINI_API_KEY` and `TOGETHER_API_KEY` in `.env`.

Optional tuning (all optional; defaults work for a typical single-user setup):

- `STEP_VLM_PAGE_WORKERS` — cap parallel per-page vision calls (use `1` if you prefer serial VLM or need to stay under strict API rate limits).
- `STEP_WEB_MAX_CONCURRENT_SOLVES` — cap overlapping PDF solves in the Flask app (lower it when GPU-backed Nougat runs alongside the web UI).

### Run

**Web UI (upload a PDF):**

```bash
python web_app.py
```

Open `http://127.0.0.1:5000`.

**CLI:**

```bash
python run.py path/to/problem.pdf
python run.py path/to/problem.pdf --with-nougat
python run.py --check
python run.py Surface_Integration/
```

Docs, reports, and extra tooling: **`Step_Project_Disari/`** in the repo root.

Editable narrative source (same content you maintain): **`../STEP_Pipeline_Rapor_Paragraf.md`** (one level above this folder).
