# Pilot transcript metrics report (3 videos × 3 methods)

Generated: `2026-05-29T20:30:54Z` (UTC)

Methods compared:
- **Gemini** — `gemini-3-flash-preview`, optimized math-education prompts (`mathe-transcribe-v2`)
- **Whisper** — `distil-large-v3` (best local ASR tier), beam search + math initial prompt
- **YouTube captions** — platform auto/manual subtitles (baseline)

> **Note:** YouTube captions for `KTNcYYHuBTY` could not be refreshed (HTTP 429 rate limit). LVL and uKfc caption rows are from this run.

## Summary table

| Video | Method | Model | Words | Chars | Duration (s) | Elapsed (s) | RTF | WPM | KW recall |
|-------|--------|-------|------:|------:|-------------:|------------:|----:|----:|----------:|
| KTNcYYHuBTY | Gemini (optimized) | gemini-3-flash-preview | 563 | 2978 | 279.2 | 56.7 | 0.203 | 121.0 | 20.0% (1/5) |
| KTNcYYHuBTY | Whisper (optimized) | distil-large-v3 | 646 | 3220 | 279.2 | 13.4 | 0.048 | 138.8 | 20.0% (1/5) |
| LVLuqNH5iWw | Gemini (optimized) | gemini-3-flash-preview | 323 | 2259 | 441.9 | 95.2 | 0.215 | 43.9 | 80.0% (4/5) |
| LVLuqNH5iWw | Whisper (optimized) | distil-large-v3 | 628 | 2954 | 441.9 | 19.2 | 0.043 | 85.3 | 60.0% (3/5) |
| LVLuqNH5iWw | YouTube captions | auto | 577 | 2412 | 442.0 | 3.4 | 0.008 | 78.3 | 60.0% (3/5) |
| uKfcS7-O6UE | Gemini (optimized) | gemini-3-flash-preview | 364 | 2068 | 252.1 | 112.4 | 0.446 | 86.6 | 20.0% (1/5) |
| uKfcS7-O6UE | Whisper (optimized) | distil-large-v3 | 506 | 2346 | 252.1 | 10.8 | 0.043 | 120.4 | 20.0% (1/5) |
| uKfcS7-O6UE | YouTube captions | auto | 448 | 2036 | 252.0 | 3.4 | 0.013 | 106.7 | 20.0% (1/5) |

RTF = real-time factor (elapsed ÷ audio duration; lower is faster). WPM = words per minute estimated from transcript length. KW recall = share of graded VLM reference keywords literally present in the transcript.

## KTNcYYHuBTY — Powers of the Imaginary Unit i

**VLM summary (existing pipeline):**

This video introduces the imaginary unit i as the square root of -1 and calculates its integer powers. It demonstrates that the powers of i follow a repeating cycle of four values: i, -1, -i, and 1.

**Reference keywords (graded):** Powers of i, Powers of complex numbers, Imaginary part, Operations with complex numbers, Nth root

### Gemini (optimized)

- model/config: `gemini-3-flash-preview`
- words / chars: **563** / **2978**
- duration_s: 279.2
- elapsed_s: 56.7
- real_time_factor: 0.203
- words_per_minute: 121.0
- paragraphs: 8
- keyword recall: **20.0%** (1/5)
- prompt_version: `mathe-transcribe-v2`
- quality pass: imaginary_unit_i, sqrt_negative_one
- keywords found: Powers of i
- keywords missing: Powers of complex numbers, Imaginary part, Operations with complex numbers, Nth root

**Preview:**

In the previous session, we understood the concept of a unit imaginary number i. It is the square root of $-1$. If we square it, we get $\sqrt{-1} \times \sqrt{-1}$ which equals $-1$. The square of the unit imaginary number is $-1$. Well, let us now try to work out more powers of i. What will be $i^3$ then? Try it out. $i^3$ is $i \times i \times i$. We can write it as $i^2 \times i$. Now as $i^2 = -1$, $i^3$ is $-1 \times i$ or $-i$. So the value of $i^3$ is $-i$. You can now similarly work out fourth power of i. Give it a try. The fourth power of i can be written as $i^2 \times i^2$. This is nothing but $-1 \times -1$. And we already know that $-1 \times -1$ is $1$. Hence the fourth power of i is just $1$. So $i^2 = -1$, $i^3 = -i$, and fourth power of i is $1$. Can you similarly work...


### Whisper (optimized)

- model/config: `distil-large-v3`
- words / chars: **646** / **3220**
- duration_s: 279.2
- elapsed_s: 13.4
- real_time_factor: 0.048
- words_per_minute: 138.8
- paragraphs: 1
- keyword recall: **20.0%** (1/5)
- segments: 67
- prompt_version: `mathe-transcribe-v2`
- device: `cuda`
- beam_size: 5
- quality pass: imaginary_unit_i, sqrt_negative_one
- keywords found: Powers of i
- keywords missing: Powers of complex numbers, Imaginary part, Operations with complex numbers, Nth root

**Preview:**

In the previous session, we understood the concept of a unit imaginary number i. It is the square root of negative 1. If we square it, we get root of negative 1 times root of negative 1, which equals negative 1. The square of the unit imaginary number is negative 1. Well, let us now try to work out more powers of i. What will be i cubed then? Try it out. I cubed is i times i times i. times i. We can write it as i squared multiplied by i. Now as i squared is equal to negative 1, i cubed is negative 1 multiplied by i or negative i. So the value of i cubed is negative i. You can now similarly work out fourth power of i? Give it a try. The fourth power of i can be written as i squared multiplied by i squared. This is nothing but negative 1 multiplied by negative And we already know that neg...


---

## LVLuqNH5iWw — Local Extrema of a Multivariable Function

**VLM summary (existing pipeline):**

The video explains how to find the local extrema of a function with two variables. It demonstrates the process by first finding the critical points and then classifying them using the second derivative test.

**Reference keywords (graded):** Local maximum, Local minimum, Second derivative test, Critical point, Saddle point

### Gemini (optimized)

- model/config: `gemini-3-flash-preview`
- words / chars: **323** / **2259**
- duration_s: 441.9
- elapsed_s: 95.2
- real_time_factor: 0.215
- words_per_minute: 43.9
- paragraphs: 7
- keyword recall: **80.0%** (4/5)
- prompt_version: `mathe-transcribe-v2`
- quality pass: saddle_point_correct, critical_points, partial_derivatives
- keywords found: Local maximum, Local minimum, Critical point, Saddle point
- keywords missing: Second derivative test

**Preview:**

Find the local extrema of the function $f:\mathbb{R}^2\to\mathbb{R}$, $f(x,y)=x^4+y^4-x^2-y^2+3$. So first of all we have to find the critical points, yeah, let's write the critical points are the solutions of the system consisting of the partial derivatives of $f$ with respect to $x$ and $y$. But this is equivalent to in our case $4x^3-2x=0$ and $4y^3-2y=0$ or equivalently $2x(2x^2-1)=0$ and $2y(2y^2-1)=0$. So in this case we have the following solutions: $x_1=0$, $x_1=1/\sqrt{2}$ and $x_2=-1/\sqrt{2}$, $y_1=0$, $y_2=1/\sqrt{2}$, $y_3=-1/\sqrt{2}$. So totally we have nine points, so we have 9 critical points which are $M_{ij}(x_i,y_j)$ where $i,j=1,3$. So the points are $M_{11}(0,0)$, $M_{12}(0,1/\sqrt{2})$, $M_{13}(0,-1/\sqrt{2})$, $M_{21}(1/\sqrt{2},0)$, $M_{22}(1/\sqrt{2},1/\sqrt{2}...


### Whisper (optimized)

- model/config: `distil-large-v3`
- words / chars: **628** / **2954**
- duration_s: 441.9
- elapsed_s: 19.2
- real_time_factor: 0.043
- words_per_minute: 85.3
- paragraphs: 1
- keyword recall: **60.0%** (3/5)
- segments: 34
- prompt_version: `mathe-transcribe-v2`
- device: `cuda`
- beam_size: 5
- quality pass: critical_points, partial_derivatives
- quality miss: saddle_point_correct
- **quality error detected:** saddle_point_wrong
- keywords found: Local maximum, Local minimum, Critical point
- keywords missing: Second derivative test, Saddle point

**Preview:**

Find the local extrema of the function, f from r2 into r real function, f depending by two variables, x and y, which will be x to the power of four, plus y to the power of four, minus x squared, minus y squared, plus three. So first of all, we have to find the critical points, yeah? the critical points are the solutions of the system consisting of the partial derivatives of f with respect to x and y. But this is equivalent to, in our case, 4x cube minus 2x equals to 0, and 4 y cube minus 2y equals to 0, or equivalently 2x multiplied by 2x multiplied by 2x,000, squared minus 1 equals to 0 and 2 y squared minus 1 equals to 0 so in this case we have the following solutions x1 equals to 0 x1 1 over radical of 2 and x2 equals minus 1 over radical of 2 y 1 equals to 0 y 2 1 over radical of 2 ...


### YouTube captions

- model/config: `auto`
- words / chars: **577** / **2412**
- duration_s: 442.0
- elapsed_s: 3.4
- real_time_factor: 0.008
- words_per_minute: 78.3
- paragraphs: 1
- keyword recall: **60.0%** (3/5)
- quality pass: critical_points, partial_derivatives
- quality miss: saddle_point_correct
- **quality error detected:** saddle_point_wrong
- keywords found: Local maximum, Local minimum, Critical point
- keywords missing: Second derivative test, Saddle point

**Preview:**

find the local extrema of the function f r 2 into R real function f depending by two variables X and Y which will be X to the^ of 4 + y ^ of 4 - x^ 2 - y^ 2 + three so first of all we have to find the critical points yeah let's write the critical points are the Solutions of the system consisting of the partial derivatives of f with respect to X and Y but this is equivalent to in our case 4X cubus 2x = to 0 and for y Cub - 2 y = to 0 or equivalently 2x multili by H 2x^ 2- - 1 = to 0 and 2 Y 2 y^ 2 - 1 = to 0 so in this case we have the following Solutions X1 equal to 0 X1 1/ radical of 2 and X2 = -1 / radical of 2 y1 = to 0 Y2 1 / radical of 2 Y3 - 1 / radical of two so totally we have nine points so we have nine critical points which are M let's say m i j of x i y j where I and J runs o...


---

## uKfcS7-O6UE — Quotient Rule for Differentiation

**VLM summary (existing pipeline):**

This video explains the formula for the quotient rule, which is used to find the derivative of a function expressed as a fraction. A step-by-step example is worked through to demonstrate the application of the rule.

**Reference keywords (graded):** Quotient rule, Constant rule, Power rule, Sum/Difference rule, Leibniz notation

### Gemini (optimized)

- model/config: `gemini-3-flash-preview`
- words / chars: **364** / **2068**
- duration_s: 252.1
- elapsed_s: 112.4
- real_time_factor: 0.446
- words_per_minute: 86.6
- paragraphs: 8
- keyword recall: **20.0%** (1/5)
- prompt_version: `mathe-transcribe-v2`
- quality pass: quotient_rule, dy_dx_notation
- keywords found: Quotient rule
- keywords missing: Constant rule, Power rule, Sum/Difference rule, Leibniz notation

**Preview:**

Hi, I'm Aisling Lynch, I'm a lecturer with Limerick Institute of Technology and as part of the Math-E project, I'm going to be recording a video today talking about the quotient rule. The quotient rule then is used to differentiate a function that consists of two functions, one divided by another. So, for example, if we have $y = \frac{2x+1}{3x-4}$, we can't use the general rule here, we must use a special rule called the quotient rule. So the quotient rule tells us that if $y = \frac{u}{v}$, in other words, one function divided by another function, then the derivative of $y$, $\frac{dy}{dx}$, is equal to $v$ multiplied by $\frac{du}{dx}$ minus $u$ multiplied by $\frac{dv}{dx}$ all over $v^2$. So what we have here is $v$, the lower function, multiplied by the derivative of $u$, minus $u...


### Whisper (optimized)

- model/config: `distil-large-v3`
- words / chars: **506** / **2346**
- duration_s: 252.1
- elapsed_s: 10.8
- real_time_factor: 0.043
- words_per_minute: 120.4
- paragraphs: 1
- keyword recall: **20.0%** (1/5)
- segments: 18
- prompt_version: `mathe-transcribe-v2`
- device: `cuda`
- beam_size: 5
- quality pass: quotient_rule
- quality miss: dy_dx_notation
- keywords found: Quotient rule
- keywords missing: Constant rule, Power rule, Sum/Difference rule, Leibniz notation

**Preview:**

Hi, I'm Aschling Lynch, I'm a lecturer in Limerick Institute of Technology, and as part of the Math E project, I'm going to be recording a video today talking about the quotient rule. The quotient rule then is used to differentiate a function that consists of two functions, one divided by another. So, for example, if we have y is equal to 2x plus 1 over 3x minus 4, we can't use the general rule here, we must use, use a special rule called a quotient rule. So the quotient rule tells us that if y is equal to u over v, in other words, one function divided by another function, then the derivative of y, d y, d x, is equal to v multiplied by d u d x, minus u multiplied by d u d x, all over v. squared. So what we have here is v, the lower function, multiplied by the derivative of u minus u, th...


### YouTube captions

- model/config: `auto`
- words / chars: **448** / **2036**
- duration_s: 252.0
- elapsed_s: 3.4
- real_time_factor: 0.013
- words_per_minute: 106.7
- paragraphs: 1
- keyword recall: **20.0%** (1/5)
- quality pass: quotient_rule, dy_dx_notation
- keywords found: Quotient rule
- keywords missing: Constant rule, Power rule, Sum/Difference rule, Leibniz notation

**Preview:**

hi I'm Ashley Lynch I'm a lecturer with Network Institute of Technology and as part as lat project I'm going to be recording a video today talking about the control the quotient rule then is used to differentiate a function that consists of two functions one divided by another so for example if we have y is equal to 2x plus 1 over 3x minus 4 we can't use the general rule here we must use a special rule called the quotient rule so the quotient rule tells us that if y is equal to u over V in other words one function divided by another function then the derivative of Y dy DX is equal to V multiplied by D u DX minus u multiplied by DV DX all over V squared so what we have here is V the lower function multiplied by the derivative of U minus u the upper function multiplied by the derivative o...


---
