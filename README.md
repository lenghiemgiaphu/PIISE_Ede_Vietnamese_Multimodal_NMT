<div align="center">

# Ede–Vietnamese Machine Translation

**Low-resource neural machine translation for Ede (Rade) via transfer learning on NLLB-200 with a proxy language code**

PIISE Research Program — Board B (Research Idea) | Pho Thong Nang Khieu (PTNK), VNU-HCM | Sep–Oct 2026

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/docs/transformers)
[![Base Model](https://img.shields.io/badge/Base%20Model-NLLB--200%20distilled%20600M-0467DF?style=flat-square)](https://huggingface.co/facebook/nllb-200-distilled-600M)
[![SacreBLEU](https://img.shields.io/badge/SacreBLEU-12.46-16A34A?style=flat-square)](#4-key-results--benchmarks)
[![Val Loss](https://img.shields.io/badge/Validation%20Loss-4.709%20%E2%86%92%202.888-2563EB?style=flat-square)](#4-key-results--benchmarks)
[![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-F59E0B?style=flat-square)](#7-scope-limitations--caveats)

[📄 Research Proposal (VI)](./docs/Research_Proposal.pdf) • [🖼️ Poster (VI)](./docs/Research_Poster.pdf) • [📈 Loss Curves](./results/training_validation_loss_curve.png)

</div>

---

## 1. Overview

Ede is an Austronesian (Chamic) language spoken by roughly 400,000 people in the Central Highlands of Vietnam (2019 Census). It has almost no parallel text, no official language code in NLLB-200, and no commercial machine-translation support. This repository documents a **proof-of-concept text-to-text pipeline** that asks one question: *how far can a small, automatically built Ede–Vietnamese corpus take a pretrained multilingual model on commodity (free-tier) hardware?*

**Contributions**

1. **Closed-loop data pipeline:** scraping of public Ede-language news → Unicode NFC normalization → LLM-assisted sentence pairing → bidirectional augmentation (**408 sentence pairs → 816 training samples**).
2. **Proxy-language fine-tuning:** Ede has no NLLB language token, so Indonesian (`ind_Latn`) is used as the source-side proxy and Vietnamese (`vie_Latn`) as the target, fine-tuning `facebook/nllb-200-distilled-600M`.
3. **Memory-bounded training recipe:** batch size 1 × gradient accumulation 8 with fp16 trains the 600M-parameter model on a single Colab T4 (16 GB) without out-of-memory errors.
4. **Quantified result:** validation loss falls from 4.709 to 2.888 over 328 optimizer steps; SacreBLEU reaches **12.46 / 100**. Qualitative analysis shows the model is **not yet reliable at sentence level** (see §4.3 and §7).

> **Scope.** The proposal describes a multimodal system (Text-to-Text + Text-to-Speech). Only the text-to-text component and a Gradio demo are implemented here; VITS-based speech synthesis is **planned, not built** (see Roadmap in §7).

---

## 2. System Architecture

```text
   VOV4 Ede-language news pages (public)
                  |
                  v
+-----------------------------------+
| 1. Web scraping                   |  requests + BeautifulSoup4
|    article-link regex filter      |  category / audio pages dropped
+-----------------+-----------------+
                  | raw Ede text (CSV, utf-8-sig)
                  v
+-----------------------------------+
| 2. Cleaning & normalization       |  Unicode NFC, HTML / regex cleanup,
|                                   |  punctuation-based sentence split
+-----------------+-----------------+
                  v
+-----------------------------------+
| 3. Pairing / labeling             |  Gemini 1.5 Flash, structured JSON
|    + manual cross-check           |  output -> Ede-Vietnamese CSV
+-----------------+-----------------+
                  | 408 sentence pairs
                  v
+-----------------------------------+
| 4. Bidirectional augmentation     |  swap source <-> target
|                                   |  -> 816 training samples
+-----------------+-----------------+
                  v
+-----------------------------------+
| 5. Fine-tune NLLB-200 (600M)      |  src = ind_Latn (Ede proxy)
|    AdamW, fp16, grad. accum. = 8  |  tgt = vie_Latn, Colab T4 16 GB
+-----------------+-----------------+
                  v
+-----------------------------------+      +--------------------------------+
| 6. Evaluation                     |      | 7. Gradio web demo             |
|    loss curves, SacreBLEU         |----->|    interactive translation     |
+-----------------------------------+      +----------------+---------------+
                                                            .
                                                            v   (planned)
                                           +--------------------------------+
                                           | 8. VITS text-to-speech (.wav)  |
                                           +--------------------------------+
```

---

## 3. Methodology

### 3.1 Data Collection & Preprocessing

| Step | Implementation |
| :--- | :--- |
| **Source** | Ede-language service of VOV4 (`vov4.vov.vn/ede`). The Ede page of Báo Đắk Lắk (`baodaklak.vn/tieng-e-de`) is identified as an additional source for Stage 2. |
| **Scraping** | `requests` + `BeautifulSoup4`; article URLs are selected by a regular expression on the numeric article-ID suffix so that category pages and audio pages are excluded. |
| **Normalization** | `unicodedata.normalize('NFC', text)` to unify composed and decomposed forms of Ede-specific characters (ê, ô, ơ, ư, ƀ, č, đ, ñ, ğ); regex removal of HTML remnants; sentence segmentation on standard punctuation. |
| **Pairing** | Gemini 1.5 Flash (`gemini-1.5-flash`) with structured-JSON output produces aligned Ede–Vietnamese pairs, written to CSV with `utf-8-sig`. The team read the exported pairs for meaning and font errors. |
| **Augmentation** | Each pair is used in both directions (Ede → Vietnamese and Vietnamese → Ede): 408 pairs → **816 samples**. |

### 3.2 Proxy Language Code

NLLB-200 covers 200 languages but has no token for Ede. A zero-shot probe on a CPU run confirmed two things: the tokenizer splits Ede text without font or character errors, but the untuned model essentially **copies** its input (it changed only the first letter of the test sentence). The fine-tuning setup therefore sets:

- `src_lang = "ind_Latn"` — Indonesian, a Latin-script Austronesian language, used as a stand-in so that the model can learn a new vocabulary under an existing language embedding;
- `tgt_lang = "vie_Latn"` — Vietnamese.

No controlled comparison against other proxy codes was run (see §7).

### 3.3 Fine-Tuning Configuration

| Parameter | Value |
| :--- | :--- |
| Base model | `facebook/nllb-200-distilled-600M` (≈600M parameters, encoder–decoder Transformer) |
| Language codes | `ind_Latn` (Ede proxy) → `vie_Latn` |
| Optimizer | AdamW, learning rate $5 \times 10^{-5}$, weight decay $0.01$ |
| Precision | fp16 |
| Per-device batch size | 1 |
| Gradient accumulation | 8 steps (effective batch size 8) |
| Training length | 4 epochs = 328 optimizer steps (82 per epoch) |
| Logging / evaluation interval | every 10 steps |
| Hardware | Google Colab, NVIDIA T4 (16 GB VRAM) |

Training minimizes the token-level cross-entropy

$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_{\theta}\left(y_t \mid y_{<t}, \, x\right)
$$

and gradient accumulation gives an effective batch size $B_{\text{eff}} = 1 \times 8 = 8$ while keeping peak GPU memory at the batch-size-1 level.

### 3.4 Evaluation

Translation quality is measured with SacreBLEU (`evaluate.load("sacrebleu")`):

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \ln p_n\right), \qquad BP = \begin{cases} 1, & c > r \\ e^{1 - r/c}, & c \le r \end{cases}$$

where $p_n$ is the modified $n$-gram precision, $c$ the candidate length and $r$ the reference length. Training dynamics are tracked through training and validation loss logged every 10 steps.

---

## 4. Key Results & Benchmarks

### 4.1 Training Runs

| Run | Training data | Epochs / steps | Validation loss | SacreBLEU |
| :--- | :--- | :---: | :--- | :---: |
| Zero-shot (untuned) | — | — | — | not quantified; output ≈ copy of input |
| **Stage 1 — pilot** | 30 sentences | 10 epochs | 3.883 (epoch 1) → min 3.704 (epoch 3) → 3.843 (epoch 10) | 9.39 *(logged 4 Oct 2026)* |
| **Stage 2 — scaled** | 408 pairs → 816 samples | 4 epochs / 328 steps | **4.709 (step 10) → 2.888 (step 328)** | **12.46** |

The pilot's training loss fell from 4.03 to 1.86 while validation loss stopped improving after epoch 3 and drifted upward — mild overfitting on a 30-sentence set, which motivated the data pipeline. The proposal's success criterion (BLEU > 12/100) is met by a narrow margin.

### 4.2 Loss Trajectory (Stage 2)

| Step | Training loss (as logged) | Training loss ÷ 8 *(derived)* | Validation loss |
| :---: | :---: | :---: | :---: |
| 10 | 46.81 | 5.85 | 4.709 |
| 50 | 30.59 | 3.82 | 3.570 |
| 100 | 26.57 | 3.32 | 3.218 |
| 150 | 24.33 | 3.04 | 3.052 |
| 200 | 21.61 | 2.70 | 2.964 |
| 250 | 21.39 | 2.67 | 2.910 |
| 300 | 21.56 | 2.69 | 2.893 |
| 328 | 21.09 | 2.64 | 2.888 |

**Reading the table.** The logged training loss is roughly eight times larger than the validation loss. This ratio equals `gradient_accumulation_steps`, which suggests the trainer reports the loss summed over the 8 accumulation micro-steps; the ÷ 8 column is a derived, comparable-scale view and is **not** a separately logged quantity. Validation loss decreases monotonically and flattens around 2.89 after step ≈ 250. The raw loss figure in [`results/`](./results/training_validation_loss_curve.png) plots the as-logged values, so its training and validation curves should not be read as being on the same scale.

### 4.3 Qualitative Analysis (Ede → Vietnamese, Stage 2 model)

| Ede source | Model output | Human reference |
| :--- | :--- | :--- |
| *Mâo dŭm mnuih ƀuôn sang amâo thâo dlăng, thâo čih klei yuăn.* | Trong đó, các nhà đầu tư đã xác định cách xử lý các vấn đề, giải quyết các vấn đề. | Một số người dân không biết lái xe, cũng không biết các quy tắc giao thông. |
| *Ară anei hlăm wăl anôk mnia mblei snăn hbâo pruê snăn djăp anôk, djăp mta* | Các cơ sở đã xác định cách thức sử dụng thuốc theo quy định của cơ sở và cơ sở. | Hiện nay trên thị trường, phân bón được bày bán ở khắp nơi với đủ mọi chủng loại. |

The outputs are fluent Vietnamese but **semantically unfaithful**, with repeated phrases ("các vấn đề … các vấn đề", "của cơ sở và cơ sở"). Informal inspection through the web demo showed that some domain keywords (e.g. *sầu riêng*, *cà phê*, *cây trồng*) are captured, while sentence-level structure is not. A BLEU of 12.46 should therefore be read as evidence of partial lexical learning, not of usable translation quality.

---

## 5. Repository Structure

```text
PIISE_Ede_Vietnamese_Multimodal_NMT/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── ede_raw_from_scraper.csv            # Scraped Ede text (VOV4)
│   └── processed/
│       └── ede_vi_parallel_corpus.csv          # Labeled Ede–Vietnamese pairs (408)
├── src/
│   ├── data_scraper.py                         # VOV4 scraper
│   └── plot_training_curves.py                 # Training / validation loss plot
├── notebooks/
│   └── nllb200_ede_vi_finetune.ipynb           # Colab: fine-tuning, BLEU, Gradio demo
├── results/
│   └── training_validation_loss_curve.png      # Loss curves (as-logged values)
└── docs/
    ├── Research_Proposal.pdf                   # PIISE research proposal (Vietnamese)
    └── Research_Poster.pdf                     # PIISE poster (Vietnamese)
```

Fine-tuned weights are **not** stored in this repository (a 600M-parameter checkpoint exceeds GitHub's file-size limit); regenerate them with the notebook as described below.

---

## 6. Installation & Reproducibility

### 6.1 Environment

```bash
git clone https://github.com/lenghiemgiaphu/PIISE_Ede_Vietnamese_Multimodal_NMT.git
cd PIISE_Ede_Vietnamese_Multimodal_NMT
pip install -r requirements.txt
```

Fine-tuning was run on a Google Colab **T4 GPU (16 GB)**. A local GPU with at least 16 GB of VRAM should behave similarly; CPU-only runs are suitable for tokenizer checks, not for training.

### 6.2 Reproducing the Pipeline

1. **Collect raw Ede text** (optional — the scraped CSV is already provided in `data/raw/`):

   ```bash
   python src/data_scraper.py
   ```

2. **Label Ede–Vietnamese pairs.** The labeling step calls the Gemini API. Provide your own key and never commit it:

   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```

   The processed corpus is already provided in `data/processed/`, so downstream steps do not require re-running this step. Model availability and outputs from hosted LLM APIs can change over time; re-labeling may therefore not reproduce the exact pairs.

3. **Fine-tune and evaluate.** Open `notebooks/nllb200_ede_vi_finetune.ipynb` in Google Colab, select a **T4 GPU** runtime, and run all cells. The notebook uses the configuration in §3.3 and reports validation loss every 10 steps and SacreBLEU at the end.

4. **Plot the loss curves:**

   ```bash
   python src/plot_training_curves.py
   ```

### 6.3 Data Notice

The Ede text originates from publicly accessible VOV4 articles and is used strictly for **non-commercial research**; all rights to the original articles remain with the publisher. The Vietnamese side was produced with LLM assistance (§8) and is not an official translation.

---

## 7. Scope, Limitations & Caveats

1. **Corpus size and domain.** 408 pairs from a single news outlet (mostly economic and agricultural topics, written register). Results say nothing about conversational Ede or other orthographic conventions.
2. **Silver-standard references.** The Vietnamese side was produced with Gemini assistance and cross-checked by the team. No validation by native Ede speakers or professional translators is documented, so BLEU measures agreement with these labels rather than with expert translations.
3. **Thin evaluation.** A single aggregate BLEU (12.46) was computed over 15 evaluation batches, with no confidence interval, repeated seeds, or per-direction (Ede → Vietnamese vs. Vietnamese → Ede) breakdown. No numeric zero-shot BLEU was recorded, so the improvement over the base model is supported qualitatively (the untuned model copies its input), not quantitatively.
4. **Poor sentence-level fidelity.** Outputs are fluent but often wrong and repetitive (§4.3).
5. **Loss-scale artifact.** The logged training loss is not on the same scale as the validation loss (§4.2). Claims about the train/validation gap must use the normalized column, and the plot in `results/` should be regenerated with normalized values for any future report.
6. **Proxy-code assumption.** Using `ind_Latn` for Ede rests on typological plausibility; no ablation against other codes (or a newly added language token) was performed.
7. **Not implemented.** Speech synthesis (VITS), a mobile/web release for end users, and field testing with native speakers.

**Roadmap (from the proposal)**

| Phase | Planned work | Target |
| :--- | :--- | :--- |
| 2 — Data expansion (Nov–Dec 2026) | Scrape more VOV4 and Báo Đắk Lắk articles; back-translation and augmentation; re-train on 3,000–5,000 pairs | BLEU > 25 |
| 3 — Multimodal integration (Dec 2026–Jan 2027) | Add VITS text-to-speech; package as a web/mobile demo | End-to-end text + speech demo |
| 4 — Field evaluation | Trials with native speakers in Đắk Lắk; interface refinement | MOS > 4.0 |

---

## 8. Team, AI Disclosure & Citation

**Authors** — Grade 11 Mathematics, High School for the Gifted (PTNK), VNU-HCM. The project had no faculty advisor.

* **Lê Nghiêm Gia Phú (lead author):** designed and implemented the data pipeline (scraping and Gemini-based labeling), configured hyperparameters, ran fine-tuning on Colab, measured loss and BLEU, and surveyed the literature.
* **Trần Hưng Phát:** collected and manually checked Ede data, performed Unicode NFC preprocessing, built the Gradio web interface, and edited the written report.

**AI and digital-tool disclosure** (full declaration submitted with the PIISE dossier)

| Tool | Purpose | Verification by the team |
| :--- | :--- | :--- |
| Google Gemini API (`gemini-1.5-flash`) | Automatic Ede–Vietnamese pair labeling into JSON/CSV | Manual reading and cross-checking of exported pairs for meaning and font errors |
| ChatGPT-4o / Claude 3.5 | Python script support (scraping, Unicode handling, library debugging) and report-structure advice | Code executed on Colab; text rewritten by the team |
| Google Colab (T4) & Hugging Face | Compute and open-source model/libraries | Loss monitored directly; inference spot-checked |

**References**

- NLLB Team, Costa-jussà, M. R., et al. *No Language Left Behind: Scaling Human-Centered Machine Translation.* arXiv:2207.04672, 2022.
- Kim, J., Kong, J., & Son, J. *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS).* ICML 2021.
- Post, M. *A Call for Clarity in Reporting BLEU Scores.* WMT 2018 (SacreBLEU).

**Citation**

```bibtex
@techreport{piise2026_ede_vi_nmt,
  title       = {Research and Development of a Multimodal Ede--Vietnamese Bilingual Machine Translation System (Text-to-Text \& Text-to-Speech) for Communication Support and Cultural Preservation},
  author      = {Le, Nghiem Gia Phu and Tran, Hung Phat},
  institution = {High School for the Gifted, Vietnam National University Ho Chi Minh City (VNU-HCM)},
  year        = {2026},
  month       = {October},
  note        = {PIISE Research Program, Board B (Research Idea)}
}
```
