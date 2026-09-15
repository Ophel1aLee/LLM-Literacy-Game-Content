# Thesis Config

Research code for my master thesis "A Serious Game For Enhancing LLM Literacy Among Non-Technical Users."
It has two parts:

1. **Level 1–4**: four in-game content for each level. Each produces a JSON config consumed by a companion Unity project that turns the results into an interactive demo.
2. **Survey analysis**: quantitative and qualitative analysis of the user study (pre/post test scores, questionnaire responses, participant profile).

## Repository structure

```
level1.py                      Level 1: factual-misconception probe (Mistral-7B)
level2_attention.py             shared library: attention-intervention + JSON config builder (GPT-2-large)
level2_medication.py            Level 2 scenario: medication negation ("not safe")
level2_bridge.py                Level 2 scenario: bridge structural cracks
level2_crane.py                 Level 2 scenario: crane operator not paying attention
level3.py                       Level 3: knowledge-cutoff comparison across model generations
level4.py                       Level 4: headline framing across audience/tone/engagement tags

survey_stats_analysis.py        quantitative analysis of the user study (charts + significance tests)
survey_thematic_coding.py       thematic coding of open-ended survey responses
subject_profile_charts.py       participant demographic charts

models/                         local HF model caches (gitignored, not tracked)
outputs/                        charts from survey_stats_analysis.py
charts/                         charts from subject_profile_charts.py
```

Each `levelN_*.py` script writes a `levelN_*.json` file with the same prefix (e.g. `level2_medication.py` → `level2_config_medication.json`); those JSON files are what the Unity project reads.

## Setup

Python 3.10+, with:

```
pip install torch transformers bitsandbytes pandas numpy scipy matplotlib openpyxl python-docx
```

The Level 1–4 scripts also need local copies of the relevant Hugging Face models (GPT-2-large, Mistral-7B-Instruct-v0.2, Pythia-1B, OPT-1.3B, etc.) under `models/`, and a CUDA GPU for anything beyond GPT-2-large.
