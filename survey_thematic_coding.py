import re

import pandas as pd
from docx import Document

df = pd.read_excel("Raw_Data.xlsx")

GROUPS = ["Control", "Experimental"]

# Placeholder text treated as an empty (non-substantive) response
EMPTY_VALUES = {"none.", "none", "nan", ""}
INVALID_PATTERN = re.compile(r"^\[invalid response", re.IGNORECASE)

# Theme codebook: one set of themes per open-ended question, each with a
# definition and matching keywords (regex). Keywords were derived from a
# full read-through of all 50 raw responses (25 per group).
THEMES = {
    "Understanding Change": [
        dict(id="mechanism_weighting", label="Mechanism awareness: word weighting / attention",
             definition="References AI generating or ignoring content based on word weighting or attention mechanisms.",
             patterns=[r"weigh", r"attention", r"ignore.*word", r"keyword"]),
        dict(id="probabilistic_nature", label="Probabilistic / statistical nature",
             definition="Recognizes that AI is fundamentally a probabilistic/statistical pattern predictor, "
                         "not a precise retrieval system or a system with true comprehension.",
             patterns=[r"probability", r"predict", r"statistic", r"sampl", r"pattern",
                       r"no true cognition", r"random", r"not necessarily objective",
                       r"generates responses based on"]),
        dict(id="anthropomorphism_shift", label="Anthropomorphism to tool/model shift",
             definition="Shift from treating AI as an independently-thinking person to understanding it "
                         "as a tool or probability model.",
             patterns=[r"treated? (ai )?like a (person|program)", r"think of it as a",
                       r"probability model", r"movies", r"doesn'?t think independently"]),
        dict(id="recency_training_cutoff", label="Non-real-time data / training cutoff",
             definition="Awareness that AI's knowledge has a training cutoff and is not updated in real time.",
             patterns=[r"real.?time", r"training (cut.?off|stopped|point)", r"not updated",
                       r"timeliness", r"frequently"]),
        dict(id="trust_change", label="Change in trust",
             definition="Explicit change in how much the respondent trusts AI output, or emphasis on the "
                         "need to verify it themselves.",
             patterns=[r"trust", r"verify", r"more cautious", r"shouldn'?t be trusted"]),
        dict(id="interaction_wording", label="Interaction stability: wording affects output",
             definition="Notes that question phrasing, wording, or focus affects the stability of AI's answers.",
             patterns=[r"prompt", r"wording", r"clear (point|question)", r"phrase", r"focused"]),
        dict(id="vague_mechanism", label="Vague mechanism awareness",
             definition="Reports a deepened understanding without specifying which mechanism "
                         "(weighting, probability, recency, etc.) is involved.",
             patterns=[r"deeper understanding", r"smarter than", r"gained (some )?understanding",
                       r"big.?data framework", r"logic and manner", r"operational details",
                       r"basic logic", r"logic behind", r"differs from common sense",
                       r"understands? instructions", r"operating logic",
                       r"overturned.*understanding", r"understand (better )?why ai"]),
        dict(id="no_change", label="No noticeable change",
             definition="States explicitly that understanding did not change, or that AI is still 'just a tool'.",
             patterns=[r"no (major )?change", r"felt about the same", r"nothing feels different",
                       r"just a tool"]),
        dict(id="affective_comment", label="Affective / experiential comment",
             definition="Response addresses emotional experience or attitude (interesting/boring/optimistic) "
                         "rather than cognitive content itself.",
             patterns=[r"boring", r"felt very good", r"optimistic", r"innovative and educational"]),
        dict(id="detail_reinforcement", label="Reinforced existing understanding with detail",
             definition="Existing understanding was largely correct; this experience only sharpened the details.",
             patterns=[r"rough understanding.*now.*(details?|detail)", r"more accurate picture",
                       r"confirmed my general views"]),
    ],
    "Improvement Suggestion": [
        dict(id="no_suggestion", label="No suggestion / satisfied",
             definition="States explicitly that there is no improvement suggestion, or expresses satisfaction.",
             patterns=[r"no (particular |specific )?suggestion", r"very satisfied", r"well done",
                       r"felt good", r"quite interesting", r"\bfun\b"]),
        dict(id="localization_language", label="Localization / language",
             definition="Requests a Chinese-language version or raises a translation-related need.",
             patterns=[r"chinese version", r"\btranslat"]),
        dict(id="ui_font", label="UI / font issue",
             definition="Comments on the readability or style of the interface font.",
             patterns=[r"\bfont\b"]),
        dict(id="content_flow", label="Content / process optimization",
             definition="Suggests simplifying the process, enriching materials, or reducing repetitive/lengthy content.",
             patterns=[r"simpler", r"clearer", r"real examples", r"richer", r"similar in content",
                       r"too many text.?based examples", r"hard to follow",
                       r"optimize the (page|process)"]),
        dict(id="engagement_boredom", label="Lack of engagement",
             definition="Reflects a dry experience or content that felt too short, wanting more engagement.",
             patterns=[r"boring", r"more interesting", r"too short", r"content is too little",
                       r"\bplain\b"]),
        dict(id="sensory_atmosphere", label="Sensory / atmosphere suggestion",
             definition="Suggests adding sensory elements such as sound effects or background music.",
             patterns=[r"sound effect", r"background music"]),
        dict(id="technical_feature", label="Technical / feature suggestion",
             definition="Concerns mobile support, progress indicators, results screens, or functional bugs.",
             patterns=[r"\bmobile\b", r"progress indicator", r"results?.?(summary )?screen",
                       r"\bbug\b", r"pie chart"]),
        dict(id="question_design", label="Question design ambiguity",
             definition="Points out ambiguity in questionnaire wording or items not fully matching AI's "
                         "current capabilities.",
             patterns=[r"hesitated", r"web.?search feature", r"stops updating after training"]),
        dict(id="curiosity_followup", label="Curiosity-driven follow-up question",
             definition="Not an improvement suggestion, but a curiosity question about a specific AI behavior.",
             patterns=[r"curious what would happen", r"would it defer", r"reconsider and adjust"]),
    ],
}

BLANK_LABEL = "[Blank / no response]"
INVALID_LABEL = "[Invalid data / excluded]"
UNCODED_LABEL = "Uncoded"


def classify(text: str, themes: list[dict]) -> tuple[str, list[str]]:
    """returns (status, theme_ids). status in {'blank', 'invalid', 'valid'}"""
    if pd.isna(text):
        return "blank", []
    clean = str(text).strip()
    if clean.lower() in EMPTY_VALUES:
        return "blank", []
    if INVALID_PATTERN.search(clean):
        return "invalid", []

    hits = [t["id"] for t in themes if any(re.search(p, clean, re.IGNORECASE) for p in t["patterns"])]
    if not hits:
        hits = [UNCODED_LABEL]
    return "valid", hits


for col, themes in THEMES.items():
    statuses, theme_hits = [], []
    for text in df[col]:
        status, hits = classify(text, themes)
        statuses.append(status)
        theme_hits.append(hits)
    df[f"{col} - Status"] = statuses
    df[f"{col} - Themes"] = theme_hits


def theme_label(theme_id: str, themes: list[dict]) -> str:
    for t in themes:
        if t["id"] == theme_id:
            return t["label"]
    return theme_id


def summarize(data: pd.DataFrame, col: str, themes: list[dict]) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = data[data.Group == group]
        n_total = len(sub)
        n_blank = (sub[f"{col} - Status"] == "blank").sum()
        n_invalid = (sub[f"{col} - Status"] == "invalid").sum()
        n_valid = (sub[f"{col} - Status"] == "valid").sum()

        all_hits = [h for hits in sub[f"{col} - Themes"] for h in hits]
        counts = pd.Series(all_hits).value_counts()
        for theme_id, n in counts.items():
            label = UNCODED_LABEL if theme_id == UNCODED_LABEL else theme_label(theme_id, themes)
            rows.append({"Group": group, "Theme": label, "Count": n,
                         "N (valid)": n_valid, "Percent of valid": round(100 * n / n_valid, 1) if n_valid else 0})
        rows.append({"Group": group, "Theme": BLANK_LABEL, "Count": n_blank,
                     "N (valid)": n_total, "Percent of valid": round(100 * n_blank / n_total, 1) if n_total else 0})
        if n_invalid:
            rows.append({"Group": group, "Theme": INVALID_LABEL, "Count": n_invalid,
                         "N (valid)": n_total, "Percent of valid": round(100 * n_invalid / n_total, 1) if n_total else 0})
    return pd.DataFrame(rows).sort_values(["Group", "Count"], ascending=[True, False])


def representative_quotes(data: pd.DataFrame, col: str, themes: list[dict], max_per_cell: int = 2) -> pd.DataFrame:
    rows = []
    for group in GROUPS:
        sub = data[data.Group == group]
        for t in themes:
            matches = sub[sub[f"{col} - Themes"].apply(lambda hits: t["id"] in hits)]
            for _, r in matches.head(max_per_cell).iterrows():
                rows.append({"Group": group, "Theme": t["label"],
                             "Participant ID": r["Participant ID"], "Quote": str(r[col]).strip()})
    return pd.DataFrame(rows)


summaries = {col: summarize(df, col, themes) for col, themes in THEMES.items()}
quotes = {col: representative_quotes(df, col, themes) for col, themes in THEMES.items()}

for col, summary in summaries.items():
    print(f"\n=== {col} ===")
    print(summary.to_string(index=False))

invalid_rows = df[df.apply(lambda r: any(r[f"{c} - Status"] == "invalid" for c in THEMES), axis=1)]
if not invalid_rows.empty:
    print("\nResponses excluded as invalid (not counted in valid N):")
    print(invalid_rows[["Participant ID", "Group"]].to_string(index=False))

# ---------- Excel export ----------
definitions = pd.concat([
    pd.DataFrame([{"Question": col, "Theme ID": t["id"], "Theme": t["label"],
                   "Definition": t["definition"]} for t in themes])
    for col, themes in THEMES.items()
], ignore_index=True)

detail_cols = ["Participant ID", "Group"] + list(THEMES.keys()) + \
              [f"{c} - Status" for c in THEMES] + [f"{c} - Themes" for c in THEMES]
detail = df[detail_cols].copy()
for col, themes in THEMES.items():
    detail[f"{col} - Themes"] = detail[f"{col} - Themes"].apply(
        lambda hits: ", ".join(UNCODED_LABEL if h == UNCODED_LABEL else theme_label(h, themes) for h in hits))

with pd.ExcelWriter("thematic_coding_summary.xlsx") as writer:
    definitions.to_excel(writer, sheet_name="Theme Definitions", index=False)
    for col, summary in summaries.items():
        sheet = ("Q_Understanding" if "Understanding" in col else "Q_Suggestion") + "_Frequency"
        summary.to_excel(writer, sheet_name=sheet[:31], index=False)
    for col, q in quotes.items():
        sheet = ("Q_Understanding" if "Understanding" in col else "Q_Suggestion") + "_Quotes"
        q.to_excel(writer, sheet_name=sheet[:31], index=False)

detail.to_excel("thematic_coding_detail.xlsx", index=False)

# ---------- Word report export ----------
doc = Document()
doc.add_heading("Thematic Analysis Report — Open-Ended Questions", level=0)
doc.add_paragraph("Based on coding of 25 open-ended responses per group (Control / Experimental), "
                   "themes were identified separately for the 'Understanding Change' and "
                   "'Improvement Suggestion' questions.")

for col, themes in THEMES.items():
    doc.add_heading(col, level=1)

    doc.add_heading("Theme Definitions", level=2)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text = "Theme", "Definition"
    for t in themes:
        row = table.add_row().cells
        row[0].text, row[1].text = t["label"], t["definition"]

    doc.add_heading("Coding Frequency and Group Comparison", level=2)
    summary = summaries[col]
    table = doc.add_table(rows=1, cols=5)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    for i, name in enumerate(["Group", "Theme", "Count", "N (valid responses)", "Percent (%)"]):
        hdr[i].text = name
    for _, r in summary.iterrows():
        row = table.add_row().cells
        row[0].text = str(r["Group"])
        row[1].text = str(r["Theme"])
        row[2].text = str(r["Count"])
        row[3].text = str(r["N (valid)"])
        row[4].text = str(r["Percent of valid"])

    doc.add_heading("Representative Quotes", level=2)
    q = quotes[col]
    for group in GROUPS:
        doc.add_heading(group, level=3)
        for theme in themes:
            rows = q[(q.Group == group) & (q.Theme == theme["label"])]
            if rows.empty:
                continue
            p = doc.add_paragraph()
            p.add_run(f"{theme['label']}:").bold = True
            for _, r in rows.iterrows():
                doc.add_paragraph(f"P{r['Participant ID']}: “{r['Quote']}”", style="List Bullet")

doc.add_heading("Data Notes", level=1)
p = doc.add_paragraph()
p.add_run("Excluded invalid responses:").bold = True
if invalid_rows.empty:
    doc.add_paragraph("None.")
else:
    for _, r in invalid_rows.iterrows():
        doc.add_paragraph(f"P{r['Participant ID']} ({r['Group']}): the raw text was an unrelated pasted "
                           f"block, flagged and excluded during coding; not counted toward valid N.",
                           style="List Bullet")

doc.save("thematic_analysis_report.docx")

print("\nSaved: thematic_coding_summary.xlsx, thematic_coding_detail.xlsx, thematic_analysis_report.docx")
