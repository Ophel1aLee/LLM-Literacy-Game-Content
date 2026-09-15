"""
Subject Profile Chart Generator
================================
Generates four charts from "subject_profile.xlsx":
1. AI usage frequency (by group)
2. Education level (by group)
3. Frequency of AI tool usage
4. AI usage purpose distribution

Dependencies:
    pip install pandas openpyxl matplotlib

Usage:
    python subject_profile_charts.py subject_profile.xlsx

Charts are saved to a "charts/" folder in the current directory.
"""

import sys
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Colors matching the style used in the other analysis scripts
COLOR_CONTROL = "#8896A6"       # slate blue-gray = Control
COLOR_EXPERIMENTAL = "#E07A5F"  # orange-red = Experimental
GROUP_COLORS = {"Control": COLOR_CONTROL, "Experimental": COLOR_EXPERIMENTAL}
SINGLE_BAR_COLOR = "#8896A6"

# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------
FREQ_ORDER = [
    "Have deeply integrated AI into my daily work / study workflow.",
    "Habitually use them as an aid for routine tasks such as writing or looking up information.",
    "Only occasionally, when facing a specific challenge or lacking inspiration.",
    "Never use them.",
]
FREQ_LABELS = {
    FREQ_ORDER[0]: "Deeply\nintegrated",
    FREQ_ORDER[1]: "Habitual\nuse",
    FREQ_ORDER[2]: "Occasional\nuse",
    FREQ_ORDER[3]: "Never\nuse",
}

EDU_ORDER = ["Bachelor's degree", "Master's degree", "High school or equivalent", "Other:"]
EDU_LABELS = {
    "Bachelor's degree": "Bachelor's",
    "Master's degree": "Master's",
    "High school or equivalent": "High school",
    "Other:": "Other",
}

PURPOSE_CATEGORIES = [
    "Text Creation", "Information Retrieval", "Ideation and Inspiration",
    "Summarization", "Learning and Knowledge Acquisition",
    "Leisure and Conversation", "Technical Tasks", "Other",
]


def load_data(xlsx_path: str, sheet_name: str = None) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    if sheet_name is None:
        sheet_name = xl.sheet_names[0]
    df = xl.parse(sheet_name)
    print(f"[data] Loaded sheet '{sheet_name}', {len(df)} rows")
    return df


def count_tools(series: pd.Series) -> Counter:
    c = Counter()
    for val in series.dropna():
        for t in str(val).replace("、", ",").split(","):
            t = t.strip()
            if t and t != "Other:":
                c[t] += 1
    return c


def count_purposes(series: pd.Series) -> Counter:
    pattern = re.compile(r"(" + "|".join(PURPOSE_CATEGORIES) + r"):")
    c = Counter()
    for val in series.dropna():
        found = set(pattern.findall(str(val)))
        for cat in found:
            c[cat] += 1
    return c


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------
def plot_grouped_bar(labels, control_vals, exp_vals, title, ylabel, out_path):
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], control_vals, width, label="Control", color=COLOR_CONTROL)
    ax.bar([i + width / 2 for i in x], exp_vals, width, label="Experimental", color=COLOR_EXPERIMENTAL)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Group")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


def plot_single_bar(labels, values, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=SINGLE_BAR_COLOR)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {out_path}")


DEFAULT_FILENAME = "subject_profile.xlsx"


def find_input_file() -> str:
    """Use the CLI arg if given, otherwise look for the default filename,
    otherwise fall back to any single .xlsx file in the current directory."""
    if len(sys.argv) > 1:
        return sys.argv[1]

    if Path(DEFAULT_FILENAME).exists():
        return DEFAULT_FILENAME

    xlsx_files = [p for p in Path(".").glob("*.xlsx") if not p.name.startswith("~$")]
    if len(xlsx_files) == 1:
        return str(xlsx_files[0])
    if len(xlsx_files) > 1:
        print("Found multiple .xlsx files in this folder, please specify which one:")
        for p in xlsx_files:
            print(f"  {p.name}")
        sys.exit(1)

    print(f"No .xlsx file found. Put '{DEFAULT_FILENAME}' in this folder, "
          f"or run: python subject_profile_charts.py <your_file.xlsx>")
    sys.exit(1)


def main():
    xlsx_path = find_input_file()
    sheet_name = sys.argv[2] if len(sys.argv) > 2 else None

    out_dir = Path("charts")
    out_dir.mkdir(exist_ok=True)

    df = load_data(xlsx_path, sheet_name)

    # --- Chart 1: AI usage frequency (by group) ---
    ct = pd.crosstab(df["AI Usage Frequency"], df["Group"]).reindex(FREQ_ORDER).fillna(0)
    plot_grouped_bar(
        [FREQ_LABELS[f] for f in FREQ_ORDER],
        ct.get("Control", [0] * len(FREQ_ORDER)),
        ct.get("Experimental", [0] * len(FREQ_ORDER)),
        "AI Usage Frequency (by Group)",
        "Count",
        out_dir / "01_ai_frequency_by_group.png",
    )

    # --- Chart 2: Education level (by group) ---
    ct2 = pd.crosstab(df["Education Level"], df["Group"]).reindex(EDU_ORDER).fillna(0)
    plot_grouped_bar(
        [EDU_LABELS[e] for e in EDU_ORDER],
        ct2.get("Control", [0] * len(EDU_ORDER)),
        ct2.get("Experimental", [0] * len(EDU_ORDER)),
        "Education Level (by Group)",
        "Count",
        out_dir / "02_education_by_group.png",
    )

    # --- Chart 3: Frequency of AI tool usage ---
    tool_counts = count_tools(df["Common AI Tools"])
    tools_sorted = tool_counts.most_common()
    plot_single_bar(
        [t for t, _ in tools_sorted],
        [v for _, v in tools_sorted],
        "Frequency of AI Tool Usage",
        "Number of Respondents",
        out_dir / "03_tool_usage.png",
    )

    # --- Chart 4: AI usage purpose distribution ---
    purpose_counts = count_purposes(df["AI Usage Purpose"])
    purposes_sorted = sorted(
        purpose_counts.items(), key=lambda kv: kv[1], reverse=True
    )
    plot_single_bar(
        [k for k, _ in purposes_sorted],
        [v for _, v in purposes_sorted],
        f"AI Usage Purpose Distribution (multi-select, n={len(df)})",
        "Count",
        out_dir / "04_purpose_distribution.png",
    )

    print("\nAll charts generated in ./charts/")


if __name__ == "__main__":
    main()