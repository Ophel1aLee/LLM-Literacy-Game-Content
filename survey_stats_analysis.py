import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLOR_CONTROL = "#8896A6"       # slate blue-gray = Control
COLOR_EXPERIMENTAL = "#E07A5F"  # orange-red = Experimental
GROUP_COLORS = {"Control": COLOR_CONTROL, "Experimental": COLOR_EXPERIMENTAL}

df = pd.read_excel("Raw_Data.xlsx")
id_col = "Participant ID"
df["Gain"] = df["Post-test Score (/16)"] - df["Pre-test Score (/16)"]


OUT_DIR = "outputs"

fig, ax = plt.subplots(figsize=(6, 4.5))
groups = ["Control", "Experimental"]
x = np.arange(2)  # Pre, Post
width = 0.35

for i, group in enumerate(groups):
    sub = df[df.Group == group]
    means = [sub["Pre-test Score (/16)"].mean(), sub["Post-test Score (/16)"].mean()]
    sems = [sub["Pre-test Score (/16)"].sem(), sub["Post-test Score (/16)"].sem()]
    ax.bar(x + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
           label=group, color=GROUP_COLORS[group])

ax.set_xticks(x)
ax.set_xticklabels(["Pre-test", "Post-test"])
ax.set_ylabel("Score (out of 16)")
ax.set_title("Pre-test vs Post-test Scores")
ax.legend(title="Group")
ax.set_ylim(0, 16)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/viz_pre_post_scores.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(5.5, 4.5))
data_to_plot = [df[df.Group == g]["Gain"] for g in groups]
bp = ax.boxplot(data_to_plot, labels=groups, patch_artist=True, widths=0.5)
for patch, group in zip(bp["boxes"], groups):
    patch.set_facecolor(GROUP_COLORS[group])
    patch.set_alpha(0.7)

for i, group in enumerate(groups, start=1):
    y = df[df.Group == group]["Gain"]
    x_jitter = np.random.normal(i, 0.05, size=len(y))
    ax.scatter(x_jitter, y, color="black", alpha=0.5, s=18, zorder=3)

ax.set_ylabel("Gain = Post-test - Pre-test")
ax.set_title("Distribution of Pre-to-Post Gain Scores")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/viz_gain_boxplot.png", dpi=150)
plt.close()

tlx_cols = ["TLX - Workload (1-5)", "TLX - Effort (1-5)", "TLX - Frustration (1-5)"]
tlx_labels = ["Workload", "Effort", "Frustration"]

fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(tlx_cols))
for i, group in enumerate(groups):
    sub = df[df.Group == group]
    means = [sub[c].mean() for c in tlx_cols]
    sems = [sub[c].sem() for c in tlx_cols]
    ax.bar(x + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
           label=group, color=GROUP_COLORS[group])

ax.set_xticks(x)
ax.set_xticklabels(tlx_labels)
ax.set_ylabel("Rating (1-5)")
ax.set_title("NASA-TLX Cognitive Load Comparison")
ax.legend(title="Group")
ax.set_ylim(0, 5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/viz_tlx_comparison.png", dpi=150)
plt.close()

imi_cols = [
    "IMI - Interesting (1-5)",
    "IMI - Boring (1-5, reverse-scored item)",
    "IMI - Enjoy (1-5)",
    "IMI - Willing to Redo (1-5)",
    "IMI - Useful for Understanding AI (1-5)",
]
imi_labels = ["Interesting", "Boring\n(reverse-scored)", "Enjoy", "Willing\nto Redo", "Useful for\nUnderstanding AI"]

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(imi_cols))
for i, group in enumerate(groups):
    sub = df[df.Group == group]
    means = [sub[c].mean() for c in imi_cols]
    sems = [sub[c].sem() for c in imi_cols]
    ax.bar(x + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
           label=group, color=GROUP_COLORS[group])

ax.set_xticks(x)
ax.set_xticklabels(imi_labels)
ax.set_ylabel("Rating (1-5)")
ax.set_title("IMI Interest/Enjoyment Comparison")
ax.legend(title="Group")
ax.set_ylim(0, 5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/viz_imi_comparison.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(6, 4.5))
for group in groups:
    sub = df[df.Group == group]
    ax.scatter(sub["Pre-test Score (/16)"], sub["Gain"],
               label=group, color=GROUP_COLORS[group], alpha=0.7, s=40)

ax.set_xlabel("Pre-test Score")
ax.set_ylabel("Gain Score")
ax.set_title("Baseline Level vs Improvement")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.legend(title="Group")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/viz_pretest_vs_gain.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Trust in answer correctness: pre-test vs post-test, by group
# ---------------------------------------------------------------------------
# Confirmed against Raw_Data_with_pretest_trust.xlsx column headers:
#   "Pre-Test Trust - Answer Correct (1-5)"
#   "Trust - Answer Correct (1-5)"   <- this is the post-test measure
#   (there's also a "Tone Fluent" trust pair if you want a second chart later:
#    "Pre-Test Trust - Tone Fluent (1-5)" / "Trust - Tone Fluent (1-5)")

TRUST_PRE_COL = "Pre-Test Trust - Answer Correct (1-5)"
TRUST_POST_COL = "Trust - Answer Correct (1-5)"

missing = [c for c in (TRUST_PRE_COL, TRUST_POST_COL) if c not in df.columns]
if missing:
    print(f"[skipped] Trust pre/post chart: column(s) not found in Raw_Data.xlsx: {missing}")
    print("Available columns:")
    for c in df.columns:
        print("   -", c)
    print("Edit TRUST_PRE_COL / TRUST_POST_COL in the script to match your actual headers, then rerun.")
else:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(2)  # Pre, Post
    for i, group in enumerate(groups):
        sub = df[df.Group == group]
        means = [sub[TRUST_PRE_COL].mean(), sub[TRUST_POST_COL].mean()]
        sems = [sub[TRUST_PRE_COL].sem(), sub[TRUST_POST_COL].sem()]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
               label=group, color=GROUP_COLORS[group])

    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-test", "Post-test"])
    ax.set_ylabel("Rating (1-5)")
    ax.set_title("Trust in Answer Correctness: Pre-test vs Post-test")
    ax.legend(title="Group")
    ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/viz_trust_pre_post.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Trust in tone fluency: pre-test vs post-test, by group
# ---------------------------------------------------------------------------
TRUST_TONE_PRE_COL = "Pre-Test Trust - Tone Fluent (1-5)"
TRUST_TONE_POST_COL = "Trust - Tone Fluent (1-5)"

missing_tone = [c for c in (TRUST_TONE_PRE_COL, TRUST_TONE_POST_COL) if c not in df.columns]
if missing_tone:
    print(f"[skipped] Trust (Tone Fluent) chart: column(s) not found in Raw_Data.xlsx: {missing_tone}")
    print("Available columns:")
    for c in df.columns:
        print("   -", c)
    print("Edit TRUST_TONE_PRE_COL / TRUST_TONE_POST_COL in the script to match your actual headers, then rerun.")
else:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(2)  # Pre, Post
    for i, group in enumerate(groups):
        sub = df[df.Group == group]
        means = [sub[TRUST_TONE_PRE_COL].mean(), sub[TRUST_TONE_POST_COL].mean()]
        sems = [sub[TRUST_TONE_PRE_COL].sem(), sub[TRUST_TONE_POST_COL].sem()]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
               label=group, color=GROUP_COLORS[group])

    ax.set_xticks(x)
    ax.set_xticklabels(["Pre-test", "Post-test"])
    ax.set_ylabel("Rating (1-5)")
    ax.set_title("Trust in Tone Fluency: Pre-test vs Post-test")
    ax.legend(title="Group")
    ax.set_ylim(0, 5)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/viz_trust_tone_pre_post.png", dpi=150)
    plt.close()

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) Pre/Post
ax = axes[0, 0]
for i, group in enumerate(groups):
    sub = df[df.Group == group]
    means = [sub["Pre-test Score (/16)"].mean(), sub["Post-test Score (/16)"].mean()]
    sems = [sub["Pre-test Score (/16)"].sem(), sub["Post-test Score (/16)"].sem()]
    ax.bar(np.arange(2) + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
           label=group, color=GROUP_COLORS[group])
ax.set_xticks(np.arange(2))
ax.set_xticklabels(["Pre-test", "Post-test"])
ax.set_ylabel("Score (/16)")
ax.set_title("(a) Test Scores")
ax.legend(fontsize=8)


ax = axes[0, 1]
bp = ax.boxplot([df[df.Group == g]["Gain"] for g in groups], labels=groups,
                 patch_artist=True, widths=0.5)
for patch, group in zip(bp["boxes"], groups):
    patch.set_facecolor(GROUP_COLORS[group])
    patch.set_alpha(0.7)
ax.set_ylabel("Gain")
ax.set_title("(b) Gain Score")


ax = axes[1, 0]
xt = np.arange(len(tlx_cols))
for i, group in enumerate(groups):
    sub = df[df.Group == group]
    means = [sub[c].mean() for c in tlx_cols]
    sems = [sub[c].sem() for c in tlx_cols]
    ax.bar(xt + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
           label=group, color=GROUP_COLORS[group])
ax.set_xticks(xt)
ax.set_xticklabels(["Workload", "Effort", "Frustration"], fontsize=8)
ax.set_ylabel("Rating (1-5)")
ax.set_title("(c) Cognitive Load (TLX)")

ax = axes[1, 1]
xi = np.arange(len(imi_cols))
for i, group in enumerate(groups):
    sub = df[df.Group == group]
    means = [sub[c].mean() for c in imi_cols]
    sems = [sub[c].sem() for c in imi_cols]
    ax.bar(xi + (i - 0.5) * width, means, width, yerr=sems, capsize=4,
           label=group, color=GROUP_COLORS[group])
ax.set_xticks(xi)
ax.set_xticklabels(["Interest", "Boring", "Enjoy", "Redo", "Useful"], fontsize=7)
ax.set_ylabel("Rating (1-5)")
ax.set_title("(d) Interest/Engagement (IMI)")

fig.suptitle("Results Overview (Control vs Experimental)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/viz_overview_dashboard.png", dpi=150)
plt.close()

print("Saved the following charts to", OUT_DIR)
for name in [
    "viz_pre_post_scores.png",
    "viz_gain_boxplot.png",
    "viz_tlx_comparison.png",
    "viz_imi_comparison.png",
    "viz_pretest_vs_gain.png",
    "viz_trust_pre_post.png",
    "viz_trust_tone_pre_post.png",
    "viz_overview_dashboard.png",
]:
    print(" -", name)

# ---------------------------------------------------------------------------
# Statistical tests: Welch's t-test (all measures) + Mann-Whitney U
# (ordinal/Likert measures only), matching the method described in the
# thesis text: "Group comparisons were run using Welch's t-tests,
# cross-checked with Mann-Whitney U given the ordinal nature of the
# Likert-scale measures."
# ---------------------------------------------------------------------------
from scipy import stats

def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    var1, var2 = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled_sd

# (column, display label, is_ordinal) — is_ordinal=True adds the Mann-Whitney
# cross-check; the interval-scaled test-score measures don't need it.
MEASURES = [
    ("Pre-test Score (/16)", "Pre-test score (/16)", False),
    ("Post-test Score (/16)", "Post-test score (/16)", False),
    ("Gain", "Gain (Post - Pre)", False),
    ("TLX - Workload (1-5)", "TLX - Workload (1-5)", True),
    ("TLX - Effort (1-5)", "TLX - Effort (1-5)", True),
    ("TLX - Frustration (1-5)", "TLX - Frustration (1-5)", True),
    ("IMI - Interesting (1-5)", "IMI - Interesting (1-5)", True),
    ("IMI - Boring (1-5, reverse-scored item)", "IMI - Boring, reverse (1-5)", True),
    ("IMI - Enjoy (1-5)", "IMI - Enjoy (1-5)", True),
    ("IMI - Willing to Redo (1-5)", "IMI - Willing to Redo (1-5)", True),
    ("IMI - Useful for Understanding AI (1-5)", "IMI - Useful for Understanding AI (1-5)", True),
    (TRUST_POST_COL, "Trust - Answer Correct (1-5)", True),
    (TRUST_TONE_POST_COL if not missing_tone else None, "Trust - Tone Fluent (1-5)", True),
]

rows = []
for col, label, is_ordinal in MEASURES:
    if col is None or col not in df.columns:
        print(f"[stats skipped] {label}: column not found")
        continue
    a = df.loc[df.Group == "Control", col].dropna()
    b = df.loc[df.Group == "Experimental", col].dropna()

    t_stat, p_welch = stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(b, a)  # Experimental minus Control, so positive d = Experimental scored higher

    row = {
        "Measure": label,
        "Control M": round(a.mean(), 2),
        "Experimental M": round(b.mean(), 2),
        "Welch p": round(p_welch, 3),
        "Cohen's d": round(d, 2),
    }

    if is_ordinal:
        u_stat, p_mwu = stats.mannwhitneyu(a, b, alternative="two-sided")
        row["Mann-Whitney p"] = round(p_mwu, 3)
    else:
        row["Mann-Whitney p"] = ""

    rows.append(row)

stats_df = pd.DataFrame(rows)
print("\n=== Group comparison summary (Control vs Experimental) ===")
print(stats_df.to_string(index=False))

stats_df.to_csv(f"{OUT_DIR}/stats_summary.csv", index=False)
print(f"\nSaved statistics table to {OUT_DIR}/stats_summary.csv")

# ---------------------------------------------------------------------------
# Within-group pre-test vs post-test comparison, per condition.
# Reported in the thesis as "within-group p < .0001" but the test used was
# never pinned down in the write-up -- run BOTH the paired-samples t-test
# and the Wilcoxon signed-rank test here so we can check which one (if
# either) actually reproduces that number.
# ---------------------------------------------------------------------------
within_rows = []
for group in groups:
    sub = df[df.Group == group]
    pre = sub["Pre-test Score (/16)"]
    post = sub["Post-test Score (/16)"]
    paired = pd.concat([pre, post], axis=1).dropna()  # keep only complete pre/post pairs
    pre_c, post_c = paired.iloc[:, 0], paired.iloc[:, 1]
    diff = post_c - pre_c

    # Decide which within-group test is appropriate based on the normality
    # of the difference scores, rather than assuming either test up front.
    sw_stat, p_shapiro = stats.shapiro(diff)
    normal_ok = p_shapiro > .05

    t_stat, p_paired_t = stats.ttest_rel(pre_c, post_c)
    w_stat, p_wilcoxon = stats.wilcoxon(pre_c, post_c)

    d_z = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan  # Cohen's d for paired designs

    within_rows.append({
        "Group": group,
        "N (pairs)": len(paired),
        "Pre M": round(pre_c.mean(), 2),
        "Post M": round(post_c.mean(), 2),
        "Shapiro-Wilk p (diff)": p_shapiro,
        "Diff normal (p>.05)": normal_ok,
        "Recommended test": "Paired t-test" if normal_ok else "Wilcoxon signed-rank",
        "Paired t-test stat": round(t_stat, 3),
        "Paired t-test p": p_paired_t,
        "Wilcoxon W stat": round(w_stat, 3),
        "Wilcoxon signed-rank p": p_wilcoxon,
        "Cohen's d_z": round(d_z, 2),
    })

within_df = pd.DataFrame(within_rows)
print("\n=== Within-group pre-test vs post-test comparison ===")
print(within_df.to_string(index=False))

within_df.to_csv(f"{OUT_DIR}/within_group_stats.csv", index=False)
print(f"\nSaved within-group statistics table to {OUT_DIR}/within_group_stats.csv")
print("\nUse the 'Recommended test' column (driven by the Shapiro-Wilk result on the "
      "difference scores) to decide which test and p-value to report in Results \u00a74.2, "
      "and state that same decision rule explicitly in Method \u00a73.9.")