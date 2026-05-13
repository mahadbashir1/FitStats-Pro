"""
FitStats Pro - Ghost Busters
Flask backend for Probability & Statistics Semester Project
Performs statistical analysis on the BodyFat Extended dataset.
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

app = Flask(__name__)

# ── Load Dataset ──────────────────────────────────────────────
df = pd.read_csv("BodyFat - Extended.csv")
NUMERIC_COLS = df.select_dtypes(include=[np.number]).columns.tolist()
# Exclude 'Original' — it's dataset-origin metadata (Y/N), not an analytical feature
EXCLUDED_COLS = ["Original"]
CATEGORICAL_COLS = [c for c in df.select_dtypes(exclude=[np.number]).columns.tolist()
                    if c not in EXCLUDED_COLS]
ALL_ANALYSIS_COLS = NUMERIC_COLS + CATEGORICAL_COLS  # Columns available for analysis

# Feature metadata for the Data Overview
FEATURE_INFO = {
    "BodyFat": {"role": "Target", "unit": "%", "description": "Body fat percentage (ground truth)"},
    "Gender": {"role": "Feature", "unit": "M/F", "description": "Biological gender of the individual"},
    "Age": {"role": "Feature", "unit": "years", "description": "Age in years"},
    "weight": {"role": "Feature", "unit": "kg", "description": "Body weight in kilograms"},
    "Height": {"role": "Feature", "unit": "m", "description": "Height in meters"},
    "Neck": {"role": "Feature", "unit": "cm", "description": "Neck circumference"},
    "Chest": {"role": "Feature", "unit": "cm", "description": "Chest circumference"},
    "Abdomen": {"role": "Feature", "unit": "cm", "description": "Abdomen circumference"},
    "Hip": {"role": "Feature", "unit": "cm", "description": "Hip circumference"},
    "Thigh": {"role": "Feature", "unit": "cm", "description": "Thigh circumference"},
    "Knee": {"role": "Feature", "unit": "cm", "description": "Knee circumference"},
    "Ankle": {"role": "Feature", "unit": "cm", "description": "Ankle circumference"},
    "Biceps": {"role": "Feature", "unit": "cm", "description": "Biceps circumference"},
    "Forearm": {"role": "Feature", "unit": "cm", "description": "Forearm circumference"},
    "Wrist": {"role": "Feature", "unit": "cm", "description": "Wrist circumference"},
}

# Domain-specific context for each variable (used to generate rich interpretations)
DATASET_CONTEXT = {
    "BodyFat": {
        "what": "body fat percentage measured via underwater weighing (Siri equation)",
        "domain": "Healthy ranges are typically 10-20% for males and 18-28% for females. Values near 0% or above 40% may indicate measurement errors or extreme cases.",
        "note": "This is the target variable — all other measurements aim to predict this.",
    },
    "Age": {
        "what": "age of the subject in years",
        "domain": "Body composition changes with age. Interestingly, in this dataset Age has near-zero correlation (r≈0.02) with BodyFat, suggesting age alone is not a good predictor.",
        "note": "Contains a likely data entry error (Age=1).",
    },
    "weight": {
        "what": "total body weight in kilograms",
        "domain": "Males average ~81 kg while females average ~60 kg. Weight alone has moderate correlation (r≈0.35) with body fat because it doesn't distinguish muscle mass from fat mass.",
        "note": "Highly correlated with most circumference measurements (r>0.85 with Abdomen, Chest, Neck).",
    },
    "Height": {
        "what": "height in meters",
        "domain": "Has a slight negative correlation with BodyFat (r≈-0.15) — taller individuals tend to have marginally lower body fat percentages.",
        "note": "Contains a likely error (Height=0.75m). Strongly left-skewed (skewness≈-2.2).",
    },
    "Neck": {
        "what": "neck circumference in cm",
        "domain": "Neck circumference is an indicator of overall frame size. It's highly correlated with weight (r≈0.89) and other upper-body measurements.",
    },
    "Chest": {
        "what": "chest circumference in cm",
        "domain": "Reflects both muscular development and fat deposits in the torso. Very strongly correlated with Abdomen (r≈0.92).",
    },
    "Abdomen": {
        "what": "abdomen (waist) circumference in cm",
        "domain": "Abdominal fat is clinically the most important fat deposit indicator. This variable has moderate correlation with BodyFat (r≈0.36) and is the strongest single circumference predictor.",
    },
    "Hip": {
        "what": "hip circumference in cm",
        "domain": "Hip circumference has the strongest correlation with BodyFat (r≈0.59) in this dataset, likely because fat deposits around the hips are a major component of total body fat, especially in females.",
    },
    "Thigh": {
        "what": "mid-thigh circumference in cm",
        "domain": "Thigh measurements reflect both fat and muscle mass in the lower body.",
    },
    "Knee": {
        "what": "knee circumference in cm",
        "domain": "Largely influenced by skeletal frame size rather than fat deposits.",
    },
    "Ankle": {
        "what": "ankle circumference in cm",
        "domain": "Primarily reflects bone structure; has weak correlation with body fat (r≈0.18).",
    },
    "Biceps": {
        "what": "biceps circumference in cm",
        "domain": "Reflects a combination of arm muscle mass and subcutaneous fat.",
    },
    "Forearm": {
        "what": "forearm circumference in cm",
        "domain": "Mostly influenced by muscle and bone; weak predictor of body fat.",
    },
    "Wrist": {
        "what": "wrist circumference in cm",
        "domain": "Primarily a skeletal frame-size indicator with minimal fat deposits.",
    },
    "Gender": {
        "what": "biological gender (M=Male, F=Female)",
        "domain": "Males (252 samples) and females (184 samples) have fundamentally different body composition. Females naturally carry higher body fat (avg 21.8% vs 19.2% for males).",
    },
}


def get_context(var):
    """Return the domain context string for a variable."""
    ctx = DATASET_CONTEXT.get(var, {})
    parts = []
    if "what" in ctx:
        parts.append(f"{var} measures {ctx['what']}.")
    if "domain" in ctx:
        parts.append(ctx["domain"])
    if "note" in ctx:
        parts.append(f"Note: {ctx['note']}")
    return " ".join(parts) if parts else f"{var} is a numeric measurement from the BodyFat dataset."


# ── Interpretation Helpers ────────────────────────────────────
def interp_correlation(r):
    abs_r = abs(r)
    d = "positive" if r > 0 else "negative"
    if abs_r == 0:
        return "No correlation (r = 0)."
    s = "strong" if abs_r >= 0.8 else ("moderate" if abs_r >= 0.5 else "weak")
    return f"There is a {s} {d} correlation (r = {r:.4f})."


def interp_skewness(skew):
    if abs(skew) < 0.5:
        return f"Approximately symmetric (skewness = {skew:.4f}). Mean ≈ Median ≈ Mode."
    if skew > 0:
        return f"Positively skewed (skewness = {skew:.4f}). Mean > Median > Mode — tail extends right."
    return f"Negatively skewed (skewness = {skew:.4f}). Mean < Median < Mode — tail extends left."


def interp_cv(cv):
    if cv < 15:
        return f"CV = {cv:.2f}% — Low variability, data is consistent."
    if cv < 30:
        return f"CV = {cv:.2f}% — Moderate variability."
    return f"CV = {cv:.2f}% — High variability, data is widely spread."


def safe(v):
    """Convert numpy scalar to Python native for JSON."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        if np.isnan(v) or np.isinf(v):
            return None
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# ── Routes ────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


# ── Data Overview ─────────────────────────────────────────────
@app.route("/api/columns")
def get_columns():
    return jsonify({"numeric": NUMERIC_COLS, "categorical": CATEGORICAL_COLS,
                    "all_analysis": ALL_ANALYSIS_COLS})


@app.route("/api/feature-info")
def feature_info():
    return jsonify({"features": FEATURE_INFO})


@app.route("/api/preview")
def get_preview():
    n = request.args.get("n", 10, type=int)
    preview = df.head(n)
    return jsonify({
        "columns": preview.columns.tolist(),
        "data": preview.values.tolist(),
        "shape": list(df.shape),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
    })


@app.route("/api/frequency", methods=["POST"])
def frequency_table():
    body = request.json
    variable = body.get("variable")
    bins = body.get("bins", 7)
    if variable not in df.columns:
        return jsonify({"error": "Variable not found"}), 400
    col = df[variable].dropna()
    table = []
    cum = 0
    if variable in CATEGORICAL_COLS:
        freq = col.value_counts().sort_index()
        total = int(freq.sum())
        for cat, f in freq.items():
            cum += int(f)
            table.append({
                "class": str(cat), "frequency": int(f),
                "relative_frequency": round(f / total, 4),
                "percentage": round(f / total * 100, 2),
                "cumulative_frequency": cum,
            })
    else:
        cut = pd.cut(col, bins=int(bins))
        freq = cut.value_counts().sort_index()
        total = int(freq.sum())
        for interval, f in freq.items():
            cum += int(f)
            table.append({
                "class": str(interval), "frequency": int(f),
                "relative_frequency": round(f / total, 4),
                "percentage": round(f / total * 100, 2),
                "cumulative_frequency": cum,
            })
    ctx = get_context(variable)
    return jsonify({"variable": variable, "table": table, "total": total,
                    "interpretation": f"Frequency distribution for {variable}: {ctx}"})


# ── Raw Data for Charts ──────────────────────────────────────
@app.route("/api/data/column", methods=["POST"])
def column_data():
    v = request.json.get("variable")
    if v not in df.columns:
        return jsonify({"error": "Variable not found"}), 400
    col = df[v].dropna()
    vals = col.tolist()
    # Compute quick stats for histogram interpretation
    skew_v = float(col.skew())
    shape = "approximately symmetric" if abs(skew_v) < 0.5 else ("right-skewed (positively skewed)" if skew_v > 0 else "left-skewed (negatively skewed)")
    interpretation = (f"The histogram of {v} shows the frequency distribution of values. "
                      f"The distribution appears {shape} (skewness = {skew_v:.3f}). "
                      f"{get_context(v)}")
    return jsonify({"variable": v, "values": vals, "interpretation": interpretation})


@app.route("/api/data/scatter", methods=["POST"])
def scatter_data():
    b = request.json
    x, y = b.get("x"), b.get("y")
    color = b.get("color")
    r_result = {"x": df[x].tolist(), "y": df[y].tolist(), "x_name": x, "y_name": y}
    if color and color in df.columns:
        r_result["color"] = df[color].tolist()
        r_result["color_name"] = color
    # Correlation for interpretation
    cx, cy = df[x].dropna(), df[y].dropna()
    idx = cx.index.intersection(cy.index)
    r_val, _ = stats.pearsonr(cx[idx], cy[idx])
    ctx_x = DATASET_CONTEXT.get(x, {}).get("what", x)
    ctx_y = DATASET_CONTEXT.get(y, {}).get("what", y)
    r_result["interpretation"] = (f"Scatter plot of {y} ({ctx_y}) vs {x} ({ctx_x}), colored by {color if color else 'none'}. "
                                   f"Pearson r = {r_val:.4f}. {interp_correlation(float(r_val))} "
                                   f"Each dot represents one individual from the dataset (n={len(idx)}).")
    return jsonify(r_result)


@app.route("/api/data/bar", methods=["POST"])
def bar_data():
    v = request.json.get("variable")
    counts = df[v].value_counts().sort_index()
    total = int(counts.sum())
    parts = "; ".join([f"{cat}: {cnt} ({cnt/total*100:.1f}%)" for cat, cnt in counts.items()])
    ctx = get_context(v)
    return jsonify({"categories": counts.index.tolist(), "counts": counts.values.tolist(),
                    "variable": v,
                    "interpretation": f"Bar chart showing the count of each {v} category. {parts}. Total: {total}. {ctx}"})


@app.route("/api/data/boxplot", methods=["POST"])
def boxplot_data():
    v = request.json.get("variable")
    col = df[v].dropna()
    q1, q3 = float(col.quantile(0.25)), float(col.quantile(0.75))
    iqr = q3 - q1
    lf, uf = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = col[(col < lf) | (col > uf)].tolist()
    ctx = get_context(v)
    outlier_note = ""
    if v == "BodyFat" and any(o <= 0 for o in outliers):
        outlier_note = " A BodyFat value of 0% is physiologically impossible and likely a measurement error."
    elif v == "Age" and any(o <= 2 for o in outliers):
        outlier_note = " Age=1 is almost certainly a data entry error — this subject may need to be excluded from analysis."
    elif v == "Height" and any(o < 1.0 for o in outliers):
        outlier_note = " Height=0.75m (75cm) is abnormal for an adult and likely a data entry error."
    return jsonify({
        "variable": v, "values": col.tolist(),
        "q1": round(q1, 4), "q3": round(q3, 4), "iqr": round(iqr, 4),
        "lower_fence": round(lf, 4), "upper_fence": round(uf, 4),
        "median": round(float(col.median()), 4), "outliers": outliers,
        "interpretation": f"IQR = {iqr:.2f}. Values below {lf:.2f} or above {uf:.2f} are considered outliers using the 1.5×IQR rule. "
                          f"Found {len(outliers)} outlier(s).{outlier_note} {ctx}",
    })


# ── Descriptive Statistics ────────────────────────────────────
@app.route("/api/descriptive", methods=["POST"])
def descriptive():
    v = request.json.get("variable")
    if v not in NUMERIC_COLS:
        return jsonify({"error": "Must be numeric"}), 400
    col = df[v].dropna()
    mean_v = float(col.mean())
    median_v = float(col.median())
    mode_r = col.mode()
    mode_v = float(mode_r.iloc[0]) if len(mode_r) > 0 else None
    std_v = float(col.std())
    var_v = float(col.var())
    q1 = float(col.quantile(0.25))
    q2 = float(col.quantile(0.50))
    q3 = float(col.quantile(0.75))
    iqr = q3 - q1
    cv = (std_v / mean_v) * 100 if mean_v != 0 else 0
    skew_v = float(col.skew())
    return jsonify({
        "variable": v,
        "stats": {
            "count": int(col.count()), "mean": round(mean_v, 4),
            "median": round(median_v, 4),
            "mode": round(mode_v, 4) if mode_v is not None else "N/A",
            "std": round(std_v, 4), "variance": round(var_v, 4),
            "min": round(float(col.min()), 4), "max": round(float(col.max()), 4),
            "range": round(float(col.max() - col.min()), 4),
            "q1": round(q1, 4), "q2": round(q2, 4), "q3": round(q3, 4),
            "iqr": round(iqr, 4), "cv": round(cv, 4), "skewness": round(skew_v, 4),
        },
        "interpretations": {
            "central_tendency": f"The average {v} is {mean_v:.2f} with a median of {median_v:.2f}. "
                + ("Mean and median are close, suggesting a symmetric distribution."
                   if abs(mean_v - median_v) / std_v < 0.3
                   else "The gap between mean and median suggests the data may be skewed."),
            "dispersion": f"Standard deviation = {std_v:.2f}. By the empirical rule, approximately 68% of values fall within [{mean_v - std_v:.2f}, {mean_v + std_v:.2f}] and 95% within [{mean_v - 2*std_v:.2f}, {mean_v + 2*std_v:.2f}].",
            "cv": interp_cv(cv),
            "skewness": interp_skewness(skew_v),
            "context": get_context(v),
        },
    })


# ── Distributions ─────────────────────────────────────────────
@app.route("/api/distribution/uniform", methods=["POST"])
def uniform_dist():
    v = request.json.get("variable")
    col = df[v].dropna()
    a, b = float(col.min()), float(col.max())
    mean = (a + b) / 2
    var_ = (b - a) ** 2 / 12
    std_ = np.sqrt(var_)
    pdf = 1 / (b - a)
    samples = np.random.uniform(a, b, len(col)).tolist()
    # Compare actual vs uniform to see if data is really uniform
    actual_mean = float(col.mean())
    fit_note = ("The actual data mean is close to the theoretical uniform mean, suggesting a relatively even spread."
                if abs(actual_mean - mean) / std_ < 0.5
                else f"The actual data mean ({actual_mean:.2f}) differs from the theoretical uniform mean ({mean:.2f}), indicating the data is NOT uniformly distributed — it clusters in certain ranges.")
    return jsonify({
        "variable": v,
        "stats": {"min": round(a, 4), "max": round(b, 4), "mean": round(mean, 4),
                  "variance": round(float(var_), 4), "std": round(float(std_), 4),
                  "pdf": round(pdf, 6)},
        "samples": samples, "n": len(col),
        "interpretation": f"Under a uniform distribution, {v} values in [{a:.2f}, {b:.2f}] would all be equally likely "
                          f"with constant PDF = {pdf:.6f}. Theoretical mean = {mean:.2f}, SD = {std_:.2f}. "
                          f"{fit_note} {get_context(v)}",
    })


@app.route("/api/distribution/normal", methods=["POST"])
def normal_dist():
    v = request.json.get("variable")
    col = df[v].dropna()
    mu = float(col.mean())
    sigma = float(col.std())
    var_ = float(col.var())
    n = len(col)
    samples = np.random.normal(mu, sigma, n).tolist()
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    y = stats.norm.pdf(x, mu, sigma)
    # Shapiro-Wilk test for normality (on a sample if n>5000)
    test_data = col.sample(min(n, 5000), random_state=42) if n > 5000 else col
    _, sw_p = stats.shapiro(test_data)
    normality_note = (f"Shapiro-Wilk test p-value = {sw_p:.4f}. "
                      + ("The data appears to follow a normal distribution (p ≥ 0.05)." if sw_p >= 0.05
                         else "The data deviates significantly from normality (p < 0.05), but the normal approximation can still be useful for large samples."))
    return jsonify({
        "variable": v,
        "stats": {"mean": round(mu, 4), "std": round(sigma, 4),
                  "variance": round(var_, 4),
                  "pdf_at_mean": round(float(stats.norm.pdf(mu, mu, sigma)), 6)},
        "samples": samples, "actual_values": col.tolist(),
        "pdf_curve": {"x": x.tolist(), "y": y.tolist()}, "n": n,
        "interpretation": f"Normal distribution for {v}: μ = {mu:.2f}, σ = {sigma:.2f}. "
                          f"By the empirical rule, ~68% of values lie in [{mu - sigma:.2f}, {mu + sigma:.2f}] "
                          f"and ~95% in [{mu - 2 * sigma:.2f}, {mu + 2 * sigma:.2f}]. "
                          f"{normality_note} {get_context(v)}",
    })


# ── Correlation ───────────────────────────────────────────────
@app.route("/api/correlation/matrix")
def corr_matrix():
    corr = df[NUMERIC_COLS].corr()
    # Find top 3 strongest correlations (excluding self)
    pairs = []
    for i in range(len(NUMERIC_COLS)):
        for j in range(i+1, len(NUMERIC_COLS)):
            pairs.append((NUMERIC_COLS[i], NUMERIC_COLS[j], corr.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top3 = "; ".join([f"{a}↔{b} (r={c:.2f})" for a, b, c in pairs[:3]])
    # Find strongest predictor of BodyFat
    bf_corrs = [(c, corr.loc["BodyFat", c]) for c in NUMERIC_COLS if c != "BodyFat"]
    bf_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    best = bf_corrs[0]
    return jsonify({
        "columns": NUMERIC_COLS, "matrix": corr.values.tolist(),
        "interpretation": f"The heatmap shows pairwise Pearson correlation coefficients between all numeric variables. "
                          f"Values close to +1 indicate strong positive correlation, -1 strong negative, 0 no linear relationship. "
                          f"Strongest overall pairs: {top3}. "
                          f"The strongest predictor of BodyFat is {best[0]} (r={best[1]:.2f}). "
                          f"Most body measurements are highly multicollinear (r>0.85) because they all broadly measure body size.",
    })


@app.route("/api/correlation/pair", methods=["POST"])
def corr_pair():
    b = request.json
    v1, v2 = b.get("var1"), b.get("var2")
    c1, c2 = df[v1].dropna(), df[v2].dropna()
    idx = c1.index.intersection(c2.index)
    r, p = stats.pearsonr(c1[idx], c2[idx])
    ctx1 = DATASET_CONTEXT.get(v1, {}).get("what", v1)
    ctx2 = DATASET_CONTEXT.get(v2, {}).get("what", v2)
    direction_note = ""
    if r > 0.3:
        direction_note = f" As {v1} increases, {v2} tends to increase as well."
    elif r < -0.3:
        direction_note = f" As {v1} increases, {v2} tends to decrease."
    else:
        direction_note = f" There is little to no linear relationship between {v1} and {v2}."
    return jsonify({
        "var1": v1, "var2": v2,
        "r": round(float(r), 4), "p_value": round(float(p), 6),
        "interpretation": interp_correlation(float(r))
            + f" (p-value = {p:.6f}). "
            + ("Statistically significant." if p < 0.05 else "Not statistically significant at α = 0.05.")
            + direction_note
            + f" Context: {v1} measures {ctx1}; {v2} measures {ctx2}.",
    })


# ── Regression ────────────────────────────────────────────────
@app.route("/api/regression", methods=["POST"])
def regression():
    b = request.json
    dep = b.get("dependent")
    indep = b.get("independent")
    try:
        temp = df[[dep, indep]].dropna().copy()
        y = temp[dep]
        X = temp[[indep]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        pred = model.predict(X).tolist()
        coefs = {}
        for name, val in model.params.items():
            coefs[name] = round(float(val), 4)
        
        # Build interpretation
        indep_coef = coefs.get(indep, 0)
        unit_indep = DATASET_CONTEXT.get(indep, {}).get("what", indep)
        unit_dep = DATASET_CONTEXT.get(dep, {}).get("what", dep)
        r2_quality = "excellent" if model.rsquared > 0.7 else ("good" if model.rsquared > 0.5 else ("moderate" if model.rsquared > 0.3 else "weak"))
        interp = (f"R² = {model.rsquared:.4f} — the model explains {model.rsquared * 100:.1f}% of the variance in {dep} ({r2_quality} fit). "
                  f"For every 1-unit increase in {indep} ({unit_indep}), {dep} ({unit_dep}) changes by {indep_coef:.4f} units. "
                  f"The model p-value = {model.f_pvalue:.6f} {'confirms it is statistically significant.' if model.f_pvalue < 0.05 else 'suggests it may not be statistically significant.'}")
        return jsonify({
            "r_squared": round(float(model.rsquared), 4),
            "adj_r_squared": round(float(model.rsquared_adj), 4),
            "f_pvalue": round(float(model.f_pvalue), 6),
            "actual": y.tolist(), "predicted": pred,
            "dep": dep, "indep": indep,
            "x_values": temp[indep].tolist(),
            "interpretation": interp,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict", methods=["POST"])
def predict():
    b = request.json
    dep, indep = b.get("dependent"), b.get("independent")
    value = float(b.get("value"))
    temp = df[[dep, indep]].dropna().copy()
    y = temp[dep]
    X = temp[[indep]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    new_X = pd.DataFrame([{"const": 1, indep: value}])
    pred = model.predict(new_X)[0]
    return jsonify({
        "prediction": round(float(pred), 4), "value": value,
        "interpretation": f"Based on the simple linear regression model, when {indep} = {value}, the predicted {dep} is {pred:.4f}. ",
    })





if __name__ == "__main__":
    app.run(debug=True, port=5000)
