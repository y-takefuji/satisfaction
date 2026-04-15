import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder

# ── 1. Load & inspect ────────────────────────────────────────────────────────
df = pd.read_csv("Invistico_Airline.csv")
print("Shape (raw):", df.shape)

# ── Drop rows with NaNs ───────────────────────────────────────────────────────
df = df.dropna()
print("Shape (after dropna):", df.shape)
print("Missing values after dropna:", df.isnull().sum().sum(), "(should be 0)")

print("\nTarget distribution:\n", df['satisfaction'].value_counts())
print("\nTarget proportion:\n", df['satisfaction'].value_counts(normalize=True).round(4))

# ── 2. Encode string columns & target ────────────────────────────────────────
le = LabelEncoder()
string_cols = [df.columns[1], df.columns[2], df.columns[4], df.columns[5]]
print("\nString columns (2nd,3rd,5th,6th):", string_cols)

df_enc = df.copy()
for col in string_cols:
    df_enc[col] = le.fit_transform(df_enc[col].astype(str))

df_enc['satisfaction'] = le.fit_transform(df_enc['satisfaction'].astype(str))

X = df_enc.drop(columns=['satisfaction'])
y = df_enc['satisfaction']
feature_names = X.columns.tolist()
print("\nFeatures:", feature_names)

# ── helper: top-k by score array ─────────────────────────────────────────────
def top_k(scores, names, k):
    idx = np.argsort(scores)[::-1][:k]
    return [names[i] for i in idx], scores[idx]

# ══════════════════════════════════════════════════════════════════════════════
# 3. Feature Selection – RF (independently)
# ══════════════════════════════════════════════════════════════════════════════
rf_fs = RandomForestClassifier(random_state=42)
rf_fs.fit(X, y)
rf_imp     = rf_fs.feature_importances_
rf_top7, _ = top_k(rf_imp, feature_names, 7)
print("\n[RF] Top 7:", rf_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 4. Feature Selection – XGB (independently)
# ══════════════════════════════════════════════════════════════════════════════
xgb_fs = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
xgb_fs.fit(X, y)
xgb_imp     = xgb_fs.feature_importances_
xgb_top7, _ = top_k(xgb_imp, feature_names, 7)
print("[XGB] Top 7:", xgb_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 5. Feature Selection – Logistic Regression (independently, |coef|)
# ══════════════════════════════════════════════════════════════════════════════
lr_fs = LogisticRegression(random_state=42, max_iter=1000)
lr_fs.fit(X, y)
lr_imp     = np.abs(lr_fs.coef_[0])
lr_top7, _ = top_k(lr_imp, feature_names, 7)
print("[LR]  Top 7:", lr_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 6. Feature Selection – Feature Agglomeration (FA, independently)
#    Score each feature by its own variance across all clusters globally
#    Select top-7 across ALL clusters (not one per cluster)
# ══════════════════════════════════════════════════════════════════════════════
fa = FeatureAgglomeration(n_clusters=7)
fa.fit(X)
fa_var_scores = X.var().values
fa_top7, _    = top_k(fa_var_scores, feature_names, 7)
print("[FA]  Top 7:", fa_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Feature Selection – HVGS (independently)
#    Rank features by variance (higher variance = highly variable)
# ══════════════════════════════════════════════════════════════════════════════
hvgs_scores  = X.var().values
hvgs_top7, _ = top_k(hvgs_scores, feature_names, 7)
print("[HVGS] Top 7:", hvgs_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 8. Feature Selection – Spearman (independently)
# ══════════════════════════════════════════════════════════════════════════════
sp_scores  = np.array([abs(spearmanr(X[col], y).correlation)
                       for col in feature_names])
sp_top7, _ = top_k(sp_scores, feature_names, 7)
print("[Spearman] Top 7:", sp_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 9. Cross-validation with top-7 features
#    RF, XGB, LR → own algorithm
#    FA, HVGS, Spearman → RF for CV
# ══════════════════════════════════════════════════════════════════════════════
cv_folds = 5

def cv_score(model, feats, Xd, yd, cv=5):
    return cross_val_score(model, Xd[feats], yd,
                           cv=cv, scoring='accuracy').mean()

acc_rf_7   = cv_score(RandomForestClassifier(random_state=42),
                      rf_top7,   X, y, cv_folds)

acc_xgb_7  = cv_score(XGBClassifier(random_state=42,
                                     eval_metric='logloss',
                                     use_label_encoder=False),
                      xgb_top7,  X, y, cv_folds)

acc_lr_7   = cv_score(LogisticRegression(random_state=42, max_iter=1000),
                      lr_top7,   X, y, cv_folds)

acc_fa_7   = cv_score(RandomForestClassifier(random_state=42),
                      fa_top7,   X, y, cv_folds)

acc_hvgs_7 = cv_score(RandomForestClassifier(random_state=42),
                      hvgs_top7, X, y, cv_folds)

acc_sp_7   = cv_score(RandomForestClassifier(random_state=42),
                      sp_top7,   X, y, cv_folds)

print(f"\nCV Accuracies (top-7):")
print(f"  RF={acc_rf_7:.4f}  XGB={acc_xgb_7:.4f}  LR={acc_lr_7:.4f}")
print(f"  FA={acc_fa_7:.4f}  HVGS={acc_hvgs_7:.4f}  Spearman={acc_sp_7:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Identify highest feature per method independently
#     Remove the highest feature from the RF ranking (most reliable supervised)
#     Each method then re-selects top-6 from reduced dataset independently
# ══════════════════════════════════════════════════════════════════════════════
highest_feature = rf_top7[0]   # #1 feature from RF
print(f"\nHighest feature to remove (RF #1): {highest_feature}")

X_red    = X.drop(columns=[highest_feature])
feat_red = X_red.columns.tolist()

# ── RF (reduced) ─────────────────────────────────────────────────────────────
rf2_fs = RandomForestClassifier(random_state=42)
rf2_fs.fit(X_red, y)
rf2_imp    = rf2_fs.feature_importances_
rf_top6, _ = top_k(rf2_imp, feat_red, 6)

# ── XGB (reduced) ────────────────────────────────────────────────────────────
xgb2_fs = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
xgb2_fs.fit(X_red, y)
xgb2_imp    = xgb2_fs.feature_importances_
xgb_top6, _ = top_k(xgb2_imp, feat_red, 6)

# ── LR (reduced) ─────────────────────────────────────────────────────────────
lr2_fs = LogisticRegression(random_state=42, max_iter=1000)
lr2_fs.fit(X_red, y)
lr2_imp    = np.abs(lr2_fs.coef_[0])
lr_top6, _ = top_k(lr2_imp, feat_red, 6)

# ── FA (reduced) ─────────────────────────────────────────────────────────────
fa2 = FeatureAgglomeration(n_clusters=6)
fa2.fit(X_red)
fa2_var_scores = X_red.var().values
fa_top6, _     = top_k(fa2_var_scores, feat_red, 6)

# ── HVGS (reduced) ───────────────────────────────────────────────────────────
hvgs2_scores  = X_red.var().values
hvgs_top6, _  = top_k(hvgs2_scores, feat_red, 6)

# ── Spearman (reduced) ───────────────────────────────────────────────────────
sp2_scores  = np.array([abs(spearmanr(X_red[col], y).correlation)
                        for col in feat_red])
sp_top6, _  = top_k(sp2_scores, feat_red, 6)

print("\nTop-6 features (reduced dataset):")
print(f"  RF:       {rf_top6}")
print(f"  XGB:      {xgb_top6}")
print(f"  LR:       {lr_top6}")
print(f"  FA:       {fa_top6}")
print(f"  HVGS:     {hvgs_top6}")
print(f"  Spearman: {sp_top6}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Summary table → result.csv
# ══════════════════════════════════════════════════════════════════════════════
def fmt_list(lst):
    return ", ".join(lst)

summary = pd.DataFrame({
    'Method'       : ['RF', 'XGB', 'LR', 'FA', 'HVGS', 'Spearman'],
    'CV_Accuracy'  : [f"{acc_rf_7:.4f}", f"{acc_xgb_7:.4f}", f"{acc_lr_7:.4f}",
                      f"{acc_fa_7:.4f}", f"{acc_hvgs_7:.4f}", f"{acc_sp_7:.4f}"],
    'Top7_Features': [fmt_list(rf_top7),   fmt_list(xgb_top7),  fmt_list(lr_top7),
                      fmt_list(fa_top7),   fmt_list(hvgs_top7), fmt_list(sp_top7)],
    'Top6_Features': [fmt_list(rf_top6),   fmt_list(xgb_top6),  fmt_list(lr_top6),
                      fmt_list(fa_top6),   fmt_list(hvgs_top6), fmt_list(sp_top6)],
})

summary.to_csv("result.csv", index=False)
print("\n── Summary Table ──────────────────────────────────────────────────────")
print(summary.to_string(index=False))
print("\nSaved → result.csv")
