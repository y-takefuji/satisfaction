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
print("\nTarget proportion:\n",
      df['satisfaction'].value_counts(normalize=True).round(4))

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
xgb_fs = XGBClassifier(random_state=42, eval_metric='logloss',
                        use_label_encoder=False)
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
# ══════════════════════════════════════════════════════════════════════════════
fa = FeatureAgglomeration(n_clusters=7)
fa.fit(X)
fa_var_scores = X.var().values
fa_top7, _    = top_k(fa_var_scores, feature_names, 7)
print("[FA]  Top 7:", fa_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Feature Selection – HVGS (independently)
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
# ══════════════════════════════════════════════════════════════════════════════
cv_folds = 5

def cv_score(model, feats, Xd, yd, cv=5):
    return cross_val_score(model, Xd[feats], yd,
                           cv=cv, scoring='accuracy').mean()

acc_rf_7   = cv_score(RandomForestClassifier(random_state=42),
                      rf_top7, X, y, cv_folds)

acc_xgb_7  = cv_score(XGBClassifier(random_state=42,
                                     eval_metric='logloss',
                                     use_label_encoder=False),
                      xgb_top7, X, y, cv_folds)

acc_lr_7   = cv_score(LogisticRegression(random_state=42, max_iter=1000),
                      lr_top7, X, y, cv_folds)

acc_fa_7   = cv_score(RandomForestClassifier(random_state=42),
                      fa_top7, X, y, cv_folds)

acc_hvgs_7 = cv_score(RandomForestClassifier(random_state=42),
                      hvgs_top7, X, y, cv_folds)

acc_sp_7   = cv_score(RandomForestClassifier(random_state=42),
                      sp_top7, X, y, cv_folds)

print(f"\nCV Accuracies (top-7):")
print(f"  RF={acc_rf_7:.4f}  XGB={acc_xgb_7:.4f}  LR={acc_lr_7:.4f}")
print(f"  FA={acc_fa_7:.4f}  HVGS={acc_hvgs_7:.4f}  Spearman={acc_sp_7:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Remove highest RF feature; each method re-selects top-6 independently
#
#  FIX (LR):   LR also removes its own top-7 #1 feature before re-selecting.
#  FIX (FA):   FA also removes its own top-7 #1 feature before re-selecting.
#  FIX (HVGS): HVGS also removes its own top-7 #1 feature before re-selecting.
#
#  For each method, build a method-specific reduced feature pool that removes:
#    (a) RF's #1 feature  (global removal)
#    (b) The method's own #1 feature (if different from RF's #1)
#  This guarantees top-6 never repeats the method's own top-7 #1 feature.
# ══════════════════════════════════════════════════════════════════════════════
highest_feature = rf_top7[0]       # RF's #1 — globally removed
lr_highest      = lr_top7[0]       # LR's own #1
fa_highest      = fa_top7[0]       # FA's own #1
hvgs_highest    = hvgs_top7[0]     # HVGS's own #1

print(f"\nHighest feature to remove (RF   #1): {highest_feature}")
print(f"Highest feature to remove (LR   #1): {lr_highest}")
print(f"Highest feature to remove (FA   #1): {fa_highest}")
print(f"Highest feature to remove (HVGS #1): {hvgs_highest}")

# ── Shared reduced dataset (RF #1 removed) — used by RF, XGB, Spearman ──────
X_red    = X.drop(columns=[highest_feature])
feat_red = X_red.columns.tolist()

# ── LR-specific reduced dataset: remove RF #1 + LR #1 (deduplicated) ────────
lr_drop_cols  = list(dict.fromkeys([highest_feature, lr_highest]))
X_red_lr      = X.drop(columns=lr_drop_cols)
feat_red_lr   = X_red_lr.columns.tolist()

# ── FA-specific reduced dataset: remove RF #1 + FA #1 (deduplicated) ────────
fa_drop_cols  = list(dict.fromkeys([highest_feature, fa_highest]))
X_red_fa      = X.drop(columns=fa_drop_cols)
feat_red_fa   = X_red_fa.columns.tolist()

# ── HVGS-specific reduced dataset: remove RF #1 + HVGS #1 (deduplicated) ────
hvgs_drop_cols = list(dict.fromkeys([highest_feature, hvgs_highest]))
X_red_hvgs     = X.drop(columns=hvgs_drop_cols)
feat_red_hvgs  = X_red_hvgs.columns.tolist()

# ── RF (reduced) ─────────────────────────────────────────────────────────────
rf2_fs = RandomForestClassifier(random_state=42)
rf2_fs.fit(X_red, y)
rf2_imp    = rf2_fs.feature_importances_
rf_top6, _ = top_k(rf2_imp, feat_red, 6)

# ── XGB (reduced) ────────────────────────────────────────────────────────────
xgb2_fs = XGBClassifier(random_state=42, eval_metric='logloss',
                         use_label_encoder=False)
xgb2_fs.fit(X_red, y)
xgb2_imp    = xgb2_fs.feature_importances_
xgb_top6, _ = top_k(xgb2_imp, feat_red, 6)

# ── LR (reduced – LR-specific pool) ──────────────────────────────────────────
lr2_fs = LogisticRegression(random_state=42, max_iter=1000)
lr2_fs.fit(X_red_lr, y)
lr2_imp    = np.abs(lr2_fs.coef_[0])
lr_top6, _ = top_k(lr2_imp, feat_red_lr, 6)

# ── FA (reduced – FA-specific pool) ──────────────────────────────────────────
#    Recompute variance scores on FA-specific reduced feature set,
#    so FA's own #1 feature is excluded from the candidate pool entirely.
fa2 = FeatureAgglomeration(n_clusters=6)
fa2.fit(X_red_fa)
fa2_var_scores = X_red_fa.var().values          # variance on FA-specific pool
fa_top6, _     = top_k(fa2_var_scores, feat_red_fa, 6)

# ── HVGS (reduced – HVGS-specific pool) ──────────────────────────────────────
#    Recompute variance scores on HVGS-specific reduced feature set,
#    so HVGS's own #1 feature is excluded from the candidate pool entirely.
hvgs2_scores  = X_red_hvgs.var().values         # variance on HVGS-specific pool
hvgs_top6, _  = top_k(hvgs2_scores, feat_red_hvgs, 6)

# ── Spearman (reduced – shared pool) ─────────────────────────────────────────
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

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert lr_top7[0]   not in lr_top6,   \
    f"BUG: LR top-6 still contains LR's top-7 #1 feature '{lr_top7[0]}'"
assert fa_top7[0]   not in fa_top6,   \
    f"BUG: FA top-6 still contains FA's top-7 #1 feature '{fa_top7[0]}'"
assert hvgs_top7[0] not in hvgs_top6, \
    f"BUG: HVGS top-6 still contains HVGS's top-7 #1 feature '{hvgs_top7[0]}'"
print("\n✓ Sanity checks passed:")
print(f"  LR   top-6 does not repeat LR   top-7 #1 feature  '{lr_top7[0]}'")
print(f"  FA   top-6 does not repeat FA   top-7 #1 feature  '{fa_top7[0]}'")
print(f"  HVGS top-6 does not repeat HVGS top-7 #1 feature  '{hvgs_top7[0]}'")

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
