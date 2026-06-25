import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from sklearn.preprocessing import LabelEncoder
import shap

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
    scores = np.asarray(scores).flatten()
    names = list(names)
    
    # Ensure scores and names have the same length
    if len(scores) != len(names):
        raise ValueError(f"Length mismatch: scores has {len(scores)} elements, names has {len(names)} elements")
    
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
# 9. Feature Selection – SHAP (RF-SHAP, XGB-SHAP, LR-SHAP)
# ══════════════════════════════════════════════════════════════════════════════
# Sample 100 random instances for SHAP calculation to avoid errors
np.random.seed(42)
sample_size = min(100, len(X))
sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
X_sample = X.iloc[sample_indices].values  # Convert to numpy array

print(f"\n[SHAP] Using {sample_size} randomly selected instances")

# ── RF-SHAP ──────────────────────────────────────────────────────────────────
print("Computing RF-SHAP...")
rf_shap_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf_shap_model.fit(X, y)
explainer_rf = shap.TreeExplainer(rf_shap_model)
shap_values_rf = explainer_rf.shap_values(X_sample)

# For binary classification, shap_values is a list [class0, class1] or (n_samples, n_features, n_classes)
if isinstance(shap_values_rf, list):
    shap_values_rf = shap_values_rf[1]  # Use positive class
elif shap_values_rf.ndim == 3:
    # If shape is (n_samples, n_features, n_classes), take the positive class
    shap_values_rf = shap_values_rf[:, :, 1]

shap_values_rf = np.asarray(shap_values_rf)
print(f"  SHAP values shape: {shap_values_rf.shape}, Expected features: {len(feature_names)}")

# Compute mean absolute SHAP values across samples
rf_shap_importance = np.mean(np.abs(shap_values_rf), axis=0)
print(f"  Importance array length: {len(rf_shap_importance)}")

rf_shap_top7, _ = top_k(rf_shap_importance, feature_names, 7)
print("[RF-SHAP] Top 7:", rf_shap_top7)

# ── XGB-SHAP ─────────────────────────────────────────────────────────────────
print("Computing XGB-SHAP...")
xgb_shap_model = XGBClassifier(random_state=42, eval_metric='logloss',
                                use_label_encoder=False, n_estimators=100)
xgb_shap_model.fit(X, y)
explainer_xgb = shap.TreeExplainer(xgb_shap_model)
shap_values_xgb = explainer_xgb.shap_values(X_sample)

# Handle different output formats
if isinstance(shap_values_xgb, list):
    shap_values_xgb = shap_values_xgb[1] if len(shap_values_xgb) > 1 else shap_values_xgb[0]
elif hasattr(shap_values_xgb, 'ndim') and shap_values_xgb.ndim == 3:
    shap_values_xgb = shap_values_xgb[:, :, 1]

shap_values_xgb = np.asarray(shap_values_xgb)
print(f"  SHAP values shape: {shap_values_xgb.shape}, Expected features: {len(feature_names)}")

# Compute mean absolute SHAP values across samples
xgb_shap_importance = np.mean(np.abs(shap_values_xgb), axis=0)
print(f"  Importance array length: {len(xgb_shap_importance)}")

xgb_shap_top7, _ = top_k(xgb_shap_importance, feature_names, 7)
print("[XGB-SHAP] Top 7:", xgb_shap_top7)

# ── LR-SHAP ──────────────────────────────────────────────────────────────────
print("Computing LR-SHAP...")
lr_shap_model = LogisticRegression(random_state=42, max_iter=1000)
lr_shap_model.fit(X, y)

# For SHAP 0.48.0, use masker parameter
explainer_lr = shap.LinearExplainer(lr_shap_model, X_sample, feature_perturbation="interventional")
shap_values_lr = explainer_lr.shap_values(X_sample)

shap_values_lr = np.asarray(shap_values_lr)
if shap_values_lr.ndim == 3:
    shap_values_lr = shap_values_lr[:, :, 1]
    
print(f"  SHAP values shape: {shap_values_lr.shape}, Expected features: {len(feature_names)}")

# Compute mean absolute SHAP values across samples
lr_shap_importance = np.mean(np.abs(shap_values_lr), axis=0)
print(f"  Importance array length: {len(lr_shap_importance)}")

lr_shap_top7, _ = top_k(lr_shap_importance, feature_names, 7)
print("[LR-SHAP] Top 7:", lr_shap_top7)

# ══════════════════════════════════════════════════════════════════════════════
# 10. Cross-validation with top-7 features
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

acc_rf_shap_7 = cv_score(RandomForestClassifier(random_state=42),
                          rf_shap_top7, X, y, cv_folds)

acc_xgb_shap_7 = cv_score(RandomForestClassifier(random_state=42),
                           xgb_shap_top7, X, y, cv_folds)

acc_lr_shap_7 = cv_score(RandomForestClassifier(random_state=42),
                          lr_shap_top7, X, y, cv_folds)

print(f"\nCV Accuracies (top-7):")
print(f"  RF={acc_rf_7:.4f}  XGB={acc_xgb_7:.4f}  LR={acc_lr_7:.4f}")
print(f"  FA={acc_fa_7:.4f}  HVGS={acc_hvgs_7:.4f}  Spearman={acc_sp_7:.4f}")
print(f"  RF-SHAP={acc_rf_shap_7:.4f}  XGB-SHAP={acc_xgb_shap_7:.4f}  LR-SHAP={acc_lr_shap_7:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Remove highest RF feature; each method re-selects top-6 independently
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
rf_shap_highest = rf_shap_top7[0]  # RF-SHAP's own #1
xgb_shap_highest = xgb_shap_top7[0] # XGB-SHAP's own #1
lr_shap_highest = lr_shap_top7[0]  # LR-SHAP's own #1

print(f"\nHighest feature to remove (RF        #1): {highest_feature}")
print(f"Highest feature to remove (LR        #1): {lr_highest}")
print(f"Highest feature to remove (FA        #1): {fa_highest}")
print(f"Highest feature to remove (HVGS      #1): {hvgs_highest}")
print(f"Highest feature to remove (RF-SHAP   #1): {rf_shap_highest}")
print(f"Highest feature to remove (XGB-SHAP  #1): {xgb_shap_highest}")
print(f"Highest feature to remove (LR-SHAP   #1): {lr_shap_highest}")

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

# ── RF-SHAP-specific reduced dataset ─────────────────────────────────────────
rf_shap_drop_cols = list(dict.fromkeys([highest_feature, rf_shap_highest]))
X_red_rf_shap     = X.drop(columns=rf_shap_drop_cols)
feat_red_rf_shap  = X_red_rf_shap.columns.tolist()

# ── XGB-SHAP-specific reduced dataset ────────────────────────────────────────
xgb_shap_drop_cols = list(dict.fromkeys([highest_feature, xgb_shap_highest]))
X_red_xgb_shap     = X.drop(columns=xgb_shap_drop_cols)
feat_red_xgb_shap  = X_red_xgb_shap.columns.tolist()

# ── LR-SHAP-specific reduced dataset ─────────────────────────────────────────
lr_shap_drop_cols = list(dict.fromkeys([highest_feature, lr_shap_highest]))
X_red_lr_shap     = X.drop(columns=lr_shap_drop_cols)
feat_red_lr_shap  = X_red_lr_shap.columns.tolist()

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

# ── RF-SHAP (reduced – RF-SHAP-specific pool) ────────────────────────────────
print("Computing RF-SHAP for top-6...")
X_sample_rf_shap = X_red_rf_shap.iloc[sample_indices].values

rf2_shap_model = RandomForestClassifier(random_state=42, n_estimators=100)
rf2_shap_model.fit(X_red_rf_shap, y)
explainer_rf2 = shap.TreeExplainer(rf2_shap_model)
shap_values_rf2 = explainer_rf2.shap_values(X_sample_rf_shap)

if isinstance(shap_values_rf2, list):
    shap_values_rf2 = shap_values_rf2[1]
elif hasattr(shap_values_rf2, 'ndim') and shap_values_rf2.ndim == 3:
    shap_values_rf2 = shap_values_rf2[:, :, 1]

shap_values_rf2 = np.asarray(shap_values_rf2)
rf2_shap_importance = np.mean(np.abs(shap_values_rf2), axis=0)
rf_shap_top6, _ = top_k(rf2_shap_importance, feat_red_rf_shap, 6)

# ── XGB-SHAP (reduced – XGB-SHAP-specific pool) ──────────────────────────────
print("Computing XGB-SHAP for top-6...")
X_sample_xgb_shap = X_red_xgb_shap.iloc[sample_indices].values

xgb2_shap_model = XGBClassifier(random_state=42, eval_metric='logloss',
                                 use_label_encoder=False, n_estimators=100)
xgb2_shap_model.fit(X_red_xgb_shap, y)
explainer_xgb2 = shap.TreeExplainer(xgb2_shap_model)
shap_values_xgb2 = explainer_xgb2.shap_values(X_sample_xgb_shap)

if isinstance(shap_values_xgb2, list):
    shap_values_xgb2 = shap_values_xgb2[1] if len(shap_values_xgb2) > 1 else shap_values_xgb2[0]
elif hasattr(shap_values_xgb2, 'ndim') and shap_values_xgb2.ndim == 3:
    shap_values_xgb2 = shap_values_xgb2[:, :, 1]

shap_values_xgb2 = np.asarray(shap_values_xgb2)
xgb2_shap_importance = np.mean(np.abs(shap_values_xgb2), axis=0)
xgb_shap_top6, _ = top_k(xgb2_shap_importance, feat_red_xgb_shap, 6)

# ── LR-SHAP (reduced – LR-SHAP-specific pool) ────────────────────────────────
print("Computing LR-SHAP for top-6...")
X_sample_lr_shap = X_red_lr_shap.iloc[sample_indices].values

lr2_shap_model = LogisticRegression(random_state=42, max_iter=1000)
lr2_shap_model.fit(X_red_lr_shap, y)
explainer_lr2 = shap.LinearExplainer(lr2_shap_model, X_sample_lr_shap, feature_perturbation="interventional")
shap_values_lr2 = explainer_lr2.shap_values(X_sample_lr_shap)

shap_values_lr2 = np.asarray(shap_values_lr2)
if hasattr(shap_values_lr2, 'ndim') and shap_values_lr2.ndim == 3:
    shap_values_lr2 = shap_values_lr2[:, :, 1]
    
lr2_shap_importance = np.mean(np.abs(shap_values_lr2), axis=0)
lr_shap_top6, _ = top_k(lr2_shap_importance, feat_red_lr_shap, 6)

print("\nTop-6 features (reduced dataset):")
print(f"  RF:        {rf_top6}")
print(f"  XGB:       {xgb_top6}")
print(f"  LR:        {lr_top6}")
print(f"  FA:        {fa_top6}")
print(f"  HVGS:      {hvgs_top6}")
print(f"  Spearman:  {sp_top6}")
print(f"  RF-SHAP:   {rf_shap_top6}")
print(f"  XGB-SHAP:  {xgb_shap_top6}")
print(f"  LR-SHAP:   {lr_shap_top6}")

# ── Sanity checks ─────────────────────────────────────────────────────────────
assert lr_top7[0]   not in lr_top6,   \
    f"BUG: LR top-6 still contains LR's top-7 #1 feature '{lr_top7[0]}'"
assert fa_top7[0]   not in fa_top6,   \
    f"BUG: FA top-6 still contains FA's top-7 #1 feature '{fa_top7[0]}'"
assert hvgs_top7[0] not in hvgs_top6, \
    f"BUG: HVGS top-6 still contains HVGS's top-7 #1 feature '{hvgs_top7[0]}'"
assert rf_shap_top7[0] not in rf_shap_top6, \
    f"BUG: RF-SHAP top-6 still contains RF-SHAP's top-7 #1 feature '{rf_shap_top7[0]}'"
assert xgb_shap_top7[0] not in xgb_shap_top6, \
    f"BUG: XGB-SHAP top-6 still contains XGB-SHAP's top-7 #1 feature '{xgb_shap_top7[0]}'"
assert lr_shap_top7[0] not in lr_shap_top6, \
    f"BUG: LR-SHAP top-6 still contains LR-SHAP's top-7 #1 feature '{lr_shap_top7[0]}'"

print("\n✓ Sanity checks passed:")
print(f"  LR       top-6 does not repeat LR       top-7 #1 feature  '{lr_top7[0]}'")
print(f"  FA       top-6 does not repeat FA       top-7 #1 feature  '{fa_top7[0]}'")
print(f"  HVGS     top-6 does not repeat HVGS     top-7 #1 feature  '{hvgs_top7[0]}'")
print(f"  RF-SHAP  top-6 does not repeat RF-SHAP  top-7 #1 feature  '{rf_shap_top7[0]}'")
print(f"  XGB-SHAP top-6 does not repeat XGB-SHAP top-7 #1 feature  '{xgb_shap_top7[0]}'")
print(f"  LR-SHAP  top-6 does not repeat LR-SHAP  top-7 #1 feature  '{lr_shap_top7[0]}'")

# ══════════════════════════════════════════════════════════════════════════════
# 12. Summary table → result.csv
# ══════════════════════════════════════════════════════════════════════════════
def fmt_list(lst):
    return ", ".join(lst)

summary = pd.DataFrame({
    'Method'       : ['RF', 'XGB', 'LR', 'FA', 'HVGS', 'Spearman', 
                      'RF-SHAP', 'XGB-SHAP', 'LR-SHAP'],
    'CV_Accuracy'  : [f"{acc_rf_7:.4f}", f"{acc_xgb_7:.4f}", f"{acc_lr_7:.4f}",
                      f"{acc_fa_7:.4f}", f"{acc_hvgs_7:.4f}", f"{acc_sp_7:.4f}",
                      f"{acc_rf_shap_7:.4f}", f"{acc_xgb_shap_7:.4f}", f"{acc_lr_shap_7:.4f}"],
    'Top7_Features': [fmt_list(rf_top7),   fmt_list(xgb_top7),  fmt_list(lr_top7),
                      fmt_list(fa_top7),   fmt_list(hvgs_top7), fmt_list(sp_top7),
                      fmt_list(rf_shap_top7), fmt_list(xgb_shap_top7), fmt_list(lr_shap_top7)],
    'Top6_Features': [fmt_list(rf_top6),   fmt_list(xgb_top6),  fmt_list(lr_top6),
                      fmt_list(fa_top6),   fmt_list(hvgs_top6), fmt_list(sp_top6),
                      fmt_list(rf_shap_top6), fmt_list(xgb_shap_top6), fmt_list(lr_shap_top6)],
})

summary.to_csv("result.csv", index=False)
print("\n── Summary Table ──────────────────────────────────────────────────────")
print(summary.to_string(index=False))
print("\nSaved → result.csv")