import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, roc_auc_score, f1_score,
                              accuracy_score, classification_report,
                              roc_curve, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# ── Date ──
df = pd.read_csv('../../data/processed/online_shoppers_final_v2.csv')
if 'Revenue_int' in df.columns:
    df.drop(columns=['Revenue_int'], inplace=True)

X = df.drop('Revenue', axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=7, stratify=y
)

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Revenue train: {dict(y_train.value_counts())}")

# ─────────────────────────────────────────────────────────
# PASUL 1: XGBoost baseline — toate variabilele
# ─────────────────────────────────────────────────────────

print("\n── PASUL 1: XGBoost Baseline ──")

xgb_base = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=7,
    n_jobs=-1
)
xgb_base.fit(X_train, y_train,
             eval_set=[(X_test, y_test)],
             verbose=False)

y_pred_base = xgb_base.predict(X_test)
y_prob_base = xgb_base.predict_proba(X_test)[:, 1]

f1_base  = f1_score(y_test, y_pred_base)
auc_base = roc_auc_score(y_test, y_prob_base)
acc_base = accuracy_score(y_test, y_pred_base)
cv_base  = cross_val_score(xgb_base, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)

print(f"  Accuracy : {acc_base:.4f}")
print(f"  F1-Score : {f1_base:.4f}")
print(f"  AUC-ROC  : {auc_base:.4f}")
print(f"  CV F1    : {cv_base.mean():.4f} ± {cv_base.std():.4f}")

# ─────────────────────────────────────────────────────────
# PASUL 2: XGBoost pe Top4 (comparabil cu arborii)
# ─────────────────────────────────────────────────────────

print("\n── PASUL 2: XGBoost Top4 ──")

top4 = ['PageValues', 'ExitRates', 'Month', 'ProductRelated_Duration']

xgb_top4 = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=7,
    n_jobs=-1
)
xgb_top4.fit(X_train[top4], y_train, verbose=False)

y_pred_top4 = xgb_top4.predict(X_test[top4])
y_prob_top4 = xgb_top4.predict_proba(X_test[top4])[:, 1]

f1_top4  = f1_score(y_test, y_pred_top4)
auc_top4 = roc_auc_score(y_test, y_prob_top4)
acc_top4 = accuracy_score(y_test, y_pred_top4)
cv_top4  = cross_val_score(xgb_top4, X_train[top4], y_train, cv=5, scoring='f1', n_jobs=-1)

print(f"  Accuracy : {acc_top4:.4f}")
print(f"  F1-Score : {f1_top4:.4f}")
print(f"  AUC-ROC  : {auc_top4:.4f}")
print(f"  CV F1    : {cv_top4.mean():.4f} ± {cv_top4.std():.4f}")

# ─────────────────────────────────────────────────────────
# PASUL 3: Hyperparameter tuning cu RandomizedSearchCV
# ─────────────────────────────────────────────────────────

print("\n── PASUL 3: Hyperparameter Tuning ──")

param_dist = {
    'n_estimators':     [100, 200, 300, 500],
    'max_depth':        [3, 4, 5, 6, 7],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'gamma':            [0, 0.1, 0.2, 0.5],
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                  random_state=7, n_jobs=-1),
    param_distributions=param_dist,
    n_iter=30,           # 30 combinații random
    scoring='f1',
    cv=5,
    random_state=7,
    n_jobs=-1,
    verbose=1
)
xgb_search.fit(X_train, y_train)

print(f"\n  Parametri optimi: {xgb_search.best_params_}")
print(f"  Best CV F1:       {xgb_search.best_score_:.4f}")

# ── Model final cu parametri optimi ──
xgb_final = xgb_search.best_estimator_
y_pred_fin = xgb_final.predict(X_test)
y_prob_fin = xgb_final.predict_proba(X_test)[:, 1]

f1_fin  = f1_score(y_test, y_pred_fin)
auc_fin = roc_auc_score(y_test, y_prob_fin)
acc_fin = accuracy_score(y_test, y_pred_fin)
cm_fin  = confusion_matrix(y_test, y_pred_fin)
tn, fp, fn, tp = cm_fin.ravel()
prec_fin = tp / (tp + fp) if (tp + fp) > 0 else 0
rec_fin  = tp / (tp + fn) if (tp + fn) > 0 else 0
spec_fin = tn / (tn + fp) if (tn + fp) > 0 else 0
cv_fin   = cross_val_score(xgb_final, X_train, y_train, cv=10, scoring='f1', n_jobs=-1)

print(f"\n  ── XGBoost Final (tuned) ──")
print(f"  Accuracy    : {acc_fin:.4f}")
print(f"  Precision   : {prec_fin:.4f}")
print(f"  Recall      : {rec_fin:.4f}")
print(f"  Specificity : {spec_fin:.4f}")
print(f"  F1-Score    : {f1_fin:.4f}")
print(f"  AUC-ROC     : {auc_fin:.4f}")
print(f"  CV F1       : {cv_fin.mean():.4f} ± {cv_fin.std():.4f}")

# ─────────────────────────────────────────────────────────
# PASUL 4: Feature Importance XGBoost
# ─────────────────────────────────────────────────────────

print("\n── PASUL 4: Feature Importance ──")

imp_df = pd.DataFrame({
    'Feature':    X.columns,
    'Importance': xgb_final.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

print(imp_df.to_string(index=False))

# ─────────────────────────────────────────────────────────
# PASUL 5: Vizualizări complete
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('XGBoost — Analiză Completă vs Arbori de Decizie',
             fontsize=14, fontweight='bold')

# ── 1. Feature Importance ──
ax = axes[0, 0]
top15 = imp_df.head(10)
colors_imp = ['#DD8452' if i == 0 else '#4C72B0' for i in range(len(top15))]
ax.barh(top15['Feature'], top15['Importance'],
        color=colors_imp, edgecolor='white', alpha=0.85)
ax.set_title('Feature Importance — XGBoost Final', fontweight='bold')
ax.set_xlabel('Importance Score')
for i, v in enumerate(top15['Importance']):
    ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)

# ── 2. Matrice confuzie ──
ax2 = axes[0, 1]
disp = ConfusionMatrixDisplay(confusion_matrix=cm_fin,
                               display_labels=['Nu cumpără', 'Cumpără'])
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title(f'Matricea de Confuzie — XGBoost\nF1={f1_fin:.4f}  AUC={auc_fin:.4f}',
              fontweight='bold')

# ── 3. Curba ROC — comparație toți algoritmii ──
ax3 = axes[0, 2]

# XGBoost baseline, top4, final
for y_prob, label, col, ls in [
    (y_prob_base, f'XGB Baseline  (AUC={roc_auc_score(y_test, y_prob_base):.3f})', '#e377c2', '-'),
    (y_prob_top4, f'XGB Top4      (AUC={roc_auc_score(y_test, y_prob_top4):.3f})', '#ff7f0e', '-'),
    (y_prob_fin,  f'XGB Tuned     (AUC={roc_auc_score(y_test, y_prob_fin):.3f})',  '#d62728', '-'),
]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax3.plot(fpr, tpr, color=col, lw=2, linestyle=ls, label=label)

ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Curbe ROC — XGBoost variante', fontweight='bold')
ax3.legend(fontsize=8, loc='lower right')
ax3.grid(alpha=0.3)

# ── 4. Comparație F1 — toți algoritmii ──
ax4 = axes[1, 0]

modele_toate = {
    'CART (A1)':    0.8347,
    'CHAID (A2)':   0.4499,
    'QUEST (A3)':   0.4553,
    'RF Complet':   0.8576,
    'XGB Top4':     f1_top4,
    'XGB Baseline': f1_base,
    'XGB Tuned':    f1_fin,
}

culori_bar = ['#4C72B0','#55A868','#C44E52',
              '#937860','#ff7f0e','#e377c2','#d62728']
bars = ax4.bar(range(len(modele_toate)),
               list(modele_toate.values()),
               color=culori_bar, edgecolor='white', alpha=0.85)
ax4.set_xticks(range(len(modele_toate)))
ax4.set_xticklabels(list(modele_toate.keys()), rotation=35, ha='right', fontsize=9)
ax4.set_title('Comparație F1-Score — Toți Algoritmii', fontweight='bold')
ax4.set_ylabel('F1-Score')
ax4.set_ylim(0, 1.1)
ax4.axhline(0.5, color='red', linestyle='--', alpha=0.3)
ax4.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, modele_toate.values()):
    ax4.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.012,
             f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')

# ── 5. Comparație AUC ──
ax5 = axes[1, 1]

modele_auc = {
    'CART (A1)':    0.9226,
    'CHAID (A2)':   0.7292,
    'QUEST (A3)':   0.7252,
    'RF Complet':   0.9636,
    'XGB Top4':     auc_top4,
    'XGB Baseline': auc_base,
    'XGB Tuned':    auc_fin,
}

bars5 = ax5.bar(range(len(modele_auc)),
                list(modele_auc.values()),
                color=culori_bar, edgecolor='white', alpha=0.85)
ax5.set_xticks(range(len(modele_auc)))
ax5.set_xticklabels(list(modele_auc.keys()), rotation=35, ha='right', fontsize=9)
ax5.set_title('Comparație AUC-ROC — Toți Algoritmii', fontweight='bold')
ax5.set_ylabel('AUC-ROC')
ax5.set_ylim(0.6, 1.05)
ax5.grid(axis='y', alpha=0.3)
for bar, val in zip(bars5, modele_auc.values()):
    ax5.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', fontsize=8, fontweight='bold')

# ── 6. Tabel rezumat final ──
ax6 = axes[1, 2]
ax6.axis('off')

tabel_data = [
    ['Model',         'F1',    'AUC',   'CV F1'],
    ['CART (A1)',      '0.8347','0.9226','0.8295'],
    ['RF Complet',     '0.8576','0.9636','0.8514'],
    ['XGB Top4',       f'{f1_top4:.4f}', f'{auc_top4:.4f}', f'{cv_top4.mean():.4f}'],
    ['XGB Baseline',   f'{f1_base:.4f}', f'{auc_base:.4f}', f'{cv_base.mean():.4f}'],
    ['XGB Tuned',      f'{f1_fin:.4f}',  f'{auc_fin:.4f}',  f'{cv_fin.mean():.4f}'],
]

tbl = ax6.table(cellText=tabel_data[1:],
                colLabels=tabel_data[0],
                cellLoc='center', loc='center',
                bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2E75B6')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#EBF3FB')
    cell.set_edgecolor('#CCCCCC')

ax6.set_title('Tabel Rezumat Final', fontweight='bold')

plt.tight_layout()
plt.savefig('../../outputs/xgboost_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Salvat: outputs/xgboost_analysis.png")

# ─────────────────────────────────────────────────────────
# PASUL 6: Tabel comparativ complet în terminal
# ─────────────────────────────────────────────────────────

print("\n" + "═"*65)
print("  TABEL COMPARATIV FINAL — TOȚI ALGORITMII")
print("═"*65)

final_df = pd.DataFrame([
    {'Model': 'CART (A1)',       'F1': 0.8347, 'AUC': 0.9226, 'CV_F1': 0.8295, 'Note': '3 variabile'},
    {'Model': 'CHAID (A2)',      'F1': 0.4499, 'AUC': 0.7292, 'CV_F1': 0.4435, 'Note': '4 variabile'},
    {'Model': 'QUEST (A3)',      'F1': 0.4553, 'AUC': 0.7252, 'CV_F1': 0.4363, 'Note': '3 variabile'},
    {'Model': 'RF Complet',      'F1': 0.8576, 'AUC': 0.9636, 'CV_F1': 0.8514, 'Note': 'toate var.'},
    {'Model': 'XGB Top4',        'F1': round(f1_top4, 4),  'AUC': round(auc_top4, 4),  'CV_F1': round(cv_top4.mean(), 4),  'Note': '4 variabile'},
    {'Model': 'XGB Baseline',    'F1': round(f1_base, 4),  'AUC': round(auc_base, 4),  'CV_F1': round(cv_base.mean(), 4),  'Note': 'toate var.'},
    {'Model': 'XGB Tuned',       'F1': round(f1_fin, 4),   'AUC': round(auc_fin, 4),   'CV_F1': round(cv_fin.mean(), 4),   'Note': 'toate var. + tuning'},
]).sort_values('F1', ascending=False)

print(final_df.to_string(index=False))

best = final_df.iloc[0]
print(f"\n  🏆 Cel mai bun model: {best['Model']} (F1={best['F1']:.4f}, AUC={best['AUC']:.4f})")