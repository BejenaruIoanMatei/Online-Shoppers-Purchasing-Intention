import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────
# PASUL 1: Încărcare date + split
# ─────────────────────────────────────────────────────────

df = pd.read_csv('../../data/processed/online_shoppers_final_v2.csv')
df.drop(columns=['Revenue_int'], inplace=True)

X = df.drop('Revenue', axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=7, stratify=y
)

# ─────────────────────────────────────────────────────────
# PASUL 2: Clasificarea variabilelor în cantitative / calitative
# ─────────────────────────────────────────────────────────

# Cantitative — numerice continue/discrete originale
cantitative = [
    'Administrative', 'Administrative_Duration',
    'Informational', 'Informational_Duration',
    'ProductRelated', 'ProductRelated_Duration',
    'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay'
]

# Calitative — categoriale originale + discretizate + flags
calitative = [
    'Month', 'OperatingSystems', 'Browser', 'Region',
    'TrafficType', 'VisitorType', 'Weekend',
    'BounceRates_disc', 'PageValues_disc', 'ExitRates_disc',
    'ProductRelated_Duration_disc', 'Administrative_Duration_disc',
    'BounceRates_isZero', 'PageValues_isZero',
    'Administrative_Duration_isZero', 'Informational_Duration_isZero'
]

# Filtrăm doar ce există efectiv în df
cantitative = [c for c in cantitative if c in X.columns]
calitative  = [c for c in calitative  if c in X.columns]

print(f"Variabile cantitative ({len(cantitative)}): {cantitative}")
print(f"Variabile calitative  ({len(calitative)}):  {calitative}")

# ─────────────────────────────────────────────────────────
# PASUL 3: RF pe TOATE variabilele → feature importance
# ─────────────────────────────────────────────────────────

rf_full = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=20,
    random_state=7,
    n_jobs=-1
)
rf_full.fit(X_train, y_train)

importance_df = pd.DataFrame({
    'Feature':    X.columns,
    'Importance': rf_full.feature_importances_,
    'Tip':        ['Cantitativă' if c in cantitative else 'Calitativă'
                   for c in X.columns]
}).sort_values('Importance', ascending=False).reset_index(drop=True)

print("\n── TOP 10 variabile după importanță (RF complet) ──")
print(importance_df.head(10).to_string(index=False))

# ─────────────────────────────────────────────────────────
# PASUL 4: Top N din fiecare categorie
# ─────────────────────────────────────────────────────────

TOP_N = 4  # câte variabile din fiecare grup

top_cant = (importance_df[importance_df['Tip'] == 'Cantitativă']
            .head(TOP_N)['Feature'].tolist())

top_cal  = (importance_df[importance_df['Tip'] == 'Calitativă']
            .head(TOP_N)['Feature'].tolist())

print(f"\nTop {TOP_N} CANTITATIVE: {top_cant}")
print(f"Top {TOP_N} CALITATIVE:  {top_cal}")

# ─────────────────────────────────────────────────────────
# PASUL 5: Funcție evaluare model
# ─────────────────────────────────────────────────────────

def evalueaza_rf(name, features, X_train, X_test, y_train, y_test,
                 n_estimators=300, max_depth=10):
    Xtr = X_train[features]
    Xte = X_test[features]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=20,
        random_state=7,
        n_jobs=-1
    )
    rf.fit(Xtr, y_train)

    y_pred      = rf.predict(Xte)
    y_pred_prob = rf.predict_proba(Xte)[:, 1]

    cm              = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp  = cm.ravel()
    accuracy        = (tp + tn) / (tp + tn + fp + fn)
    precision       = tp / (tp + fp)  if (tp + fp) > 0 else 0
    recall          = tp / (tp + fn)  if (tp + fn) > 0 else 0
    f1              = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    auc             = roc_auc_score(y_test, y_pred_prob)
    specificity     = tn / (tn + fp)  if (tn + fp) > 0 else 0

    cv_scores = cross_val_score(rf, X_train[features], y_train,
                                cv=10, scoring='f1', n_jobs=-1)

    return {
        'Model':      name,
        'Features':   features,
        'Accuracy':   round(accuracy,   4),
        'Precision':  round(precision,  4),
        'Recall':     round(recall,     4),
        'Specificity':round(specificity,4),
        'F1':         round(f1,         4),
        'AUC':        round(auc,        4),
        'CV_F1':      round(cv_scores.mean(), 4),
        'CV_Std':     round(cv_scores.std(),  4),
        'rf_model':   rf,
        'cm':         cm,
        'y_pred_prob':y_pred_prob
    }

# ─────────────────────────────────────────────────────────
# PASUL 6: Antrenare 3 modele RF benchmark
# ─────────────────────────────────────────────────────────

print("\nAntrenare RF_Cantitativ...")
res_cant = evalueaza_rf('RF_Cantitativ', top_cant,
                         X_train, X_test, y_train, y_test)

print("Antrenare RF_Calitativ...")
res_cal  = evalueaza_rf('RF_Calitativ', top_cal,
                         X_train, X_test, y_train, y_test)

print("Antrenare RF_Complet...")
res_full = evalueaza_rf('RF_Complet', list(X.columns),
                         X_train, X_test, y_train, y_test)

benchmarks = [res_cant, res_cal, res_full]

# ─────────────────────────────────────────────────────────
# PASUL 7: Tabel comparativ complet
# (benchmark RF + cei 3 algoritmi CART/CHAID/QUEST)
# ─────────────────────────────────────────────────────────

# Rezultatele tale reale de la CART, CHAID, QUEST
arbori = pd.DataFrame([
    {'Model': 'CART (A1)',  'Accuracy': 0.8839, 'Precision': 0.7946,
     'Recall': 0.8790, 'F1': 0.8347, 'AUC': 0.9226, 'CV_F1': 0.8295, 'CV_Std': 0.0109},
    {'Model': 'CHAID (A2)', 'Accuracy': 0.7039, 'Precision': 0.5911,
     'Recall': 0.3631, 'F1': 0.4499, 'AUC': 0.7292, 'CV_F1': 0.4435, 'CV_Std': 0.0157},
    {'Model': 'QUEST (A3)', 'Accuracy': 0.7067, 'Precision': 0.5979,
     'Recall': 0.3676, 'F1': 0.4553, 'AUC': 0.7252, 'CV_F1': 0.4363, 'CV_Std': 0.0257},
])

bench_df = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ('Features', 'rf_model', 'cm', 'y_pred_prob')}
                          for r in benchmarks])

complet = pd.concat([arbori, bench_df], ignore_index=True)

print("\n── TABEL COMPARATIV FINAL ──\n")
cols_show = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'CV_F1', 'CV_Std']
print(complet[cols_show].to_string(index=False))

# ─────────────────────────────────────────────────────────
# PASUL 8: Grafice
# ─────────────────────────────────────────────────────────

modele_toate  = complet['Model'].tolist()
culori_modele = {
    'CART (A1)':      '#4C72B0',
    'CHAID (A2)':     '#55A868',
    'QUEST (A3)':     '#C44E52',
    'RF_Cantitativ':  '#DD8452',
    'RF_Calitativ':   '#8172B2',
    'RF_Complet':     '#937860',
}
culori_list = [culori_modele.get(m, '#999999') for m in modele_toate]

fig = plt.figure(figsize=(22, 18))
fig.suptitle('Benchmark RF vs Arbori de Decizie (CART / CHAID / QUEST)',
             fontsize=16, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.35)

metrici = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']

# ── 1. Bar chart metrici ──
ax1   = fig.add_subplot(gs[0, :2])
x     = np.arange(len(metrici))
width = 0.13

for i, (model, culoare) in enumerate(zip(modele_toate, culori_list)):
    vals = complet[complet['Model'] == model][metrici].values[0]
    bars = ax1.bar(x + i * width, vals, width,
                   label=model, color=culoare, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.006,
                 f'{val:.2f}', ha='center', va='bottom',
                 fontsize=6.5, fontweight='bold')

ax1.set_xticks(x + width * (len(modele_toate) - 1) / 2)
ax1.set_xticklabels(metrici, fontsize=11)
ax1.set_ylabel('Valoare')
ax1.set_title('Comparație Metrici — toți algoritmii', fontweight='bold')
ax1.legend(fontsize=8, ncol=2)
ax1.set_ylim(0, 1.15)
ax1.axhline(0.5, color='red', linestyle='--', alpha=0.3)
ax1.grid(axis='y', alpha=0.3)

# ── 2. CV F1 cu bare de eroare ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(range(len(modele_toate)),
        complet['CV_F1'], color=culori_list, alpha=0.85,
        edgecolor='white', width=0.6)
ax2.errorbar(range(len(modele_toate)),
             complet['CV_F1'], yerr=complet['CV_Std'],
             fmt='none', color='black', capsize=5, linewidth=1.5)
for i, (v, s) in enumerate(zip(complet['CV_F1'], complet['CV_Std'])):
    ax2.text(i, v + s + 0.012,
             f'{v:.3f}\n±{s:.3f}',
             ha='center', fontsize=7.5, fontweight='bold')
ax2.set_xticks(range(len(modele_toate)))
ax2.set_xticklabels(modele_toate, rotation=35, ha='right', fontsize=8)
ax2.set_title('CV F1-Score (10-fold)', fontweight='bold')
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3)

# ── 3. Feature importance RF_Cantitativ ──
ax3  = fig.add_subplot(gs[1, 0])
imp  = res_cant['rf_model'].feature_importances_
feats = top_cant
ax3.barh(feats, imp, color='#DD8452', alpha=0.85, edgecolor='white')
for i, v in enumerate(imp):
    ax3.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
ax3.set_title(f'Importanță — RF Cantitativ\n{top_cant}',
              fontweight='bold', fontsize=9)
ax3.set_xlabel('Importanță')
ax3.grid(axis='x', alpha=0.3)

# ── 4. Feature importance RF_Calitativ ──
ax4  = fig.add_subplot(gs[1, 1])
imp2 = res_cal['rf_model'].feature_importances_
ax4.barh(top_cal, imp2, color='#8172B2', alpha=0.85, edgecolor='white')
for i, v in enumerate(imp2):
    ax4.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
ax4.set_title(f'Importanță — RF Calitativ\n{top_cal}',
              fontweight='bold', fontsize=9)
ax4.set_xlabel('Importanță')
ax4.grid(axis='x', alpha=0.3)

# ── 5. Feature importance RF_Complet (top 10) ──
ax5     = fig.add_subplot(gs[1, 2])
top10   = importance_df.head(10)
culori5 = ['#DD8452' if t == 'Cantitativă' else '#8172B2'
           for t in top10['Tip']]
ax5.barh(top10['Feature'], top10['Importance'],
         color=culori5, alpha=0.85, edgecolor='white')
for i, v in enumerate(top10['Importance']):
    ax5.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)
ax5.set_title('Importanță — RF Complet (Top 10)\n🟠 Cantitativă  🟣 Calitativă',
              fontweight='bold', fontsize=9)
ax5.set_xlabel('Importanță')
ax5.grid(axis='x', alpha=0.3)

# ── 6. Radar chart ──
ax6        = fig.add_subplot(gs[2, 0], polar=True)
categorii  = metrici
N          = len(categorii)
angles     = [n / float(N) * 2 * np.pi for n in range(N)]
angles    += angles[:1]
ax6.set_theta_offset(np.pi / 2)
ax6.set_theta_direction(-1)
ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categorii, fontsize=9)
ax6.set_ylim(0, 1)

for model, culoare in zip(modele_toate, culori_list):
    vals = complet[complet['Model'] == model][categorii].values[0].tolist()
    vals += vals[:1]
    ax6.plot(angles, vals, 'o-', linewidth=1.8, color=culoare, label=model)
    ax6.fill(angles, vals, alpha=0.05, color=culoare)

ax6.set_title('Radar Chart', fontweight='bold', fontsize=11, pad=15)
ax6.legend(loc='upper right', bbox_to_anchor=(1.45, 1.2), fontsize=8)

# ── 7. Heatmap scoruri ──
ax7       = fig.add_subplot(gs[2, 1:])
heat_data = complet.set_index('Model')[['Accuracy','Precision','Recall','F1','AUC','CV_F1']]
sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.30, vmax=1.0, ax=ax7,
            linewidths=0.5, annot_kws={'size': 10, 'weight': 'bold'})
ax7.set_title('Heatmap Scoruri Comparative', fontweight='bold', fontsize=11)
ax7.set_xlabel('')

plt.savefig('../../outputs/benchmark_rf_vs_arbori.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Salvat: benchmark_rf_vs_arbori.png")

# ─────────────────────────────────────────────────────────
# PASUL 9: Concluzii automate
# ─────────────────────────────────────────────────────────

print("\n── CONCLUZII BENCHMARK ──\n")
print(f"Top {TOP_N} variabile CANTITATIVE selectate: {top_cant}")
print(f"Top {TOP_N} variabile CALITATIVE selectate:  {top_cal}\n")

for _, row in complet.iterrows():
    print(f"  {row['Model']:<18} → F1={row['F1']:.4f}  AUC={row['AUC']:.4f}  "
          f"CV_F1={row['CV_F1']:.4f} ± {row['CV_Std']:.4f}")

best = complet.loc[complet['F1'].idxmax()]
print(f"\n  Cel mai bun F1  → {best['Model']} ({best['F1']:.4f})")
best_auc = complet.loc[complet['AUC'].idxmax()]
print(f"  Cel mai bun AUC → {best_auc['Model']} ({best_auc['AUC']:.4f})")

gap_cart_rf = res_full['F1'] - 0.8347
print(f"\n  Gap F1 (RF_Complet vs CART): {gap_cart_rf:+.4f}")
print(f"  → {'RF nu aduce câștig față de CART pe aceleași variabile' if abs(gap_cart_rf) < 0.02 else 'RF îmbunătățește semnificativ față de CART'}")