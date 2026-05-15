import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PASUL 11: COMPARAȚIA FINALĂ A MODELELOR
# ─────────────────────────────────────────

print("=" * 60)
print("11. COMPARAȚIA FINALĂ – CART vs CHAID vs QUEST")
print("=" * 60)

# ── Date rezultate ──
results = pd.DataFrame([
    {
        'Model':      'CART',
        'Analiza':    'A1',
        'Variabile':  'PageValues, BounceRates, VisitorType',
        'Adâncime':   5,
        'Frunze':     14,
        'Accuracy':   0.8839,
        'Precision':  0.7946,
        'Recall':     0.8790,
        'F1':         0.8347,
        'AUC':        0.9226,
        'CV_F1':      0.8295,
        'CV_Std':     0.0109
    },
    {
        'Model':      'CHAID',
        'Analiza':    'A2',
        'Variabile':  'ExitRates_disc, Month, SpecialDay, Weekend',
        'Adâncime':   10,
        'Frunze':     49,
        'Accuracy':   0.7039,
        'Precision':  0.5911,
        'Recall':     0.3631,
        'F1':         0.4499,
        'AUC':        0.7292,
        'CV_F1':      0.4435,
        'CV_Std':     0.0157
    },
    {
        'Model':      'QUEST',
        'Analiza':    'A3',
        'Variabile':  'ProductRelated_Duration_disc, Region, TrafficType',
        'Adâncime':   17,
        'Frunze':     337,
        'Accuracy':   0.7067,
        'Precision':  0.5979,
        'Recall':     0.3676,
        'F1':         0.4553,
        'AUC':        0.7252,
        'CV_F1':      0.4363,
        'CV_Std':     0.0257
    }
])

# ── Tabel comparativ ──
print("\n── TABEL COMPARATIV ──\n")
display_cols = ['Model', 'Variabile', 'Adâncime', 'Frunze',
                'Accuracy', 'Precision', 'Recall', 'F1', 'AUC', 'CV_F1', 'CV_Std']
print(results[display_cols].to_string(index=False))

# ─────────────────────────────────────────
# 11.1 GRAFIC COMPARATIV – METRICI
# ─────────────────────────────────────────

metrics     = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
models      = results['Model'].tolist()
colors      = ['#4C72B0', '#2ca02c', '#ff7f0e']

fig = plt.figure(figsize=(20, 16))
fig.suptitle('Comparație Finală – CART vs CHAID vs QUEST',
             fontsize=16, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── 1. Bar chart metrici principale ──
ax1 = fig.add_subplot(gs[0, :2])
x      = np.arange(len(metrics))
width  = 0.25

for i, (model, color) in enumerate(zip(models, colors)):
    vals = results[results['Model'] == model][metrics].values[0]
    bars = ax1.bar(x + i * width, vals, width, label=model,
                   color=color, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax1.set_xticks(x + width)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.set_ylabel('Valoare metrică')
ax1.set_title('Comparație Metrici de Performanță', fontweight='bold', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_ylim(0, 1.12)
ax1.axhline(0.5, color='red', linestyle='--', alpha=0.3, linewidth=1)
ax1.grid(axis='y', alpha=0.3)

# ── 2. CV F1 cu bare de eroare ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(models, results['CV_F1'], color=colors, alpha=0.85,
        edgecolor='white', width=0.5)
ax2.errorbar(models, results['CV_F1'], yerr=results['CV_Std'],
             fmt='none', color='black', capsize=6, linewidth=2)
for i, (v, s) in enumerate(zip(results['CV_F1'], results['CV_Std'])):
    ax2.text(i, v + s + 0.015, f'{v:.4f}\n±{s:.4f}',
             ha='center', fontsize=9, fontweight='bold')
ax2.set_title('CV F1-Score (10-fold)\ncu interval de variație', fontweight='bold', fontsize=11)
ax2.set_ylabel('F1-Score')
ax2.set_ylim(0, 1.05)
ax2.grid(axis='y', alpha=0.3)

# ── 3. Radar chart ──
ax3 = fig.add_subplot(gs[1, 0], polar=True)
categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

ax3.set_theta_offset(np.pi / 2)
ax3.set_theta_direction(-1)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(categories, fontsize=9)
ax3.set_ylim(0, 1)
ax3.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=7)

for model, color in zip(models, colors):
    vals = results[results['Model'] == model][categories].values[0].tolist()
    vals += vals[:1]
    ax3.plot(angles, vals, 'o-', linewidth=2, color=color, label=model)
    ax3.fill(angles, vals, alpha=0.1, color=color)

ax3.set_title('Radar Chart\nProfil Performanță', fontweight='bold', fontsize=11, pad=15)
ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)

# ── 4. Complexitate arbori ──
ax4 = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(models))
bars_d = ax4.bar(x_pos - 0.2, results['Adâncime'], 0.35,
                  label='Adâncime', color=colors, alpha=0.7, edgecolor='white')
ax4_twin = ax4.twinx()
bars_f = ax4_twin.bar(x_pos + 0.2, results['Frunze'], 0.35,
                       label='Frunze', color=colors, alpha=0.4,
                       edgecolor='white', hatch='//')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(models)
ax4.set_ylabel('Adâncime arbore', fontsize=9)
ax4_twin.set_ylabel('Număr frunze', fontsize=9)
ax4.set_title('Complexitatea Arborilor', fontweight='bold', fontsize=11)

for bar, val in zip(bars_d, results['Adâncime']):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             str(val), ha='center', fontsize=10, fontweight='bold')
for bar, val in zip(bars_f, results['Frunze']):
    ax4_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                  str(val), ha='center', fontsize=10, fontweight='bold')

lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

# ── 5. Heatmap scoruri ──
ax5 = fig.add_subplot(gs[1, 2])
heat_data = results.set_index('Model')[['Accuracy','Precision','Recall','F1','AUC','CV_F1']]
sns.heatmap(heat_data, annot=True, fmt='.3f', cmap='RdYlGn',
            vmin=0.3, vmax=1.0, ax=ax5, linewidths=0.5,
            annot_kws={'size': 10, 'weight': 'bold'})
ax5.set_title('Heatmap Scoruri', fontweight='bold', fontsize=11)
ax5.set_xlabel('')

# ── 6. Concluzii text ──
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')

best_f1  = results.loc[results['F1'].idxmax(), 'Model']
best_auc = results.loc[results['AUC'].idxmax(), 'Model']
best_rec = results.loc[results['Recall'].idxmax(), 'Model']
stabil   = results.loc[results['CV_Std'].idxmin(), 'Model']
simplu   = results.loc[results['Adâncime'].idxmin(), 'Model']

concluzie = f"""
CONCLUZII COMPARATIVE:

-   Cel mai bun F1-Score -> {best_f1} (F1 = {results.loc[results['F1'].idxmax(), 'F1']:.4f})   |   identifică cel mai bine cumpărătorii reali
-   Cel mai bun AUC-ROC -> {best_auc} (AUC = {results.loc[results['AUC'].idxmax(), 'AUC']:.4f})  |   putere discriminantă maximă
-   Cel mai bun Recall  -> {best_rec} (Recall = {results.loc[results['Recall'].idxmax(), 'Recall']:.4f})  |   ratează cei mai puțini cumpărători reali
-   Cel mai stabil (CV) -> {stabil} (Std = {results.loc[results['CV_Std'].idxmin(), 'CV_Std']:.4f})   |   variație minimă între fold-uri
-   Cel mai interpretabil -> {simplu} (Adâncime = {results.loc[results['Adâncime'].idxmin(), 'Adâncime']})          |   arbore cel mai simplu și lizibil

CA NOTĂ: Diferențele de performanță reflectă în primul rând PUTEREA PREDICTIVĂ A VARIABILELOR,
    nu neapărat superioritatea unui algoritm față de altul. PageValues (A1/CART) este un predictor
    excepțional, în timp ce variabilele temporale (A2/CHAID) și geografice (A3/QUEST) sunt mai slabe.
"""

ax6.text(0.01, 0.95, concluzie, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig('../../outputs/model_comparison_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("Salvat: outputs/model_comparison_final.png")

# ─────────────────────────────────────────
# 11.2 TABEL FINAL PENTRU RAPORT
# ─────────────────────────────────────────

print("\n── TABEL FINAL PENTRU RAPORT ──\n")
print(f"{'Model':<8} {'Variabile':<50} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6} {'CV_F1':>7} {'Adânc':>6} {'Frunze':>7}")
print("-" * 115)
for _, row in results.iterrows():
    print(f"{row['Model']:<8} {row['Variabile']:<50} "
          f"{row['Accuracy']:>6.4f} {row['Precision']:>6.4f} "
          f"{row['Recall']:>6.4f} {row['F1']:>6.4f} "
          f"{row['AUC']:>6.4f} {row['CV_F1']:>7.4f} "
          f"{row['Adâncime']:>6} {row['Frunze']:>7}")

# Salvare tabel
results.to_csv('../../outputs/comparatie_finala.csv', index=False)
print("\nTabel salvat: outputs/comparatie_finala.csv")
print("\nComparație finală completă!")