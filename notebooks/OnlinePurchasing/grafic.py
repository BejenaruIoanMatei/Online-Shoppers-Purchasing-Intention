import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../../data/processed/online_shoppers_final_v2.csv')
if 'Revenue_int' in df.columns:
    df.drop(columns=['Revenue_int'], inplace=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('PageValues vs Revenue — Analiza Predictorului Principal',
             fontsize=15, fontweight='bold')

colors = {0: '#4C72B0', 1: '#DD8452'}
labels = {0: 'Nu cumpără (FALSE)', 1: 'Cumpără (TRUE)'}

# ── 1. Scatter plot complet ──
ax = axes[0, 0]
for rev in [0, 1]:
    subset = df[df['Revenue'] == rev]
    ax.scatter(subset.index, subset['PageValues'],
               c=colors[rev], alpha=0.3, s=8, label=labels[rev])
ax.set_title('Scatter — PageValues per sesiune', fontweight='bold')
ax.set_xlabel('Index sesiune')
ax.set_ylabel('PageValues')
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

# ── 2. Scatter PageValues vs ExitRates colorat pe Revenue ──
ax2 = axes[0, 1]
for rev in [0, 1]:
    subset = df[df['Revenue'] == rev]
    ax2.scatter(subset['ExitRates'], subset['PageValues'],
                c=colors[rev], alpha=0.25, s=8, label=labels[rev])
ax2.axhline(28.33, color='red', linestyle='--', linewidth=1.5,
            label='Prag CART: PageValues=28.33')
ax2.set_title('PageValues vs ExitRates\n(colorat după Revenue)', fontweight='bold')
ax2.set_xlabel('ExitRates')
ax2.set_ylabel('PageValues')
ax2.legend(fontsize=8)
ax2.grid(alpha=0.2)

# ── 3. Scatter PageValues vs ProductRelated_Duration ──
ax3 = axes[0, 2]
for rev in [0, 1]:
    subset = df[df['Revenue'] == rev]
    ax3.scatter(subset['ProductRelated_Duration'], subset['PageValues'],
                c=colors[rev], alpha=0.25, s=8, label=labels[rev])
ax3.axhline(28.33, color='red', linestyle='--', linewidth=1.5,
            label='Prag CART: PageValues=28.33')
ax3.axvline(1063.31, color='green', linestyle='--', linewidth=1.5,
            label='Prag CART: ProdDur=1063s')
ax3.set_title('PageValues vs ProductRelated_Duration\n(cu praguri CART)', fontweight='bold')
ax3.set_xlabel('ProductRelated_Duration (secunde)')
ax3.set_ylabel('PageValues')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.2)

# ── 4. Boxplot PageValues per Revenue ──
ax4 = axes[1, 0]
data_false = df[df['Revenue'] == 0]['PageValues']
data_true  = df[df['Revenue'] == 1]['PageValues']
bp = ax4.boxplot([data_false, data_true],
                 labels=['Nu cumpără (0)', 'Cumpără (1)'],
                 patch_artist=True,
                 medianprops=dict(color='black', linewidth=2),
                 flierprops=dict(marker='o', markersize=3, alpha=0.3))
bp['boxes'][0].set_facecolor('#4C72B0')
bp['boxes'][1].set_facecolor('#DD8452')
ax4.axhline(28.33, color='red', linestyle='--', linewidth=1.5,
            label='Prag CART: 28.33')
ax4.set_title('Boxplot PageValues per Revenue', fontweight='bold')
ax4.set_ylabel('PageValues')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.2)

# Statistici pe grafic
med_f = data_false.median()
med_t = data_true.median()
ax4.text(1, med_f + 1, f'Mediană: {med_f:.1f}', ha='center', fontsize=9, color='#4C72B0', fontweight='bold')
ax4.text(2, med_t + 2, f'Mediană: {med_t:.1f}', ha='center', fontsize=9, color='#DD8452', fontweight='bold')

# ── 5. Distribuție PageValues (log scale) per Revenue ──
ax5 = axes[1, 1]
bins = np.linspace(0, df['PageValues'].quantile(0.99), 50)
ax5.hist(data_false[data_false > 0], bins=bins, alpha=0.6,
         color='#4C72B0', label='Nu cumpără (>0)', density=True)
ax5.hist(data_true[data_true > 0], bins=bins, alpha=0.6,
         color='#DD8452', label='Cumpără (>0)', density=True)
ax5.axvline(28.33, color='red', linestyle='--', linewidth=1.5, label='Prag: 28.33')
ax5.set_title('Distribuție PageValues (non-zero)\nper Revenue', fontweight='bold')
ax5.set_xlabel('PageValues')
ax5.set_ylabel('Densitate')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.2)

# ── 6. Rata conversie per interval PageValues ──
ax6 = axes[1, 2]
bins_rate   = [-0.001, 0, 10, 28.33, 50, 100, df['PageValues'].max() + 1]
labels_rate = ['Zero', '1-10', '11-28', '29-50', '51-100', '>100']
df['pv_bin'] = pd.cut(df['PageValues'], bins=bins_rate, labels=labels_rate)
conv_rate    = df.groupby('pv_bin', observed=True)['Revenue'].mean() * 100

conv_rate    = conv_rate.dropna()
labels_plot  = conv_rate.index.tolist()


bar_colors = ['#4C72B0' if v < 50 else '#DD8452' for v in conv_rate.values]
bars = ax6.bar(range(len(conv_rate)), conv_rate.values,
               color=bar_colors, edgecolor='white', alpha=0.85)
ax6.set_xticks(range(len(conv_rate)))
ax6.set_xticklabels(labels_rate, fontsize=10)
ax6.axhline(df['Revenue'].mean() * 100, color='gray', linestyle='--',
            linewidth=1.5, label=f'Medie generală: {df["Revenue"].mean()*100:.1f}%')
ax6.axvline(2.5, color='red', linestyle='--', linewidth=1.5,
            label='Prag CART: PageValues=28.33')
ax6.set_title('Rata de conversie per interval PageValues\n(pragul CART marcat)', fontweight='bold')
ax6.set_ylabel('% Revenue=TRUE')
ax6.set_xlabel('Interval PageValues')
ax6.legend(fontsize=8)
ax6.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, conv_rate.values):
    ax6.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.5,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('../../outputs/pagevalues_vs_revenue_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Salvat: outputs/pagevalues_vs_revenue_analysis.png")