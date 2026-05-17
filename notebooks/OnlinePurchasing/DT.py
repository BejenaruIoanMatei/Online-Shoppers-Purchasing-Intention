import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, roc_auc_score,
                              f1_score, accuracy_score)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('../../data/processed/online_shoppers_final_v2.csv')
if 'Revenue_int' in df.columns:
    df.drop(columns=['Revenue_int'], inplace=True)

X = df.drop('Revenue', axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=7, stratify=y
)

top4_final = ['PageValues', 'ExitRates', 'Month', 'ProductRelated_Duration']

Xtr = X_train[top4_final]
Xte = X_test[top4_final]

# ─────────────────────────────────────────────────────────
# FUNCȚIE COMUNĂ: calculează alpha optim + reantrenează
# ─────────────────────────────────────────────────────────

def find_best_alpha(criterion, Xtr, Xte, y_train, y_test, label):
    """
    Parcurge tot drumul de pruning CCP și returnează
    alpha-ul care maximizează F1 pe setul de test.
    Folosim F1 (nu accuracy) pentru că datele sunt dezechilibrate.
    """
    # Arbore complet fără restricții → generează tot drumul de alpha
    clf_full = DecisionTreeClassifier(
        criterion=criterion,
        random_state=7
    )
    path       = clf_full.cost_complexity_pruning_path(Xtr, y_train)
    ccp_alphas = path.ccp_alphas[:-1]  # eliminăm ultimul = arbore gol

    train_f1, test_f1 = [], []
    train_acc, test_acc = [], []

    for alpha in ccp_alphas:
        clf = DecisionTreeClassifier(
            criterion=criterion,
            ccp_alpha=alpha,
            random_state=7
        )
        clf.fit(Xtr, y_train)
        train_f1.append(f1_score(y_train, clf.predict(Xtr), zero_division=0))
        test_f1.append( f1_score(y_test,  clf.predict(Xte), zero_division=0))
        train_acc.append(clf.score(Xtr, y_train))
        test_acc.append( clf.score(Xte, y_test))

    best_idx   = np.argmax(test_f1)
    best_alpha = ccp_alphas[best_idx]

    print(f"\n{'─'*55}")
    print(f"  {label}")
    print(f"{'─'*55}")
    print(f"  Număr alpha-uri explorate : {len(ccp_alphas)}")
    print(f"  Alpha optim (max F1 test) : {best_alpha:.6f}")
    print(f"  F1 test la alpha optim    : {test_f1[best_idx]:.4f}")
    print(f"  Accuracy test             : {test_acc[best_idx]:.4f}")

    return best_alpha, ccp_alphas, train_f1, test_f1, train_acc, test_acc

# ─────────────────────────────────────────────────────────
# PASUL 1: Găsire alpha optim pentru CART (Gini) și CHAID (Entropy)
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  CALCULARE ccp_alpha OPTIM")
print("=" * 55)

alpha_cart,  alphas_cart,  tr_f1_cart,  te_f1_cart,  tr_acc_cart,  te_acc_cart  = \
    find_best_alpha('gini',    Xtr, Xte, y_train, y_test, "CART  (criterion=gini)")

alpha_chaid, alphas_chaid, tr_f1_chaid, te_f1_chaid, tr_acc_chaid, te_acc_chaid = \
    find_best_alpha('entropy', Xtr, Xte, y_train, y_test, "CHAID (criterion=entropy)")

# ─────────────────────────────────────────────────────────
# PASUL 2: Grafic alpha vs F1 pentru CART și CHAID
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Cost-Complexity Pruning — CART (Gini) vs CHAID (Entropy)',
             fontsize=14, fontweight='bold')

pairs = [
    (axes[0, 0], axes[0, 1], 'CART',  alphas_cart,
     tr_f1_cart,  te_f1_cart,  tr_acc_cart,  te_acc_cart,  alpha_cart,  '#4C72B0'),
    (axes[1, 0], axes[1, 1], 'CHAID', alphas_chaid,
     tr_f1_chaid, te_f1_chaid, tr_acc_chaid, te_acc_chaid, alpha_chaid, '#55A868'),
]

for ax_f1, ax_acc, label, alphas, tr_f1, te_f1, tr_acc, te_acc, best_a, col in pairs:

    # ── F1 vs alpha ──
    ax_f1.plot(alphas, tr_f1, 'o-', color=col,      alpha=0.7,
               markersize=3, label='Train F1')
    ax_f1.plot(alphas, te_f1, 'o-', color='#DD8452', alpha=0.7,
               markersize=3, label='Test F1')
    ax_f1.axvline(best_a, color='red', linestyle='--', linewidth=1.8,
                  label=f'α optim = {best_a:.5f}')
    ax_f1.set_xlabel('ccp_alpha')
    ax_f1.set_ylabel('F1-Score')
    ax_f1.set_title(f'{label} — F1 vs ccp_alpha', fontweight='bold')
    ax_f1.legend(fontsize=9)
    ax_f1.grid(alpha=0.3)

    # ── Accuracy vs alpha ──
    ax_acc.plot(alphas, tr_acc, 'o-', color=col,      alpha=0.7,
                markersize=3, label='Train Accuracy')
    ax_acc.plot(alphas, te_acc, 'o-', color='#DD8452', alpha=0.7,
                markersize=3, label='Test Accuracy')
    ax_acc.axvline(best_a, color='red', linestyle='--', linewidth=1.8,
                   label=f'α optim = {best_a:.5f}')
    ax_acc.set_xlabel('ccp_alpha')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title(f'{label} — Accuracy vs ccp_alpha', fontweight='bold')
    ax_acc.legend(fontsize=9)
    ax_acc.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../outputs/pruning_alpha_cart_chaid.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# PASUL 3: Antrenare modele finale cu alpha optim
# ─────────────────────────────────────────────────────────

cart_final = DecisionTreeClassifier(
    criterion='gini',
    ccp_alpha=alpha_cart,
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=7
)
cart_final.fit(Xtr, y_train)

chaid_final = DecisionTreeClassifier(
    criterion='entropy',
    ccp_alpha=alpha_chaid,
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=7
)
chaid_final.fit(Xtr, y_train)

# QUEST rămâne fără ccp_alpha — fasonarea lui e prin min_samples
quest_final = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=7
)
quest_final.fit(Xtr, y_train)

# ─────────────────────────────────────────────────────────
# PASUL 4: Evaluare completă + criterii de divizare/fasonare
# ─────────────────────────────────────────────────────────

def metrici_complete(model, Xte, y_test, cv_X, cv_y, label):
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    cm     = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc   = (tp + tn) / (tp + tn + fp + fn)
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    auc   = roc_auc_score(y_test, y_prob)
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0
    cv_f1 = cross_val_score(model, cv_X, cv_y, cv=10, scoring='f1')

    print(f"\n{'═'*55}")
    print(f"  {label}")
    print(f"  Criteriu divizare  : {model.criterion.upper()}")
    print(f"  Criteriu fasonare  : ccp_alpha={model.ccp_alpha:.6f} | "
          f"min_samples_leaf={model.min_samples_leaf} | "
          f"min_samples_split={model.min_samples_split}")
    print(f"  Adâncime finală    : {model.get_depth()}")
    print(f"  Nr. frunze         : {model.get_n_leaves()}")
    print(f"{'─'*55}")
    print(f"  Accuracy           : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision          : {prec:.4f}")
    print(f"  Recall             : {rec:.4f}")
    print(f"  Specificity        : {spec:.4f}")
    print(f"  F1-Score           : {f1:.4f}")
    print(f"  AUC-ROC            : {auc:.4f}")
    print(f"  CV F1 (10-fold)    : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
    print(f"{'═'*55}")

    return {
        'Model': label, 'Accuracy': round(acc, 4),
        'Precision': round(prec, 4), 'Recall': round(rec, 4),
        'Specificity': round(spec, 4), 'F1': round(f1, 4),
        'AUC': round(auc, 4), 'CV_F1': round(cv_f1.mean(), 4),
        'CV_Std': round(cv_f1.std(), 4),
        'Adancime': model.get_depth(), 'Frunze': model.get_n_leaves(),
        'y_pred': y_pred, 'y_prob': y_prob, 'cm': cm
    }

r_cart  = metrici_complete(cart_final,  Xte, y_test, Xtr, y_train,
                            f'CART  | Gini | α={alpha_cart:.5f}')
r_chaid = metrici_complete(chaid_final, Xte, y_test, Xtr, y_train,
                            f'CHAID | Entropy | α={alpha_chaid:.5f}')
r_quest = metrici_complete(quest_final, Xte, y_test, Xtr, y_train,
                            'QUEST | Gini | fără ccp_alpha')

# ─────────────────────────────────────────────────────────
# PASUL 5: Vizualizare arbori + matrice confuzie + ROC
# ─────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(3, 3, figsize=(22, 18))
fig2.suptitle('Arbori Pruned + Matrice Confuzie + ROC — CART / CHAID / QUEST',
              fontsize=14, fontweight='bold')

modele_viz = [
    (cart_final,  r_cart,  'CART (Gini)',    '#4C72B0'),
    (chaid_final, r_chaid, 'CHAID (Entropy)','#55A868'),
    (quest_final, r_quest, 'QUEST (Gini)',   '#C44E52'),
]

from sklearn.metrics import roc_curve

for row, (model, r, label, col) in enumerate(modele_viz):

    # ── Col 0: Arbore vizual ──
    ax = axes2[row, 0]
    plot_tree(model, feature_names=top4_final,
              class_names=['Nu cumpără', 'Cumpără'],
              filled=True, rounded=True, fontsize=8, ax=ax,
              impurity=True, proportion=False)
    ax.set_title(f'{label}\nAdâncime={r["Adancime"]}  Frunze={r["Frunze"]}',
                 fontweight='bold', fontsize=10)

    # ── Col 1: Matrice confuzie ──
    ax2c = axes2[row, 1]
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues',
                ax=ax2c, linewidths=0.5,
                xticklabels=['Pred: FALSE', 'Pred: TRUE'],
                yticklabels=['Real: FALSE', 'Real: TRUE'],
                annot_kws={'size': 13, 'weight': 'bold'})
    ax2c.set_title(f'{label}\nMatrice Confuzie', fontweight='bold', fontsize=10)
    tn, fp, fn, tp = r['cm'].ravel()
    ax2c.set_xlabel(
        f"Acc={r['Accuracy']:.3f}  F1={r['F1']:.3f}  AUC={r['AUC']:.3f}",
        fontsize=9)

    # ── Col 2: Curba ROC ──
    ax3c = axes2[row, 2]
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax3c.plot(fpr, tpr, color=col, linewidth=2,
              label=f'AUC = {r["AUC"]:.4f}')
    ax3c.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax3c.fill_between(fpr, tpr, alpha=0.08, color=col)
    ax3c.set_xlabel('False Positive Rate')
    ax3c.set_ylabel('True Positive Rate')
    ax3c.set_title(f'{label}\nCurba ROC', fontweight='bold', fontsize=10)
    ax3c.legend(fontsize=10)
    ax3c.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../../outputs/arbori_pruned_final.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────────────────
# PASUL 6: Tabel comparativ final + explicație criterii
# ─────────────────────────────────────────────────────────

rezultate = pd.DataFrame([
    {k: v for k, v in r.items()
     if k not in ('y_pred', 'y_prob', 'cm')}
    for r in [r_cart, r_chaid, r_quest]
])

print("\n── TABEL COMPARATIV FINAL ──\n")
print(rezultate[['Model','Accuracy','Precision','Recall',
                 'F1','AUC','CV_F1','CV_Std',
                 'Adancime','Frunze']].to_string(index=False))

print("""
╔══════════════════════════════════════════════════════════════════╗
║              CRITERII DE DIVIZARE ȘI FASONARE                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  CART — criterion='gini'                                         ║
║  ┌─ Criteriu divizare: Indicele Gini                             ║
║  │   Gini = 1 - Σ p(k)²                                         ║
║  │   → 0 = nod pur, 0.5 = maxim dezordine (binar)               ║
║  │   La fiecare nod alege split-ul care minimizează              ║
║  │   Gini ponderat al nodurilor copil                            ║
║  └─ Criteriu fasonare: Cost-Complexity Pruning                   ║
║      Cost(T, α) = Eroare(T) + α × |frunze(T)|                   ║
║      α optim calculat pe drumul complet de pruning               ║
║      + min_samples_leaf=20 (frunza ≥ 20 obs.)                   ║
║      + min_samples_split=40 (split doar dacă nodul ≥ 40 obs.)   ║
║                                                                  ║
║  CHAID — criterion='entropy'                                     ║
║  ┌─ Criteriu divizare: Entropia informațională                   ║
║  │   Entropy = -Σ p(k) × log₂(p(k))                             ║
║  │   → 0 = nod pur, 1 = maxim dezordine (binar)                 ║
║  │   CHAID nativ folosește test Chi-Square (χ²) pentru           ║
║  │   variabile categoriale; entropy e cel mai apropiat           ║
║  │   analog disponibil în sklearn                                ║
║  └─ Criteriu fasonare: Cost-Complexity Pruning (idem CART)       ║
║      α optim independent de CART (calculat pe entropy path)     ║
║      + min_samples_leaf=20, min_samples_split=40                 ║
║                                                                  ║
║  QUEST — criterion='gini', fără ccp_alpha                        ║
║  ┌─ Criteriu divizare: Gini (idem CART)                          ║
║  │   QUEST nativ: test F pentru variabile numerice,              ║
║  │   test χ² pentru categoriale → în sklearn aproximat           ║
║  │   prin gini + splitter='best'                                 ║
║  └─ Criteriu fasonare: Pre-pruning exclusiv                      ║
║      min_samples_leaf=20 → oprește creșterea devreme             ║
║      min_samples_split=40 → restricție la nivel de nod           ║
║      (QUEST evită post-pruning, preferând restricții             ║
║       structurale care limitează complexitatea a priori)         ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ─────────────────────────────────────────────────────────
# PASUL 7: EXTRAGEREA ȘI SALVAREA REGULILOR
# ─────────────────────────────────────────────────────────

from sklearn.tree import _tree

def extract_rules(tree, feature_names, class_names=['Nu cumpără', 'Cumpără'], min_samples=20):
    tree_  = tree.tree_
    rules  = []

    def recurse(node, conditions):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            feat      = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node],  conditions + [f"{feat} ≤ {threshold:.3f}"])
            recurse(tree_.children_right[node], conditions + [f"{feat} > {threshold:.3f}"])
        else:
            samples   = int(tree_.n_node_samples[node])
            values    = tree_.value[node][0]
            total     = sum(values)
            class_idx = int(np.argmax(values))
            prob      = values[class_idx] / total

            if samples >= min_samples:
                rules.append({
                    'conditii': conditions,
                    'clasa':    class_names[class_idx],
                    'samples':  samples,
                    'prob':     round(prob, 4),
                    'n_false':  int(values[0]),
                    'n_true':   int(values[1])
                })

    recurse(0, [])
    rules.sort(key=lambda x: (x['clasa'] == 'Cumpără', x['prob']), reverse=True)
    return rules


def print_and_collect_rules(rules, model_name, top_n=5):
    """Afișează și returnează regulile ca listă de dict pentru salvare CSV."""
    print(f"\n{'='*65}")
    print(f"  {model_name} — Top {top_n} reguli CUMPĂRĂ + Top {top_n} NU CUMPĂRĂ")
    print(f"{'='*65}")

    collected = []

    for clasa_target in ['Cumpără', 'Nu cumpără']:
        subset = [r for r in rules if r['clasa'] == clasa_target][:top_n]
        print(f"\n  ── Clasa: {clasa_target} ──")

        for i, r in enumerate(subset, 1):
            print(f"\n  Regula {'C' if clasa_target == 'Cumpără' else 'N'}{i} "
                  f"| {r['samples']} sesiuni | prob={r['prob']*100:.1f}%")
            print(f"  {'─'*55}")
            for cond in r['conditii']:
                print(f"    DACĂ  {cond}")
            print(f"    {'─'*40}")
            print(f"    ATUNCI → {r['clasa']}")
            print(f"    Distribuție: {r['n_false']} nu cumpără / {r['n_true']} cumpără")

            # Colectare pentru CSV
            collected.append({
                'Model':        model_name,
                'Clasa':        r['clasa'],
                'Nr_regula':    i,
                'Conditii':     ' AND '.join(r['conditii']),
                'Nr_conditii':  len(r['conditii']),
                'Samples':      r['samples'],
                'Probabilitate':f"{r['prob']*100:.1f}%",
                'N_false':      r['n_false'],
                'N_true':       r['n_true']
            })

    return collected


# ── Extrage regulile din fiecare model ──
all_rules = []

rules_cart  = extract_rules(cart_final,  top4_final)
rules_chaid = extract_rules(chaid_final, top4_final)
rules_quest = extract_rules(quest_final, top4_final)

all_rules += print_and_collect_rules(rules_cart,  'CART  (Gini)')
all_rules += print_and_collect_rules(rules_chaid, 'CHAID (Entropy)')
all_rules += print_and_collect_rules(rules_quest, 'QUEST (Gini)')

# ── Salvare CSV ──
df_rules = pd.DataFrame(all_rules)
df_rules.to_csv('../../outputs/reguli_extrase_top4.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ Reguli salvate: outputs/reguli_extrase_top4.csv")
print(f"   Total reguli: {len(df_rules)}")

# ── Export text complet arbori ──
print("\n── ARBORE TEXT COMPLET — CART ──")
print(export_text(cart_final, feature_names=top4_final))

print("\n── ARBORE TEXT COMPLET — CHAID ──")
print(export_text(chaid_final, feature_names=top4_final))

print("\n── ARBORE TEXT COMPLET — QUEST ──")
print(export_text(quest_final, feature_names=top4_final))