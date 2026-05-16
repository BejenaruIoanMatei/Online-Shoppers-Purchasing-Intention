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

df = pd.read_csv('../../data/processed/online_shoppers_final_v2.csv')
df.drop(columns=['Revenue_int'], inplace=True)

X = df.drop('Revenue', axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=7, stratify=y
)

top4_final = ['PageValues', 'ExitRates', 'Month', 'ProductRelated_Duration']

print(f"Set comun folosit pentru toți algoritmii: {top4_final}")

# ── CART pe top4 ──
from sklearn.tree import DecisionTreeClassifier

cart = DecisionTreeClassifier(
    criterion='gini',
    ccp_alpha=0.001,
    min_samples_leaf=20,
    random_state=7
)
cart.fit(X_train[top4_final], y_train)
y_pred_cart = cart.predict(X_test[top4_final])
y_prob_cart = cart.predict_proba(X_test[top4_final])[:, 1]

# ── CHAID pe top4 ──
chaid = DecisionTreeClassifier(
    criterion='entropy',
    ccp_alpha=0.001,
    min_samples_leaf=20,
    random_state=7
)
chaid.fit(X_train[top4_final], y_train)
y_pred_chaid = chaid.predict(X_test[top4_final])
y_prob_chaid = chaid.predict_proba(X_test[top4_final])[:, 1]

# ── QUEST pe top4 ──
quest = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    min_samples_leaf=20,
    min_samples_split=40,
    random_state=7
)
quest.fit(X_train[top4_final], y_train)
y_pred_quest = quest.predict(X_test[top4_final])
y_prob_quest = quest.predict_proba(X_test[top4_final])[:, 1]

# ── Evaluare și comparație ──
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

rezultate_egale = pd.DataFrame([
    {
        'Model': 'CART',
        'Set': 'Top4 RF',
        'F1':  round(f1_score(y_test, y_pred_cart), 4),
        'AUC': round(roc_auc_score(y_test, y_prob_cart), 4),
        'Acc': round(accuracy_score(y_test, y_pred_cart), 4)
    },
    {
        'Model': 'CHAID',
        'Set': 'Top4 RF',
        'F1':  round(f1_score(y_test, y_pred_chaid), 4),
        'AUC': round(roc_auc_score(y_test, y_prob_chaid), 4),
        'Acc': round(accuracy_score(y_test, y_pred_chaid), 4)
    },
    {
        'Model': 'QUEST',
        'Set': 'Top4 RF',
        'F1':  round(f1_score(y_test, y_pred_quest), 4),
        'AUC': round(roc_auc_score(y_test, y_prob_quest), 4),
        'Acc': round(accuracy_score(y_test, y_pred_quest), 4)
    },
])

print(rezultate_egale.to_string(index=False))