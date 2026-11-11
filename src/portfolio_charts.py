# ============================================================
# Portfolio charts: paste this as ONE new cell at the END of
# uhi_analysis.ipynb, after all other cells have been run.
# It uses df_labelled, importances and grid_* from earlier cells
# and saves 4 PNG files into an "images" folder.
# ============================================================
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

os.makedirs('images', exist_ok=True)
sns.set_theme(style='whitegrid')
ORDER = ['low', 'moderate', 'severe']
COLORS = {'low': '#4C9F70', 'moderate': '#F2B134', 'severe': '#D64545'}

# Clean copy of the data for plotting (same fixes as the pipeline)
plot_df = df_labelled.copy()
plot_df['council_area'] = plot_df['council_area'].astype(str).str.lower().str.title()
plot_df = plot_df.drop_duplicates(subset=[c for c in plot_df.columns if c != 'index'])

def pretty(name):
    """num__tree_canopy_pct -> Tree canopy pct"""
    name = name.split('__', 1)[-1]
    name = re.sub(r'_pct$', ' %', name)
    name = re.sub(r'_km2$', ' km²', name)
    name = re.sub(r'_km$', ' (km)', name)
    name = re.sub(r'_m$', ' (m)', name)
    name = name.replace('_', ' ')
    return name[0].upper() + name[1:]

# ---------- 1. Top 15 feature importances ----------
top15 = importances.head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh([pretty(n) for n in top15.index], top15.values, color='#2E6F9E')
plt.xlabel('Importance (Random Forest)')
plt.title('Top 15 drivers of Urban Heat Island severity')
plt.tight_layout()
plt.savefig('images/top15_feature_importance.png', dpi=150)
plt.show()

# ---------- 2. Severity share by council area ----------
share = (pd.crosstab(plot_df['council_area'], plot_df['uhi_severity_class'], normalize='index')
           [ORDER] * 100).sort_values('severe')
ax = share.plot(kind='barh', stacked=True, figsize=(9, 5),
                color=[COLORS[c] for c in ORDER], width=0.75)
ax.set_xlabel('% of grid cells')
ax.set_ylabel('')
ax.set_title('UHI severity by council area (Perth)')
ax.legend(title='Severity', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('images/severity_by_council.png', dpi=150)
plt.show()

# ---------- 3. Key drivers vs severity ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, col, title in zip(axes,
                          ['tree_canopy_pct', 'impervious_surface_pct'],
                          ['Tree canopy %', 'Impervious surface %']):
    sns.boxplot(data=plot_df, x='uhi_severity_class', y=col, order=ORDER,
                hue='uhi_severity_class', palette=COLORS, legend=False, ax=ax)
    ax.set_xlabel('UHI severity')
    ax.set_ylabel(title)
    ax.set_title(f'{title} by severity class')
plt.tight_layout()
plt.savefig('images/drivers_vs_severity.png', dpi=150)
plt.show()

# ---------- 4. Model comparison ----------
scores = pd.Series({
    'Random Forest': grid_rf.best_score_,
    'k-NN': grid_knn.best_score_,
    'Naive Bayes': grid_nb.best_score_,
    'Decision Tree': grid_dt.best_score_,
}).sort_values()
plt.figure(figsize=(7, 3.5))
bars = plt.barh(scores.index, scores.values,
                color=['#B0B7C3'] * 3 + ['#2E6F9E'])
plt.bar_label(bars, fmt='%.3f', padding=4)
plt.xlim(0, 1)
plt.xlabel('Macro-F1 (5-fold stratified CV)')
plt.title('Model comparison')
plt.tight_layout()
plt.savefig('images/model_comparison.png', dpi=150)
plt.show()
