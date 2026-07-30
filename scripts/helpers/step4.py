
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_label_vs_ranking_table(df_results, 
                                  column_name, 
                                  percentage = False,
                                  rank_order = ['top 1', 'top 3', 'top 5', 'top 15', 'outside top 15'],
                                  label_order = ['exact', 'close match', 'broader', 'narrow', 'related']):
    df_results = df_results.copy()
    df_results[column_name] = pd.Categorical(
        df_results[column_name],
        categories=rank_order,
        ordered=True
    )
    df_results['final_label'] = pd.Categorical(
        df_results['final_label'],
        categories=label_order,
        ordered=True
    )
    ct = pd.crosstab(
        df_results[column_name],
        df_results['final_label']
    )
    if percentage:
        ct = ct.div(ct.sum(axis=0), axis=1) * 100

    # Top-1 is also top-3, top-5, top-15 so adapt for that
    top_rows = ct.iloc[:-1].cumsum()

    # Samenvoegen
    ct_cumsum = pd.concat([top_rows, ct.iloc[[-1]]])

    return ct_cumsum

def plot_heatmap(df_results, percentage = False, ax=None, title='FSM'):
    if percentage:
        min_value = 0
        max_value = 100
    else: 
        min_value = min(df_results.min(axis=0))
        max_value = max(df_results.max(axis=0))
        if min_value != 0:
            max_value = max(abs(min_value), max_value)
            min_value = -max_value

    hm = sns.heatmap(df_results, cmap='coolwarm', annot=True, vmin=min_value, vmax=max_value,
                annot_kws={"size": 16}, ax=ax)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Labels', fontsize=18)
    ax.set_ylabel('Ranking Position Category', fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15, rotation=30, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15, rotation=0)

    # legenda / colorbar fontgrootte
    cbar = hm.collections[0].colorbar      # colorbar ophalen
    cbar.ax.tick_params(labelsize=14)      # grootte van de waarden naast de colorbar