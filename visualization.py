import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc

def plot_prs_distribution(results_df):
    """Plot distribution of PRS by disease status"""
    plt.figure(figsize=(10, 6))
    sns.kdeplot(results_df[results_df.disease_status==0]['estimated_prs'],
                label='Controls', fill=True, alpha=0.5)
    sns.kdeplot(results_df[results_df.disease_status==1]['estimated_prs'],
                label='Cases', fill=True, alpha=0.5)
    plt.axvline(results_df['estimated_prs'].quantile(0.9), color='red', linestyle='--',
                label='90th Percentile (High Risk)')
    plt.title('Distribution of Polygenic Risk Scores by Disease Status')
    plt.xlabel('Polygenic Risk Score')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prs_distribution.png')
    plt.close()

def plot_roc_curve(y_true, y_pred, title="Receiver Operating Characteristic (ROC) Curve"):
    """Plot ROC curve for a prediction"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    plt.close()

    return roc_auc

def plot_multiple_roc_curves(results_dict, title="ROC Curves for Different Models"):
    """Plot multiple ROC curves for comparison"""
    plt.figure(figsize=(8, 6))
    for name, metrics in results_dict.items():
        plt.plot(metrics['fpr'], metrics['tpr'], lw=2,
                 label=f'{name} (AUC = {metrics["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('multiple_roc_curves.png')
    plt.close()

def plot_risk_stratification(stratification_df):
    """Plot disease rates by risk stratum and population"""
    results = stratification_df.groupby(['risk_stratum', 'population']).agg(
        disease_rate=('disease', 'mean'),
        count=('disease', 'count')
    ).reset_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='risk_stratum', y='disease_rate', hue='population', data=results)
    plt.title('Disease Rate by Risk Stratum and Population')
    plt.xlabel('Risk Stratum')
    plt.ylabel('Disease Rate')
    plt.ylim(0, 1)

    # Add sample counts above bars
    for i, row in enumerate(results.itertuples()):
        plt.text(i % 3 + (i//3)*0.1 - 0.1, row.disease_rate + 0.03,
                 f'n={row.count}', ha='center')

    plt.tight_layout()
    plt.savefig('risk_stratification.png')
    plt.close()