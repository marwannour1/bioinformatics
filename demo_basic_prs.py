import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from data_generator import generate_synthetic_data
from models import train_gwas_model, calculate_prs
from visualization import plot_prs_distribution, plot_roc_curve

def run_basic_prs_demo():
    print("Running Basic PRS Demo...")

    # Generate synthetic data
    print("Generating synthetic genomic data...")
    df, true_effects = generate_synthetic_data(n_samples=2000, n_snps=200)

    # Split data
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["true_prs", "disease"], axis=1),
        df["disease"],
        test_size=0.3,
        random_state=42
    )

    # Train model to estimate effect sizes
    print("Training GWAS-like model...")
    gwas_model = train_gwas_model(X_train, y_train)
    estimated_effects = gwas_model.coef_[0]

    # Calculate PRS for test set
    print("Calculating PRS...")
    prs_test = calculate_prs(X_test, estimated_effects)

    # Create results dataframe for visualization
    results_df = pd.DataFrame({
        'true_prs': df.loc[X_test.index, 'true_prs'],
        'estimated_prs': prs_test,
        'disease_status': y_test
    })

    # Plot PRS distribution
    print("Plotting PRS distribution...")
    plot_prs_distribution(results_df)

    # Plot ROC curve
    print("Plotting ROC curve...")
    pred_probs = 1 / (1 + np.exp(-prs_test))
    auc_value = plot_roc_curve(y_test, pred_probs)

    print(f"Model AUC: {auc_value:.3f}")
    print("Basic PRS demo complete!")

    return df, true_effects

if __name__ == "__main__":
    df, true_effects = run_basic_prs_demo()