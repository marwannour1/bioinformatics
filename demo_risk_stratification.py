import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

from data_generator import generate_population_data
from models import stratify_risk
from visualization import plot_risk_stratification

def run_risk_stratification_demo(true_effects, pop_A, pop_B, model):
    print("Running Risk Stratification Demo...")

    # Combine populations
    combined_df = pd.concat([pop_A, pop_B], ignore_index=True)

    # Calculate PRS
    print("Calculating risk scores...")
    X_test_combined = combined_df.drop(["true_prs", "disease", "population"], axis=1)
    y_test_combined = combined_df["disease"]
    pop_test = combined_df["population"]

    # Apply model to get predicted probabilities
    prs_values = model.predict_proba(X_test_combined)[:, 1]

    # Stratify into risk categories
    print("Stratifying individuals by risk...")
    risk_strata, cutoffs = stratify_risk(prs_values)
    print(f"Risk cutoffs - Low: {cutoffs['low']:.3f}, High: {cutoffs['high']:.3f}")

    # Create results dataframe
    stratification_df = pd.DataFrame({
        'prs': prs_values,
        'disease': y_test_combined,
        'risk_stratum': risk_strata,
        'population': pop_test
    })

    # Plot risk stratification results
    print("Plotting risk stratification results...")
    plot_risk_stratification(stratification_df)

    print("Risk stratification demo complete!")

if __name__ == "__main__":
    import pickle

    # Load data from previous demos
    try:
        with open('population_data.pkl', 'rb') as f:
            data = pickle.load(f)
        true_effects, pop_A, pop_B, model = data['true_effects'], data['pop_A'], data['pop_B'], data['model']
    except FileNotFoundError:
        print("Running population bias demo first...")
        from demo_basic_prs import run_basic_prs_demo
        _, true_effects = run_basic_prs_demo()

        from demo_population_bias import run_population_bias_demo
        pop_A, pop_B, model = run_population_bias_demo(true_effects)

        with open('population_data.pkl', 'wb') as f:
            pickle.dump({
                'true_effects': true_effects,
                'pop_A': pop_A,
                'pop_B': pop_B,
                'model': model
            }, f)

    run_risk_stratification_demo(true_effects, pop_A, pop_B, model)