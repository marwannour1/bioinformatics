import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_generator import generate_population_data
from models import calculate_model_metrics
from visualization import plot_multiple_roc_curves

def run_population_bias_demo(true_effects):
    print("Running Population Bias Demo...")

    # Generate data for two populations
    print("Generating population-specific data...")
    pop_A = generate_population_data(n_samples=500, n_snps=len(true_effects),
                                    population='A', true_effects=true_effects)
    pop_B = generate_population_data(n_samples=500, n_snps=len(true_effects),
                                    population='B', true_effects=true_effects)

    # Train model only on population A
    print("Training model on population A only...")
    X_train = pop_A.drop(["true_prs", "disease", "population"], axis=1)
    y_train = pop_A["disease"]

    # Test on both populations
    X_test_A = pop_A.drop(["true_prs", "disease", "population"], axis=1)
    y_test_A = pop_A["disease"]
    X_test_B = pop_B.drop(["true_prs", "disease", "population"], axis=1)
    y_test_B = pop_B["disease"]

    # Train the model
    model = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
    model.fit(X_train, y_train)

    # Calculate performance on both populations
    print("Evaluating model on both populations...")
    metrics_A = calculate_model_metrics(model, X_test_A, y_test_A)
    metrics_B = calculate_model_metrics(model, X_test_B, y_test_B)

    # Plot ROC curves
    results = {
        'Population A (Training)': metrics_A,
        'Population B': metrics_B
    }
    print("Plotting ROC curves...")
    plot_multiple_roc_curves(results, "Impact of Population Bias on Model Performance")

    print("Population bias demo complete!")

    return pop_A, pop_B, model

if __name__ == "__main__":
    import pickle
    # Load true_effects from previous demo
    try:
        with open('true_effects.pkl', 'rb') as f:
            true_effects = pickle.load(f)
    except FileNotFoundError:
        print("Running basic PRS demo first to generate true effects...")
        from demo_basic_prs import run_basic_prs_demo
        _, true_effects = run_basic_prs_demo()
        with open('true_effects.pkl', 'wb') as f:
            pickle.dump(true_effects, f)

    pop_A, pop_B, model = run_population_bias_demo(true_effects)