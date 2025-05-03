from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_generator import generate_synthetic_data
from models import train_multiple_models
from visualization import plot_multiple_roc_curves

def run_ml_models_demo():
    print("Running ML Models Comparison Demo...")

    # Generate synthetic data
    print("Generating synthetic genomic data...")
    df, _ = generate_synthetic_data(n_samples=2000, n_snps=200)

    # Split data
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(["true_prs", "disease"], axis=1),
        df["disease"],
        test_size=0.3,
        random_state=42
    )

    # Train and evaluate multiple models
    print("Training multiple models...")
    results = train_multiple_models(X_train, y_train, X_test, y_test)

    # Plot ROC curves
    print("Plotting ROC curves...")
    plot_multiple_roc_curves(results, "ROC Curves for Different Machine Learning Models")

    print("ML models comparison complete!")

if __name__ == "__main__":
    run_ml_models_demo()