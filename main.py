import os
import pickle

def run_all_demos():
    """Run all demos in sequence"""
    print("========== Genomic Disease Risk Prediction Demo ==========")
    print("Running complete demo workflow...")

    # Create output directory for plots
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Run Basic PRS Demo
    print("\n1. BASIC POLYGENIC RISK SCORE DEMO")
    print("=====================================")
    from demo_basic_prs import run_basic_prs_demo
    df, true_effects = run_basic_prs_demo()

    # Save true effects for future demos
    with open('true_effects.pkl', 'wb') as f:
        pickle.dump(true_effects, f)

    # Run ML Models Demo
    print("\n2. MACHINE LEARNING MODELS DEMO")
    print("================================")
    from demo_ml_models import run_ml_models_demo
    run_ml_models_demo()

    # Run Population Bias Demo
    print("\n3. POPULATION BIAS DEMO")
    print("=======================")
    from demo_population_bias import run_population_bias_demo
    pop_A, pop_B, model = run_population_bias_demo(true_effects)

    # Save population data for risk stratification demo
    with open('population_data.pkl', 'wb') as f:
        pickle.dump({
            'true_effects': true_effects,
            'pop_A': pop_A,
            'pop_B': pop_B,
            'model': model
        }, f)

    # Run Risk Stratification Demo
    print("\n4. RISK STRATIFICATION DEMO")
    print("===========================")
    from demo_risk_stratification import run_risk_stratification_demo
    run_risk_stratification_demo(true_effects, pop_A, pop_B, model)

    print("\nAll demos completed successfully!")
    print("Check the generated plots to visualize the results.")

if __name__ == "__main__":
    run_all_demos()