import numpy as np
import pandas as pd

def generate_synthetic_data(n_samples=1000, n_snps=100, seed=42):
    """Generate synthetic SNP data with disease outcomes"""
    np.random.seed(seed)

    # Generate SNP data (0, 1, or 2 copies of risk allele)
    snp_data = np.random.randint(0, 3, size=(n_samples, n_snps))

    # Generate effect sizes for each SNP (most will be small)
    effect_sizes = np.random.normal(0, 0.05, n_snps)
    # Make a few SNPs have larger effects
    effect_sizes[0:5] = np.random.normal(0, 0.5, 5)

    # Calculate the true polygenic risk score
    true_prs = np.dot(snp_data, effect_sizes)

    # Generate disease status based on PRS + some noise
    probability = 1 / (1 + np.exp(-(true_prs + np.random.normal(0, 1, n_samples))))
    disease_status = np.random.binomial(1, probability)

    # Create dataframe
    columns = [f"SNP_{i}" for i in range(n_snps)]
    df = pd.DataFrame(snp_data, columns=columns)
    df["true_prs"] = true_prs
    df["disease"] = disease_status

    return df, effect_sizes

def generate_population_data(n_samples=500, n_snps=200, population='A', true_effects=None, seed=42):
    """Generate population-specific genomic data with different allele frequencies"""
    np.random.seed(seed)

    # Generate SNP data with different allele frequencies based on population
    if population == 'A':  # Reference population
        snp_data = np.random.binomial(2, 0.3, size=(n_samples, n_snps))
    else:  # Non-reference population
        snp_data = np.random.binomial(2, 0.5, size=(n_samples, n_snps))

    # Disease mechanism is slightly different between populations
    if population == 'A':
        # Effect sizes used to create model
        effect_sizes = true_effects
    else:
        # Effect sizes for population B are correlated but not identical
        effect_sizes = 0.7 * true_effects + 0.3 * np.random.normal(0, 0.1, n_snps)

    # Calculate PRS
    true_prs = np.dot(snp_data, effect_sizes)

    # Generate disease status
    probability = 1 / (1 + np.exp(-(true_prs + np.random.normal(0, 1, n_samples))))
    disease_status = np.random.binomial(1, probability)

    # Create dataframe
    columns = [f"SNP_{i}" for i in range(n_snps)]
    df = pd.DataFrame(snp_data, columns=columns)
    df["true_prs"] = true_prs
    df["disease"] = disease_status
    df["population"] = population

    return df