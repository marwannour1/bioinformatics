import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train_gwas_model(X_train, y_train, C=0.1):
    """Train a GWAS-like model to estimate effect sizes"""
    gwas_model = LogisticRegression(penalty='l1', solver='liblinear', C=C)
    gwas_model.fit(X_train, y_train)
    return gwas_model

def calculate_prs(X_data, effect_sizes):
    """Calculate polygenic risk score"""
    return np.dot(X_data, effect_sizes)

def calculate_model_metrics(model, X_test, y_test):
    """Calculate ROC curve metrics for model evaluation"""
    pred_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    return {'fpr': fpr, 'tpr': tpr, 'auc': auc(fpr, tpr)}

def train_multiple_models(X_train, y_train, X_test, y_test):
    """Train multiple ML models and calculate their metrics"""
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Evaluate each model
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        results[name] = calculate_model_metrics(model, X_test_scaled, y_test)

    return results

def stratify_risk(prs_values):
    """Stratify individuals into risk categories based on PRS"""
    cutoffs = {
        'low': np.percentile(prs_values, 20),
        'high': np.percentile(prs_values, 80)
    }

    risk_categories = []
    for prs in prs_values:
        if prs <= cutoffs['low']:
            risk_categories.append('Low Risk')
        elif prs >= cutoffs['high']:
            risk_categories.append('High Risk')
        else:
            risk_categories.append('Average Risk')

    return risk_categories, cutoffs