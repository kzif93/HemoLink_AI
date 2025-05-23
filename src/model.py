from sklearn.ensemble import RandomForestClassifier

def get_default_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)
