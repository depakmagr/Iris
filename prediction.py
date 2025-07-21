import joblib

model = joblib.load('model/iris_rf_model.pkl')

def predict(features):
    return model.predict(features)