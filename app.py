from flask import Flask, request, jsonify, render_template
import joblib, numpy as np

app = Flask(__name__)

# Load the saved models
logreg = joblib.load('ml_models/logistic_regression_tuned_model.pkl')
dtree = joblib.load('ml_models/decision_tree_tuned_model.pkl')
rf = joblib.load('ml_models/random_forest_tuned_model.pkl')
gboost = joblib.load('ml_models/gradient_boosting_tuned_model.pkl')
svm = joblib.load('ml_models/support_vector_machine_model.pkl')
xgb = joblib.load('ml_models/xgboost_tuned_model.pkl')
vcl = joblib.load('ml_models/voting_classifier_model.pkl')

# Dictionary of loaded models
models_dict = {
    'vcl': vcl,
    'xgb': xgb,
    'rf': rf,
    'gboost': gboost,
    'logreg': logreg,
    'dtree': dtree,
    'svm': svm
}

# Function to decode predictions
def show_pred(pred):
    return 'Customer Churns' if pred == 1 else 'Customer Does Not Churn'

# Convert data to numeric values from string values in the form
def form_to_numeric(form_data):
    numeric_data = []

    binary_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
    ]
    binary_map = {'Yes': 1, 'No': 0, '1': 1, '0': 0, 1: 1, 0: 0}

    for col in binary_cols:
        if form_data[col] in binary_map:
            numeric_data.append(binary_map[form_data[col]])
        else:
            raise ValueError(f"Unexpected value for {col}: {form_data[col]}")

    # tenure, MonthlyCharges, TotalCharges
    numeric_data.append(int(form_data['tenure']))
    numeric_data.append(float(form_data['MonthlyCharges']))
    numeric_data.append(float(form_data['TotalCharges']))

    # InternetService
    internet_service = form_data['InternetService']
    numeric_data.append(1 if internet_service == 'No' else 0)
    numeric_data.append(1 if internet_service == 'DSL' else 0)
    numeric_data.append(1 if internet_service == 'Fiber optic' else 0)

    # Contract
    contract = form_data['Contract']
    numeric_data.append(1 if contract == 'Month-to-month' else 0)
    numeric_data.append(1 if contract == 'One year' else 0)
    numeric_data.append(1 if contract == 'Two year' else 0)

    # PaymentMethod
    payment_method = form_data['PaymentMethod']
    numeric_data.append(1 if payment_method == 'Bank transfer (automatic)' else 0)
    numeric_data.append(1 if payment_method == 'Credit card (automatic)' else 0)
    numeric_data.append(1 if payment_method == 'Electronic check' else 0)
    numeric_data.append(1 if payment_method == 'Mailed check' else 0)

    # Gender
    gender = form_data['gender']
    numeric_data.append(1 if gender == 'Female' else 0)
    numeric_data.append(1 if gender == 'Male' else 0)

    return np.array(numeric_data).reshape(1, -1)

@app.route('/')
def home():
    result = [
            {'model':'Voting Classifier', 'prediction':' '},
            {'model':'XGBoost', 'prediction':' '},
            {'model':'Random Forest', 'prediction':' '},
            {'model':'Gradient Boosting', 'prediction':' '},
            {'model':'Logistic Regression', 'prediction':' '},
            {'model':'Decision Tree', 'prediction':' '},
            {'model':'SVM', 'prediction':' '}
              ]

    maind = {'customer': {}, 'predictions': result}

    return render_template('index.html', maind=maind)

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    new_array = form_to_numeric(form_data)

    # Key names for customer dictionary 
    cols = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 
            'InternetService_0', 'InternetService_DSL', 'InternetService_Fiber optic',
            'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
            'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 
            'gender_Female', 'gender_Male']
    
    custard = {}
    for k, v in zip(cols, new_array[0]):
        custard[k] = v

    # Make predictions
    predl = [show_pred(m.predict(new_array)[0]) for m in models_dict.values()]

    result = [
            {'model':'Voting Classifier', 'prediction':predl[0]},
            {'model':'XGBoost', 'prediction':predl[1]},
            {'model':'Random Forest', 'prediction':predl[2]},
            {'model':'Gradient Boosting', 'prediction':predl[3]},
            {'model':'Logistic Regression', 'prediction':predl[4]},
            {'model':'Decision Tree', 'prediction':predl[5]},
            {'model':'SVM', 'prediction':predl[6]}
    ]

    maind = {'customer': custard, 'predictions': result}

    return render_template('index.html', maind=maind)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.json
    new_array = form_to_numeric(data)

    # Make precictions
    predl = {model_name: show_pred(model.predict(new_array)[0]) for model_name, model in models_dict.items()}

    return jsonify(predl)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'Thumbs Up guys'}), 200

if __name__ == '__main__':
    app.run(debug=True)