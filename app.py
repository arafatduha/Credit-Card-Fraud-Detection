from flask import Flask, render_template, request
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
# For checking acuracy
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Create the Flask application
app = Flask(__name__)

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form values submitted by the user
    step = float(request.form['step'])
    amount = float(request.form['amount'])
    obalance = float(request.form['Obalance'])
    nbalance = float(request.form['nbalance'])
    obalance_dest = float(request.form['ObalanceDest'])
    nbalance_dest = float(request.form['nbalanceDest'])
    cashout = float(request.form['cashout'])
    debit = float(request.form['debit'])
    payment = float(request.form['payment'])
    transfer = float(request.form['transfer'])
    print(step)
    print(amount)

    # Perform fraud detection prediction using the form values
    #prediction = perform_fraud_detection(step, amount, obalance, nbalance, obalance_dest, nbalance_dest,cashout, debit, payment, transfer)

   
    with open('lrs_model.pkl', 'rb') as f:
        lrs2 = pickle.load(f)
    print(type(lrs2))

    # Make the prediction using the loaded model
    prediction = lrs2.predict([[step, amount, obalance,nbalance , obalance_dest, nbalance_dest,cashout, debit, payment, transfer]])
    result_label = "Non-fraud"
    if prediction == 1:
        result_label = "Fraud"
    return render_template('result.html', prediction=result_label)



    # Return the final prediction result (e.g., True for fraud, False for non-fraud)
    print(prediction)




# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
