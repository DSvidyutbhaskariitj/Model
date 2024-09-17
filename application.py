from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

##import pickle file

lr_model = pickle.load(open('models/LogisticRegression.pkl','rb'))
svc_model = pickle.load(open('models/LogisticRegression.pkl','rb'))
rf_model = pickle.load(open('models/LogisticRegression.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_data',methods=['GET','POST'])

def predict_data():
    if request.method == 'POST':
        gender = request.form.get('Gender')
        multiple_lines = request.form.get('MultipleLines')
        internet_service = request.form.get('InternetService')
        online_security = request.form.get('OnlineSecurity')
        online_backup = request.form.get('OnlineBackup')
        device_protection = request.form.get('DeviceProtection')
        tech_support = request.form.get('TechSupport')
        streaming_tv = request.form.get('StreamingTV')
        streaming_movies = request.form.get('StreamingMovies')
        contract = request.form.get('Contract')
        payment_method = request.form.get('PaymentMethod')
        tenure = float(request.form.get('tenure'))  # Assuming tenure is numeric
        monthly_charges = float(request.form.get('MonthlyCharges'))  # Assuming monthly_charges is numeric
        total_charges = float(request.form.get('TotalCharges'))  # Assuming total_charges is numeric

        # Prepare feature array (convert categorical variables to numerical if necessary)
        features = np.array([[gender, multiple_lines, internet_service, online_security, online_backup,
                              device_protection, tech_support, streaming_tv, streaming_movies,
                              contract, payment_method, tenure, monthly_charges, total_charges]])
        
        
        # Make the prediction
        prediction = lr_model.predict(features)

        # Return the result as JSON
        return render_template('home.html', results= prediction[0])

    else:
        return render_template('home.html')


if __name__ == '__main__':
    application.run(debug=True)
