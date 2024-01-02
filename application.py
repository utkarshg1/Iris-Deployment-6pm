# Import the necessary packages
from flask import Flask, request, render_template
import pickle
import pandas as pd 

# Initalize a flask application
application = Flask(__name__)
app = application

# Show the homepage
@app.route('/')
def home_page():
    return render_template('index.html')

# Create the prediction Logic
@app.route('/predict', methods=['POST'])
def predict_species():
    if request.method=='GET':
        return render_template('index.html')
    else:
        # Take input from user
        sep_len = float(request.form.get('sepal_length'))
        sep_wid = float(request.form.get('sepal_width'))
        pet_len = float(request.form.get('petal_length'))
        pet_wid = float(request.form.get('petal_width'))
        # Load the preprocessor and model
        with open('notebook/preprocess.pkl', 'rb') as file1:
            pre = pickle.load(file1)
        with open('notebook/Model.pkl', 'rb') as file2:
            model = pickle.load(file2)
        # Convert the inputs to dataframe
        xnew = pd.DataFrame([sep_len, sep_wid, pet_len, pet_wid]).T
        xnew.columns = pre.get_feature_names_out()
        # Transform the dataframe
        xnew_pre = pre.transform(xnew)  
        # Perform predictions
        prediction = model.predict(xnew_pre)[0]
        # Get the probaiblity
        prob = model.predict_proba(xnew_pre).max()
        prob = round(prob, 4)
        return render_template('index.html', prediction=prediction, prob=prob)

# Run the application
if __name__=='__main__':
    app.run(host='0.0.0.0', debug=True)