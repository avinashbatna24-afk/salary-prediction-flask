import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)  # Initialize the Flask App

# Load the already-trained model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # Get input values from the HTML form
    int_features = [int(x) for x in request.form.values()]

    # Convert inputs into numpy array
    final_features = [np.array(int_features)]

    # Make prediction
    prediction = model.predict(final_features)

    # Round prediction
    output = round(prediction[0], 2)

    # Display salary in Indian Rupees
    prediction_text = 'Employee Salary should be ₹ {:,.2f}'.format(output)

    return render_template(
        'index.html',
        prediction_text=prediction_text
    )


if __name__ == "__main__":
    app.run(debug=True)