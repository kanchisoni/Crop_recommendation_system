from flask import Flask, render_template, request
import pickle
import numpy as np

flask_app = Flask(__name__, template_folder='templates')

# Load the model
model = pickle.load(open('crop_recommendation_model.pkl', 'rb'))

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from form
        input_values = request.form.to_dict()  # This will store the form data
        float_features = [float(value) for value in input_values.values()]
        features = [np.array(float_features)]
        
        # Make prediction
        prediction = model.predict(features)
        
        # Display input values and prediction
        return render_template("index.html", 
                               prediction_text=f"The Predicted Crop is {prediction[0]}",
                               input_values=input_values)  # Pass input values to the template for retention
    except Exception as e:
        print(f"Error: {e}")
        return render_template("index.html", prediction_text="Error occurred during prediction.", input_values=None)

if __name__ == '__main__':
    flask_app.run(debug=True)
