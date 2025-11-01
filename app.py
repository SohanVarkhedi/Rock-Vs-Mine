from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# load saved model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form['features']
    try:
        input_data = [float(x.strip()) for x in features.split(',')]
        input_array = np.asarray(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        result = "Rock" if prediction == 'R' else "Mine"
        return render_template('index.html', prediction_text=f'The object is a {result}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {e}')

if __name__ == "__main__":
    app.run(debug=True)
