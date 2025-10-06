# app.py
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))
species = ['Setosa', 'Versicolor', 'Virginica']

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            prediction_index = model.predict([features])[0]
            prediction_label = species[prediction_index]
            return render_template('index.html', prediction=prediction_label)
        except Exception as e:
            return f"Error: {e}"

    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
