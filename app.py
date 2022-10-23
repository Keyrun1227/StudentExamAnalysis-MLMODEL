from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('studentperformance.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    arr = np.array([[data1, data2, data3, data4, data5, data6]])
    pred = model.predict(arr)
    return render_template('predict.html', data=pred[0])


if __name__ == "__main__":
    app.run(debug=True)
