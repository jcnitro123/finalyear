from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('modelRF.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return "D-DOC API ML MODEL"


@app.route('/predict', methods=['POST'])
def predict():
    age = request.form.get('age')
    gender = request.form.get('gender')
    symptom_1 = request.form.get('symptom_1')
    symptom_2 = request.form.get('symptom_2')
    symptom_3 = request.form.get('symptom_3')
    symptom_4 = request.form.get('symptom_4')
    symptom_5 = request.form.get('symptom_5')
    symptom_6 = request.form.get('symptom_6')
    symptom_7 = request.form.get('symptom_7')
    symptom_8 = request.form.get('symptom_8')
    symptom_9 = request.form.get('symptom_9')

    input_query = np.array([[age, gender, symptom_1, symptom_2, symptom_3, symptom_4, symptom_5, symptom_6, symptom_7,
                             symptom_8, symptom_9]])

    predictions = model.predict_proba(input_query)[0]
    top3_indexes = predictions.argsort()[-3:][::-1]
    top3_results = [model.classes_[i] for i in top3_indexes]

    return jsonify({'top3_diseases': top3_results})


if __name__ == '__main__':
    app.run(debug=True)
