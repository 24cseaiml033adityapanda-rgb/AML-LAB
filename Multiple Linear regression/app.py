import os
import joblib
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        study_hours = float(data.get('studyHours', 0))
        attendance = float(data.get('attendance', 0))
        assignment = float(data.get('assignment', 0))
        
        # Predict using the loaded model
        prediction = model.predict([[study_hours, attendance, assignment]])
        
        # Ensure it's not below 0 or above 100 conceptually (optional, but good practice)
        final_mark = max(0, min(100, prediction[0]))
        
        return jsonify({
            'prediction': round(final_mark, 2),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    app.run(debug=True)
