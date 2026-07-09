import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

def calculate_linear_regression(x, y):
    n = len(x)
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
    
    if denominator == 0:
        return None 
    
    m = numerator / denominator
    c = mean_y - (m * mean_x)
    
    
    y_pred = [m * val + c for val in x]
    
    mae = sum(abs(y[i] - y_pred[i]) for i in range(n)) / n
    mse = sum((y[i] - y_pred[i]) ** 2 for i in range(n)) / n
    rmse = mse ** 0.5
    
    ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        "mean_x": round(mean_x, 4),
        "mean_y": round(mean_y, 4),
        "slope": round(m, 4),
        "intercept": round(c, 4),
        "equation": f"Y = {m:.4f}X {'+' if c >= 0 else '-'} {abs(c):.4f}",
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "y_pred_line": y_pred
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    x_vals = [float(val) for val in data.get('x', [])]
    y_vals = [float(val) for val in data.get('y', [])]
    
    if len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return jsonify({"error": "Need at least 2 data points of matching X and Y."}), 400
        
    result = calculate_linear_regression(x_vals, y_vals)
    if result is None:
        return jsonify({"error": "Variance in X is 0 (all X values are the same)."}), 400
        
    plt.figure(figsize=(8, 5))
    plt.scatter(x_vals, y_vals, color='#3498db', label='Data Points', zorder=5)
    
    y_pred_line = result['y_pred_line']
    sorted_pairs = sorted(zip(x_vals, y_pred_line))
    x_sorted = [p[0] for p in sorted_pairs]
    y_sorted = [p[1] for p in sorted_pairs]
    
    plt.plot(x_sorted, y_sorted, color='#e74c3c', label='Regression Line', linewidth=2)
    plt.xlabel('X Value')
    plt.ylabel('Y Value')
    plt.title('Linear Regression Plot')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
    result['plot_url'] = plot_url
        
    return jsonify(result)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        m = float(data.get('m'))
        c = float(data.get('c'))
        x = float(data.get('x'))
        y_pred = m * x + c
        return jsonify({"prediction": round(y_pred, 4)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
