from flask import Flask, jsonify
from stockbot import StockAnalyzer
import pandas as pd

app = Flask(__name__)
analyzer = StockAnalyzer()

@app.route('/')
def serve_dashboard():
    with open('dashboard.html', 'r') as f:
        return f.read()

@app.route('/api/refresh', methods=['POST'])
def refresh_data():
    analyzer.update_data()
    analyzer.analyze_stocks()
    return jsonify(analyzer.analysis_results)

@app.route('/api/stock/<symbol>')
def get_stock_data(symbol):
    if symbol not in analyzer.analysis_results:
        return jsonify({'error': 'Stock not found'}), 404
        
    data = analyzer.analysis_results[symbol]
    price_data = analyzer.price_data[symbol]
    
    # Convert price data to format needed for plotly
    price_dict = {
        'dates': price_data.index.strftime('%Y-%m-%d').tolist(),
        'open': price_data['Open'].tolist(),
        'high': price_data['High'].tolist(),
        'low': price_data['Low'].tolist(),
        'close': price_data['Close'].tolist()
    }
    
    response = {**data}
    response['price_data'] = price_dict
    return jsonify(response)

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
