import yfinance as yf
from transformers import pipeline
import numpy as np
from datetime import datetime, timedelta
import os

class StockAgent:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.log_file = os.path.join(os.getcwd(), 'stock_predictions.txt')
        print(f"Log file will be created at: {self.log_file}")  # Debug print

    def log_prediction(self, data):
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            
            with open(self.log_file, 'a') as f:
                f.write(f"\n=== Stock Analysis Report - {timestamp} ===\n")
                f.write(f"Symbol: {data['symbol']}\n")
                f.write(f"Current Price: ${data['currentPrice']:.2f}\n")
                f.write(f"Change: {data['change']}\n")
                f.write(f"RSI: {data['rsi']:.2f}\n")
                f.write(f"Trend: {data['trend']}\n")
                f.write(f"Sentiment: {data['sentiment']} (Confidence: {data['confidence']})\n")
                f.write(f"7-Day Forecast: {data['forecast7d']}\n")
                f.write(f"21-Day Forecast: {data['forecast21d']}\n")
                f.write(f"Support Level: ${data['supportLevels'][0]:.2f}\n")
                f.write(f"Resistance Level: ${data['resistanceLevels'][0]:.2f}\n")
                f.write("-" * 50 + "\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")
            print(f"Attempted to write to: {self.log_file}")

    def get_stock_data(self, symbol):
        if symbol in self.cache and (datetime.now() - self.cache[symbol]['timestamp']).seconds < self.cache_timeout:
            return self.cache[symbol]['data']

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='3mo')
            
            # Calculate basic metrics
            prices = hist['Close'].tolist()
            returns = np.diff(prices) / prices[:-1]
            
            # Simple RSI
            gains = [max(0, r) for r in returns]
            losses = [abs(min(0, r)) for r in returns]
            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14
            rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
            rsi = 100 - (100 / (1 + rs))

            # Simple trend calculation
            short_ma = sum(prices[-5:]) / 5
            long_ma = sum(prices[-20:]) / 20
            trend = "UPTREND" if short_ma > long_ma else "DOWNTREND"

            # Basic sentiment analysis
            news = ticker.news[0].get('title', '') if ticker.news else ''
            sentiment = self.sentiment_analyzer(news)[0]
            
            # Simple price prediction
            price_change = (prices[-1] - prices[-7]) / prices[-7] * 100
            forecast7d = price_change
            forecast21d = price_change * 3

            data = {
                'symbol': symbol,
                'currentPrice': prices[-1],
                'change': f"{prices[-1] - prices[-2]:.2f}",
                'rsi': rsi,
                'trend': trend,
                'sentiment': sentiment['label'],
                'confidence': f"{sentiment['score']:.1%}",
                'forecast7d': f"{forecast7d:.1f}%",
                'forecast21d': f"{forecast21d:.1f}%",
                'supportLevels': [min(prices)],
                'resistanceLevels': [max(prices)]
            }

            self.cache[symbol] = {
                'timestamp': datetime.now(),
                'data': data
            }
            
            # Add this line before returning data
            self.log_prediction(data)
            
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def get_all_stocks(self):
        return [self.get_stock_data(symbol) for symbol in self.symbols if self.get_stock_data(symbol)]
