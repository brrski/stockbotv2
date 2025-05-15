import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from transformers import pipeline
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QTableWidget, 
                            QTableWidgetItem)
from PyQt6.QtCore import Qt, QTimer

class StockAnalysisLLM:
    def __init__(self):
        # Using FinBERT model which is specifically trained on financial text
        self.model_name = "ProsusAI/finbert"
        self.analyzer = pipeline("text-classification", model=self.model_name)
        self.predictor = pipeline("text2text-generation", model="facebook/bart-large-cnn")
        
    def analyze_support_resistance(self, price_data, current_price):
        """Analyze support and resistance levels using price action across multiple timeframes"""
        # Calculate levels for different timeframes
        timeframes = {
            '5d': 5,
            '10d': 10,
            '30d': 30,
            '60d': 60,
            '90d': 90
        }
        
        levels = {}
        for period, days in timeframes.items():
            if len(price_data) >= days:
                period_data = price_data[-days:]
                levels[period] = {
                    'high': period_data['High'].max(),
                    'low': period_data['Low'].min(),
                    'avg': period_data['Close'].mean(),
                    'vol': period_data['Volume'].mean()
                }
        
        prompt = f"""
        Analyze the following price data to identify key support and resistance levels across multiple timeframes:
        Current Price: ${current_price}
        
        5-Day Analysis:
        High: ${levels['5d']['high']:.2f}
        Low: ${levels['5d']['low']:.2f}
        Average: ${levels['5d']['avg']:.2f}
        
        10-Day Analysis:
        High: ${levels['10d']['high']:.2f}
        Low: ${levels['10d']['low']:.2f}
        Average: ${levels['10d']['avg']:.2f}
        
        30-Day Analysis:
        High: ${levels['30d']['high']:.2f}
        Low: ${levels['30d']['low']:.2f}
        Average: ${levels['30d']['avg']:.2f}
        
        60-Day Analysis:
        High: ${levels['60d']['high']:.2f}
        Low: ${levels['60d']['low']:.2f}
        Average: ${levels['60d']['avg']:.2f}
        
        90-Day Analysis:
        High: ${levels['90d']['high']:.2f}
        Low: ${levels['90d']['low']:.2f}
        Average: ${levels['90d']['avg']:.2f}
        
        Identify the most significant support and resistance levels for each timeframe.
        Focus on:
        1. Key price levels with high volume
        2. Previous major pivots
        3. Round number psychological levels
        4. Clustering of highs and lows
        """
        
        analysis = self.predictor(prompt, max_length=200, num_return_sequences=1)[0]['generated_text']
        
        try:
            # Parse the response to extract levels for each timeframe
            timeframe_levels = {}
            for timeframe in timeframes.keys():
                period_lines = [l for l in analysis.lower().split('\n') if timeframe in l.lower()]
                support_lines = [l for l in period_lines if 'support' in l]
                resistance_lines = [l for l in period_lines if 'resistance' in l]
                
                supports = [self.extract_price(l, levels[timeframe]['low']) for l in support_lines]
                resistances = [self.extract_price(l, levels[timeframe]['high']) for l in resistance_lines]
                
                timeframe_levels[timeframe] = {
                    'support': sorted(set(supports)) if supports else [levels[timeframe]['low']],
                    'resistance': sorted(set(resistances)) if resistances else [levels[timeframe]['high']]
                }
            
            return {
                'timeframe_levels': timeframe_levels,
                'analysis': analysis
            }
            
        except Exception as e:
            print(f"Error parsing support/resistance levels: {e}")
            return {
                'timeframe_levels': {
                    period: {
                        'support': [data['low']],
                        'resistance': [data['high']]
                    } for period, data in levels.items()
                },
                'analysis': analysis
            }
    
    def extract_price(self, text, default_value):
        """Helper method to extract price values from text"""
        import re
        matches = re.findall(r'\$?(\d+\.?\d*)', text)
        return float(matches[0]) if matches else default_value

    def analyze_stock_data(self, stock_data, price_data):
        """Analyze stock data using LLM"""
        # Get support/resistance levels
        levels = self.analyze_support_resistance(price_data, stock_data['current_price'])
        
        # Create enhanced prompt with LLM-generated levels
        prompt = f"""
        Technical Analysis Summary:
        Current Price: ${stock_data['current_price']}
        RSI: {stock_data['rsi']}
        Trend: {stock_data['trend']}
        
        Support Levels: ${', $'.join(f'{x:.2f}' for x in levels['timeframe_levels']['30d']['support'])}
        Resistance Levels: ${', $'.join(f'{x:.2f}' for x in levels['timeframe_levels']['30d']['resistance'])}
        
        Analysis: {levels['analysis']}
        """
        
        # Get model's analysis
        result = self.analyzer(prompt)
        result[0]['support_resistance'] = levels
        return result[0]

    def predict_future_performance(self, stock_data):
        """Predict future stock performance using technical analysis and LLM"""
        # Calculate prediction weights based on technical indicators
        rsi_signal = 1 if stock_data['rsi'] < 30 else (-1 if stock_data['rsi'] > 70 else 0)
        trend_signal = 1 if stock_data['trend'] == 'UPTREND' else (-1 if stock_data['trend'] == 'DOWNTREND' else 0)
        
        # Calculate price position relative to support/resistance
        current_price = stock_data['current_price']
        support_7d = stock_data['support_7d'] or current_price
        resistance_7d = stock_data['resistance_7d'] or current_price
        price_position = (current_price - support_7d) / (resistance_7d - support_7d) if resistance_7d != support_7d else 0.5
        
        # Calculate momentum from recent changes
        momentum_7d = stock_data['change_7d'] / 7  # Daily rate
        momentum_30d = stock_data['change_30d'] / 30  # Daily rate
        
        # Combine signals for 7-day prediction
        base_prediction_7d = (
            (rsi_signal * 2.0) +  # RSI has double weight
            (trend_signal * 1.5) +  # Trend has 1.5x weight
            ((0.5 - price_position) * 3.0) +  # Price position relative to range
            (momentum_7d * 3.5) +  # Recent momentum has high weight
            (momentum_30d * 1.5)   # Longer-term momentum has less weight
        )
        
        # Scale prediction to reasonable range (-5% to +5% for 7 days)
        pred_7d = np.clip(base_prediction_7d, -5, 5)
        
        # 21-day prediction uses more weight on longer-term indicators
        base_prediction_21d = (
            (rsi_signal * 1.5) +   # RSI has less weight long-term
            (trend_signal * 2.5) +  # Trend has more weight
            ((0.5 - price_position) * 2.0) +
            (momentum_7d * 2.0) +
            (momentum_30d * 3.0)    # Long-term momentum more important
        )
        
        # Scale 21-day prediction (-12% to +12% range)
        pred_21d = np.clip(base_prediction_21d * 1.5, -12, 12)
        
        # Add confidence levels based on signal alignment
        signals = [rsi_signal, trend_signal, momentum_7d > 0, momentum_30d > 0]
        confidence = (sum(1 for s in signals if s != 0) / len(signals)) * 100
        
        return {
            'prediction_7d': round(pred_7d, 1),
            'prediction_21d': round(pred_21d, 1),
            'confidence': round(confidence, 1),
            'signals': {
                'rsi': rsi_signal,
                'trend': trend_signal,
                'price_position': price_position,
                'momentum_7d': momentum_7d,
                'momentum_30d': momentum_30d
            }
        }

class StockAnalyzer:
    def __init__(self):
        self.symbols = ['QQQ', 'SPY', 'AAPL', 'MSFT', 'NVDA', 'AMZN', 'AVGO', 'META', 'NFLX', 'COST', 'GOOGL', 'GOOG', 'TSLA', 'TMUS', 'CSCO', 'LIN', 'PLTR', 'UNH', 'JNJ', 'XOM', 'JPM', 'V']
        self.price_data = {}
        self.analysis_results = {}
        self.llm = StockAnalysisLLM()
        self.update_data()

    def update_data(self):
        """Fetch and update price data for all symbols"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=305)  # 200 days of data
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date)
                if not df.empty:
                    self.price_data[symbol] = df
                    # Get real-time current price using both methods
                    self.price_data[symbol]['current_price'] = ticker.fast_info['last_price']
                    # Method 2
                    # Alternative: ticker.info.get('currentPrice') # Method 1
                    print(f"Fetched data for {symbol}: {len(df)} rows")
                else:
                    print(f"No data received for {symbol}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")

    def find_resistance_zones(self, df, lookback=50, sensitivity=0.005):
        """
        Identifies resistance zones based on swing highs.
        
        Parameters:
            df (pd.DataFrame): DataFrame with 'High' and 'Low' columns.
            lookback (int): Number of periods to look back for swing highs.
            sensitivity (float): Tolerance for grouping resistance levels (default: 0.5% of price).
        
        Returns:
            List of resistance zones.
        """
        highs = df['High'].rolling(window=lookback, center=True).max()
        swing_highs = df[df['High'] == highs]['High']
        
        # Group resistance levels into zones
        zones = []
        for level in sorted(swing_highs.dropna().values, reverse=True):
            if not zones or abs(level - zones[-1]) / level > sensitivity:
                zones.append(level)

        return zones

    def analyze_stocks(self):
        """Perform analysis on all stocks"""
        for symbol in self.symbols:
            if symbol in self.price_data and not self.price_data[symbol].empty:
                df = self.price_data[symbol]
            
                # Use the real-time price from update_data method
                current_price = df['current_price'].iloc[-1]
            
                # Calculate returns for different periods
                price_30d = df['Close'][-30] if len(df) >= 30 else df['Close'][0]
                price_90d = df['Close'][-90] if len(df) >= 90 else df['Close'][0]
                price_365d = df['Close'][-365] if len(df) >= 365 else df['Close'][0]
                price_7d = df['Close'][-7] if len(df) >= 7 else df['Close'][0]
            
                change_30d = ((current_price - price_30d) / price_30d * 100)
                change_90d = ((current_price - price_90d) / price_90d * 100)
                change_365d = ((current_price - price_365d) / price_365d * 100)
                change_7d = ((current_price - price_7d) / price_7d * 100)
            
                # Calculate RSI
                rsi = self.calculate_rsi(df['Close'])
            
                # Determine trend
                trend = self.determine_trend(df)
            
                # Calculate support and resistance levels
                support = df['Low'][-20:].min()
                resistance = df['High'][-20:].max()
                resistance_zones = self.find_resistance_zones(df)

                # Calculate 7-day and 30-day support and resistance levels
                support_7d = df['Low'][-7:].min() if len(df) >= 7 else None
                resistance_7d = df['High'][-7:].max() if len(df) >= 7 else None
                support_30d = df['Low'][-30:].min() if len(df) >= 30 else None
                resistance_30d = df['High'][-30:].max() if len(df) >= 30 else None
            
                # Generate recommendation
                recommendation = self.generate_recommendation(rsi, trend, change_30d)
            
                self.analysis_results[symbol] = {
                    'current_price': round(current_price, 2),
                    'change_30d': round(change_30d, 2),
                    'change_90d': round(change_90d, 2),
                    'change_365d': round(change_365d, 2),
                    'change_7d': round(change_7d, 2),
                    'rsi': round(rsi, 2),
                    'trend': trend,
                    'support': round(support, 2),
                    'resistance': round(resistance, 2),
                    'resistance_zones': resistance_zones,
                    'support_7d': round(support_7d, 2) if support_7d else None,
                    'resistance_7d': round(resistance_7d, 2) if resistance_7d else None,
                    'support_30d': round(support_30d, 2) if support_30d else None,
                    'resistance_30d': round(resistance_30d, 2) if resistance_30d else None,
                    'recommendation': recommendation
                }
                
                # Add LLM analysis
                llm_analysis = self.llm.analyze_stock_data(self.analysis_results[symbol], df)
                self.analysis_results[symbol]['llm_analysis'] = {
                    'sentiment': llm_analysis['label'],
                    'confidence': round(llm_analysis['score'], 3),
                    'support_resistance': llm_analysis['support_resistance']
                }
                
                # Add predictions
                predictions = self.llm.predict_future_performance(self.analysis_results[symbol])
                self.analysis_results[symbol]['predictions'] = predictions
                
                print(f"Analysis and predictions completed for {symbol}")
            else:
                print(f"No data available for analysis of {symbol}")

    def calculate_rsi(self, prices, periods=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:periods+1]
        up = seed[seed >= 0].sum()/periods
        down = -seed[seed < 0].sum()/periods
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:periods] = 100. - 100./(1.+rs)

        for i in range(periods, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(periods-1) + upval)/periods
            down = (down*(periods-1) + downval)/periods
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        return rsi[-1]

    def determine_trend(self, df):
        """Determine price trend"""
        if len(df) >= 50:  # Ensure we have enough data for both MAs
            ma20 = df['Close'].rolling(window=20).mean()
            ma50 = df['Close'].rolling(window=50).mean()
            
            if df['Close'][-1] > ma20[-1] > ma50[-1]:
                return 'UPTREND'
            elif df['Close'][-1] < ma20[-1] < ma50[-1]:
                return 'DOWNTREND'
        return 'SIDEWAYS'

    def generate_recommendation(self, rsi, trend, change_30d):
        """Generate trading recommendation"""
        if trend == 'UPTREND' and rsi < 70:
            return 'BUY'
        elif trend == 'DOWNTREND' and rsi > 30:
            return 'SELL'
        elif rsi > 70:
            return 'OVERBOUGHT'
        elif rsi < 30:
            return 'OVERSOLD'
        return 'HOLD'

    def get_analysis_summary(self):
        """Generate a summary of the current analysis for the AI assistant"""
        summary = "Current Stock Analysis:\n\n"
        for symbol, data in self.analysis_results.items():
            summary += f"{symbol}: ${data['current_price']} - {data['recommendation']} - RSI: {data['rsi']} - Trend: {data['trend']}\n"
        return summary


class AnalyzerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = StockAnalyzer()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('TraderBot')
        self.setGeometry(100, 100, 1200, 600)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Header with refresh button and last update time
        header = QHBoxLayout()
        refresh_btn = QPushButton('Refresh Analysis')
        refresh_btn.clicked.connect(self.refresh_analysis)
        header.addWidget(refresh_btn)
        
        self.last_update_label = QLabel('Last Update: Never')
        header.addWidget(self.last_update_label)
        header.addStretch()
        main_layout.addLayout(header)
        
        # Analysis table
        self.analysis_table = QTableWidget()
        self.analysis_table.setColumnCount(20)  # Updated column count
        self.analysis_table.setHorizontalHeaderLabels([
            'Symbol', 'Current Price', '7d Change %', '30d Change %', '90d Change %', 'YTD Change %',
            'RSI', 'Trend', 'Support', 'Resistance', 'Recommendation', 'Resistance Zones',
            '7d Support', '7d Resistance', '30d Support', '30d Resistance',
            'LLM Sentiment', 'LLM Confidence', '7d Prediction %', '21d Prediction %'
        ])
        main_layout.addWidget(self.analysis_table)
        
        # Initial analysis
        self.refresh_analysis()
        
        # Auto-refresh timer (5 minutes)
        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_analysis)
        self.timer.start(300000)  # 300,000 ms = 5 minutes

    def refresh_analysis(self):
        """Refresh all analysis data"""
        try:
            print("Starting analysis refresh...")
            self.analyzer.update_data()
            self.analyzer.analyze_stocks()
            self.update_table()
            
            self.last_update_label.setText(
                f'Last Update: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            )
            print("Analysis refresh completed")
        except Exception as e:
            print(f"Error refreshing analysis: {e}")

    def update_table(self):
        """Update the analysis table with latest results"""
        results = self.analyzer.analysis_results
        self.analysis_table.setRowCount(len(results))
        
        for i, (symbol, data) in enumerate(results.items()):
            try:
                # Create and set table items
                items = [
                    QTableWidgetItem(symbol),
                    QTableWidgetItem(f"${data['current_price']}"),
                    QTableWidgetItem(f"{data['change_7d']}%"),
                    QTableWidgetItem(f"{data['change_30d']}%"),
                    QTableWidgetItem(f"{data['change_90d']}%"),
                    QTableWidgetItem(f"{data['change_365d']}%"),
                    QTableWidgetItem(f"{data['rsi']}"),
                    QTableWidgetItem(data['trend']),
                    QTableWidgetItem(f"${data['support']}"),
                    QTableWidgetItem(f"${data['resistance']}"),
                    QTableWidgetItem(data['recommendation']),
                    QTableWidgetItem(", ".join([f"${zone:.2f}" for zone in data['resistance_zones']])),
                    QTableWidgetItem(f"${data['support_7d']}" if data['support_7d'] else "N/A"),
                    QTableWidgetItem(f"${data['resistance_7d']}" if data['resistance_7d'] else "N/A"),
                    QTableWidgetItem(f"${data['support_30d']}" if data['support_30d'] else "N/A"),
                    QTableWidgetItem(f"${data['resistance_30d']}" if data['resistance_30d'] else "N/A"),
                    QTableWidgetItem(data['llm_analysis']['sentiment']),
                    QTableWidgetItem(f"{data['llm_analysis']['confidence']:.1%}"),
                    QTableWidgetItem(f"{data['predictions']['prediction_7d']:.1f}%"),
                    QTableWidgetItem(f"{data['predictions']['prediction_21d']:.1f}%")
                ]
                
                # Set items in table
                for col, item in enumerate(items):
                    self.analysis_table.setItem(i, col, item)
                
                # Color code the recommendation
                rec_item = self.analysis_table.item(i, 10)
                if data['recommendation'] == 'BUY':
                    rec_item.setBackground(Qt.GlobalColor.green)
                elif data['recommendation'] == 'SELL':
                    rec_item.setBackground(Qt.GlobalColor.red)
                elif data['recommendation'] in ['OVERBOUGHT', 'OVERSOLD']:
                    rec_item.setBackground(Qt.GlobalColor.yellow)
                
            except Exception as e:
                print(f"Error updating table row for {symbol}: {e}")
        
        # Adjust column widths
        self.analysis_table.resizeColumnsToContents()
        print("Table updated successfully")


def main():
    app = QApplication(sys.argv)
    ex = AnalyzerGUI()
    ex.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()