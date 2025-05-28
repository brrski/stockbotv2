import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from stockbot import StockAnalyzer
import time

st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = StockAnalyzer()
    st.session_state.last_update = None

def create_candlestick_chart(symbol, price_data):
    """Create a candlestick chart using plotly"""
    fig = go.Figure(data=[go.Candlestick(x=price_data.index,
                                        open=price_data['Open'],
                                        high=price_data['High'],
                                        low=price_data['Low'],
                                        close=price_data['Close'])])
    fig.update_layout(title=f'{symbol} Price History',
                     xaxis_title='Date',
                     yaxis_title='Price ($)',
                     template='plotly_dark')
    return fig

def main():
    st.title("Stock Analysis Dashboard")
    
    # Add refresh button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 Refresh Data"):
            st.session_state.analyzer.update_data()
            st.session_state.analyzer.analyze_stocks()
            st.session_state.last_update = time.strftime("%Y-%m-%d %H:%M:%S")
            st.success("Data refreshed!")
    
    with col2:
        if st.session_state.last_update:
            st.text(f"Last Update: {st.session_state.last_update}")

    # Display stock analysis results
    results = st.session_state.analyzer.analysis_results
    if results:
        # Convert results to DataFrame for easier display
        data = []
        for symbol, info in results.items():
            data.append({
                'Symbol': symbol,
                'Price': f"${info['current_price']}",
                '7d Change': f"{info['change_7d']}%",
                '30d Change': f"{info['change_30d']}%",
                '90d Change': f"{info['change_90d']}%",
                'YTD Change': f"{info['change_365d']}%",
                'RSI': round(info['rsi'], 2),
                'Trend': info['trend'],
                'Support': f"${info['support']}",
                'Resistance': f"${info['resistance']}",
                'Recommendation': info['recommendation'],
                'LLM Sentiment': info['llm_analysis']['sentiment'],
                'LLM Confidence': f"{info['llm_analysis']['confidence']:.1%}",
                '7d Prediction': f"{info['predictions']['prediction_7d']}%",
                '21d Prediction': f"{info['predictions']['prediction_21d']}%"
            })
        
        df = pd.DataFrame(data)
        
        # Style the dataframe
        def color_negative_red(val):
            try:
                value = float(val.replace('%', '').replace('$', ''))
                color = 'red' if value < 0 else 'green' if value > 0 else 'white'
                return f'color: {color}'
            except:
                return ''
        
        styled_df = df.style.applymap(color_negative_red, subset=['7d Change', '30d Change', '90d Change', 'YTD Change'])
        
        # Display the main table
        st.dataframe(styled_df, height=400)
        
        # Add detailed view for selected stock
        st.subheader("Detailed Stock Analysis")
        selected_symbol = st.selectbox("Select a stock for detailed analysis", list(results.keys()))
        
        if selected_symbol:
            col1, col2 = st.columns(2)
            
            with col1:
                # Display candlestick chart
                if selected_symbol in st.session_state.analyzer.price_data:
                    price_data = st.session_state.analyzer.price_data[selected_symbol]
                    fig = create_candlestick_chart(selected_symbol, price_data)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display detailed analysis
                stock_info = results[selected_symbol]
                st.write("### Technical Analysis")
                st.write(f"**Current Price:** ${stock_info['current_price']}")
                st.write(f"**RSI:** {stock_info['rsi']:.2f}")
                st.write(f"**Trend:** {stock_info['trend']}")
                st.write(f"**Support Levels:** ${stock_info['support']}")
                st.write(f"**Resistance Levels:** ${stock_info['resistance']}")
                
                st.write("### LLM Analysis")
                st.write(f"**Sentiment:** {stock_info['llm_analysis']['sentiment']}")
                st.write(f"**Confidence:** {stock_info['llm_analysis']['confidence']:.1%}")
                
                st.write("### Predictions")
                st.write(f"**7-Day Forecast:** {stock_info['predictions']['prediction_7d']}%")
                st.write(f"**21-Day Forecast:** {stock_info['predictions']['prediction_21d']}%")
                st.write(f"**Confidence:** {stock_info['predictions']['confidence']}%")
                
                # Display support/resistance analysis from LLM
                if 'support_resistance' in stock_info['llm_analysis']:
                    st.write("### Support & Resistance Analysis")
                    sr_analysis = stock_info['llm_analysis']['support_resistance']
                    st.write(sr_analysis['analysis'])

if __name__ == "__main__":
    main()
