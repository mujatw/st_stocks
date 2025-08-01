
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta


# Use Streamlit wide layout and compact controls
st.set_page_config(layout="wide")
compact_css = """
<style>
.stTextInput, .stTextArea, .stSlider, .stCheckbox, .stButton, .stSelectbox, .stNumberInput {
    margin-bottom: 0.2rem !important;
    padding: 0.1rem 0.3rem !important;
    font-size: 0.9rem !important;
}
.stSlider > div[data-baseweb="slider"] {
    min-height: 1.5em !important;
}
.st-bb, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ck, .st-cl, .st-cm, .st-cn, .st-co, .st-cp, .st-cq, .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz {
    margin-bottom: 0.1rem !important;
}
</style>
"""
st.markdown(compact_css, unsafe_allow_html=True)




# --- CONTROLS AT TOP ---
st.title('Stock Watchlist Chart Viewer')

default_watchlists = {
    'HK Tech': '0683.hk,1205.hk,2348.hk,1568.hk',
    'HK Small Caps': '0316.hk,8001.hk,0366.hk,1569.hk',
    'HK Misc': '2488.hk,0609.hk,1419.hk',
    'US Big Tech': 'AAPL,MSFT,GOOGL,AMZN,TSLA',
}
if 'watchlists' not in st.session_state:
    st.session_state['watchlists'] = default_watchlists.copy()

today = datetime.today()
five_years_ago = today - timedelta(days=5*365)
one_week_ago = today - timedelta(days=7)

cols = st.columns(len(st.session_state['watchlists']))
selected_watchlists = []
for idx, name in enumerate(st.session_state['watchlists']):
    with cols[idx]:
        if st.checkbox(name, value=True, key=f"cb_{name}"):
            selected_watchlists.append(name)
tickers = []
for name in selected_watchlists:
    tickers += [t.strip().upper() for t in st.session_state['watchlists'][name].split(',') if t.strip()]
start_date = st.slider(
    '',
    min_value=five_years_ago.date(),
    max_value=one_week_ago.date(),
    value=(today - timedelta(days=180)).date(),
    format='YYYY-MM-DD',
    key='compact_slider',
)





# --- CHART RENDERING ---
if tickers:
    tickers_list = tickers
    if tickers_list:
        data = yf.download(tickers_list, start=start_date, end=today, group_by='ticker', auto_adjust=True)
        # --- Chart 1: Performance (start=100%) ---
        fig = go.Figure()
        performance = {}
        for ticker in tickers_list:
            try:
                if len(tickers_list) == 1:
                    prices = data['Close']
                else:
                    prices = data[ticker]['Close']
                # Normalize performance: start at 100
                if not prices.empty:
                    norm = prices / prices.iloc[0] * 100
                    fig.add_trace(go.Scatter(
                        x=norm.index,
                        y=norm,
                        mode='lines',
                        name=ticker,
                        hovertemplate='%{y:.0f}%<br>Date: %{x|%Y-%m-%d}<extra>' + ticker + '</extra>'
                    ))
                    perf = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                    performance[ticker] = perf
                else:
                    performance[ticker] = None
            except Exception as e:
                performance[ticker] = None
        # Custom y-axis ticks: round up to next integer percent
        import numpy as np
        y_min = -100
        y_max_candidates = [np.ceil(trace.y.max()) for trace in fig.data if hasattr(trace, 'y') and len(trace.y) > 0 and not np.isnan(trace.y.max())]
        y_max = max(y_max_candidates) if y_max_candidates else 100
        if y_max < 100 or np.isnan(y_max):
            y_max = 100
        # Use log scale, so ticks at 100, 200, 300, ... up to y_max
        tickvals = list(np.arange(100, y_max+1, 100))
        if y_min < 0:
            tickvals = [y_min] + tickvals
        ticktext = [f"{int(np.ceil(val))}%" for val in tickvals]
        # Add horizontal lines
        for y in tickvals:
            fig.add_shape(type="line", x0=norm.index.min(), x1=norm.index.max(), y0=y, y1=y, line=dict(color="lightgray", width=1, dash="dot"), xref="x", yref="y", layer="below")
        fig.update_layout(
            title=f'Performance (from {start_date}, start=100%)',
            xaxis_title='Date',
            yaxis_title='Percentage Change (start = 100%)',
            yaxis_type='log',
            yaxis=dict(
                range=[y_min, None],
                tickvals=tickvals,
                ticktext=ticktext
            ),
            autosize=True,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Chart 2: Min-Max Normalized Prices or Absolute Price if 1 ticker ---
        fig2 = go.Figure()
        for ticker in tickers_list:
            try:
                if len(tickers_list) == 1:
                    prices = data['Close']
                    if not prices.empty:
                        fig2.add_trace(go.Scatter(
                            x=prices.index,
                            y=prices,
                            mode='lines',
                            name=ticker,
                            hovertemplate='%{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra>' + ticker + '</extra>'
                        ))
                else:
                    prices = data[ticker]['Close']
                    if not prices.empty:
                        min_p = prices.min()
                        max_p = prices.max()
                        if max_p > min_p:
                            norm_mm = (prices - min_p) / (max_p - min_p)
                            fig2.add_trace(go.Scatter(
                                x=norm_mm.index,
                                y=norm_mm,
                                mode='lines',
                                name=ticker,
                                hovertemplate='%{y:.2f}<br>Date: %{x|%Y-%m-%d}<extra>' + ticker + '</extra>'
                            ))
            except Exception:
                pass
        # Add horizontal lines to second chart
        if len(tickers_list) == 1:
            ylines = np.linspace(prices.min(), prices.max(), 5) if not prices.empty else []
            for y in ylines:
                fig2.add_shape(type="line", x0=prices.index.min(), x1=prices.index.max(), y0=y, y1=y, line=dict(color="lightgray", width=1, dash="dot"), xref="x", yref="y", layer="below")
            fig2.update_layout(
                title='Absolute Price',
                xaxis_title='Date',
                yaxis_title='Price',
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20)
            )
        else:
            ylines = np.linspace(0, 1, 5)
            for y in ylines:
                fig2.add_shape(type="line", x0=norm_mm.index.min(), x1=norm_mm.index.max(), y0=y, y1=y, line=dict(color="lightgray", width=1, dash="dot"), xref="x", yref="y", layer="below")
            fig2.update_layout(
                title='Min-Max Normalized Prices',
                xaxis_title='Date',
                yaxis_title='Min-Max Normalized Price',
                autosize=True,
                margin=dict(l=20, r=20, t=40, b=20)
            )
        st.plotly_chart(fig2, use_container_width=True)


# --- INPUT CONTROLS BELOW CHART ---
import re
def watchlists_to_str(watchlists):
    return ' '.join(f'[{name}] {tickers}' for name, tickers in watchlists.items())
def str_to_watchlists(s):
    pattern = r'\[(.*?)\]\s*([^\[]*)'
    matches = re.findall(pattern, s)
    return {name.strip(): tickers.strip() for name, tickers in matches if name.strip()}
current_watchlists_str = watchlists_to_str(st.session_state['watchlists'])
new_watchlists_str = st.text_area('', current_watchlists_str, height=60)
if new_watchlists_str != current_watchlists_str:
    st.session_state['watchlists'] = str_to_watchlists(new_watchlists_str)


