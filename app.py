
import streamlit as st
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta

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

default_watchlists = {
    'sg': 're4.si,BVA.si,c2pu.si,chj.si,q5t.si,j85.si,BWCU.SI',
    'hk': '0316.hk,8001.hk,1569.hk,0683.hk,1205.hk,2348.hk,1568.hk,2488.hk,0609.hk,1419.hk,2001.hk,3618.hk,9616.hk,0366.hk,0422.hk,2219.hk,1883.hk',
    'us': 'jfin,ntla,land,tlk,atlx,indo,coin,cphi,sxtc,nvts,lit,nisn,gmm,dswl',
    'can': 'dtwo.v,rck.v,lth.v',
    'hk2':'0863.hk,1062.hk,0020.hk,0763.hk,0767.hk,3800.hk,2018.hk,2031.hk,0320.hk,2663.hk',
    'us2':'glxy,coin',
    'sz2':'300673.sz,000661.sz,603566.ss,300871.sz,000876.sz,688027.ss,002891.sz',
    'sg2':'f83.si'
}
if 'watchlists' not in st.session_state:
    st.session_state['watchlists'] = default_watchlists.copy()

today = datetime.today()
five_years_ago = today - timedelta(days=5*365)

# --- INPUT CONTROLS (moved above charts) ---
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
one_week_ago = today - timedelta(days=7)

cols = st.columns(len(st.session_state['watchlists']))
selected_watchlists = []
for idx, name in enumerate(st.session_state['watchlists']):
    with cols[idx]:
        if st.checkbox(name, value=True, key=f"cb_{name}"):
            selected_watchlists.append(name)
# --- Ticker and company name mapping, data caching ---
import yfinance as yf
tickers = []
company_names = {}
for name in selected_watchlists:
    tickers += [t.strip().upper() for t in st.session_state['watchlists'][name].split(',') if t.strip()]

# Use session_state to cache data and info for current watchlist
def get_watchlist_key(tickers):
    return ','.join(sorted(tickers))

watchlist_key = get_watchlist_key(tickers)

if 'cached_data' not in st.session_state:
    st.session_state['cached_data'] = {}
if 'cached_info' not in st.session_state:
    st.session_state['cached_info'] = {}

reload_needed = False
if st.session_state.get('last_watchlist_key') != watchlist_key:
    reload_needed = True
    st.session_state['last_watchlist_key'] = watchlist_key

if reload_needed and tickers:
    # Download 5 years of data for all tickers
    five_years_ago = today - timedelta(days=5*365)
    data = yf.download(tickers, start=five_years_ago, end=today, group_by='ticker', auto_adjust=True)
    st.session_state['cached_data'][watchlist_key] = data
    # Download info for all tickers
    info_dict = {}
    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            info = yf_ticker.info
            cname = info.get('shortName') or info.get('longName') or info.get('name') or ticker
            info_dict[ticker] = {'info': info, 'company_name': cname}
        except Exception:
            info_dict[ticker] = {'info': {}, 'company_name': ticker}
    st.session_state['cached_info'][watchlist_key] = info_dict

# Use cached data for all slider changes
data = st.session_state['cached_data'].get(watchlist_key, None)
info_dict = st.session_state['cached_info'].get(watchlist_key, {})
company_names = {k: v['company_name'] for k, v in info_dict.items()}

# --- Continuous date slider ---
import datetime as dt
min_date = five_years_ago.date()
max_date = today.date()
default_date = (today - timedelta(days=180)).date()
start_date = st.slider(
    label='Select start date for the chart',
    min_value=min_date,
    max_value=max_date,
    value=default_date,
    format="YYYY-MM-DD",
    key='compact_slider',
    help='Select a start date for the chart.'
)


if tickers:
    tickers_list = tickers
    if tickers_list:
        # Only use the full cached data for table calculations; do not overwrite the full cache with the chart's range
        chart_data = yf.download(tickers_list, start=start_date, end=today, group_by='ticker', auto_adjust=True)
        # --- Chart 1: Performance (start=100%) ---
        fig = go.Figure()
        performance = {}
        all_norms = []
        all_x = []
        for ticker in tickers_list:
            try:
                if len(tickers_list) == 1:
                    prices = chart_data['Close']
                else:
                    prices = chart_data[ticker]['Close']
                # Normalize performance: start at 100
                if not prices.empty:
                    # Find first non-NaN value and its index
                    first_valid = prices.first_valid_index()
                    if first_valid:
                        # Only use data from first valid point onwards
                        valid_prices = prices[first_valid:]
                        norm = valid_prices / valid_prices.iloc[0] * 100
                        all_norms.append(norm)
                        all_x.append(norm.index)
                        tkey = ticker.upper()
                        # Fix for HK tickers: ensure .HK suffix
                        if tkey.endswith('.HK') and len(tkey) == 7 and tkey[:4].isdigit():
                            tkey = tkey.zfill(7)
                        abbr = company_names.get(tkey, ticker)
                    fig.add_trace(go.Scatter(
                        x=norm.index,
                        y=norm,
                        mode='lines',
                        name=ticker,
                        hovertemplate=f'%{{y:.0f}}%<br>Date: %{{x|%Y-%m-%d}}<br>Company: {abbr}<extra>' + ticker + '</extra>'
                    ))
                    perf = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100
                    performance[ticker] = perf
                else:
                    performance[ticker] = None
            except Exception as e:
                performance[ticker] = None
        # Custom y-axis ticks: round up to next integer percent
        import numpy as np
        if all_norms:
            y_min = min([norm.min() for norm in all_norms])
            y_max = max([norm.max() for norm in all_norms])
            y_min = np.floor(y_min)
            y_max = np.ceil(y_max)
        else:
            y_min = -100
            y_max = 100
        if y_max < 100 or np.isnan(y_max):
            y_max = 100
        if y_min > 0 or np.isnan(y_min):
            y_min = 0
        # Use log scale, so ticks at 100, 200, 300, ... up to y_max
        tickvals = list(np.arange(100, y_max+1, 100))
        if y_min < 0:
            tickvals = [y_min] + tickvals
        ticktext = [f"{int(np.ceil(val))}%" for val in tickvals]
        # Add horizontal lines
        if all_x:
            x0 = min([x.min() for x in all_x])
            x1 = max([x.max() for x in all_x])
        else:
            x0 = x1 = None
        for y in tickvals:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(color="lightgray", width=1, dash="dot"), xref="x", yref="y", layer="below")
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
                tkey = ticker.upper()
                if tkey.endswith('.HK') and len(tkey) == 7 and tkey[:4].isdigit():
                    tkey = tkey.zfill(7)
                abbr = company_names.get(tkey, ticker)
                if len(tickers_list) == 1:
                    prices = chart_data['Close']
                    if not prices.empty:
                        fig2.add_trace(go.Scatter(
                            x=prices.index,
                            y=prices,
                            mode='lines',
                            name=ticker,
                            hovertemplate=f'%{{y:.2f}}<br>Date: %{{x|%Y-%m-%d}}<br>Company: {abbr}<extra>' + ticker + '</extra>'
                        ))
                else:
                    prices = chart_data[ticker]['Close']
                    if not prices.empty:
                        # Find first non-NaN value
                        first_valid = prices.first_valid_index()
                        if first_valid:
                            # Only use data from first valid point onwards
                            valid_prices = prices[first_valid:]
                            min_p = valid_prices.min()
                            max_p = valid_prices.max()
                            if max_p > min_p:
                                norm_mm = (valid_prices - min_p) / (max_p - min_p)
                                fig2.add_trace(go.Scatter(
                                x=norm_mm.index,
                                y=norm_mm,
                                mode='lines',
                                name=ticker,
                                hovertemplate=f'%{{y:.2f}}<br>Date: %{{x|%Y-%m-%d}}<br>Company: {abbr}<extra>' + ticker + '</extra>'
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

        # --- Dividend Yield Table ---
        import pandas as pd
        dividend_data = []
        for ticker in tickers_list:
            tkey = ticker.upper()
            if tkey.endswith('.HK') and len(tkey) == 7 and tkey[:4].isdigit():
                tkey = tkey.zfill(7)
            abbr = company_names.get(tkey, ticker)
            info = info_dict.get(ticker, {}).get('info', {})
            yield_val = info.get('dividendYield', None)  # already a fraction
            pe_ratio = info.get('trailingPE', None)
            price_to_book = info.get('priceToBook', None)
            profit_margin = info.get('profitMargins', None)
            current_price = info.get('currentPrice', None)
            # Get last close price from cached data if possible
            last_close = None
            price_1y = None
            price_2y = None
            price_5y = None
            try:
                # Always use the full 5-year cached data for yearly changes
                full_data = st.session_state['cached_data'][watchlist_key]
                if ticker in full_data:
                    full_prices = full_data[ticker]['Close']
                else:
                    full_prices = full_data['Close']
                if not full_prices.empty:
                    if not isinstance(full_prices.index, pd.DatetimeIndex):
                        full_prices.index = pd.to_datetime(full_prices.index)
                    full_prices = full_prices.sort_index()
                    def get_past_price(prices, years):
                        target_date = today - timedelta(days=365*years)
                        target_date = pd.Timestamp(target_date.date())
                        val = prices.asof(target_date)
                        if pd.isna(val):
                            return prices.iloc[0] if len(prices) > 0 else None
                        return val
                    price_1y = get_past_price(full_prices, 1)
                    price_2y = get_past_price(full_prices, 2)
                    price_5y = get_past_price(full_prices, 5)
                # For last_close, use the current chart data (start_date to today)
                if ticker in chart_data:
                    prices = chart_data[ticker]['Close']
                else:
                    prices = chart_data['Close']
                if not prices.empty:
                    if not isinstance(prices.index, pd.DatetimeIndex):
                        prices.index = pd.to_datetime(prices.index)
                    last_close = prices.iloc[-2] if len(prices) > 1 else prices.iloc[-1]
            except Exception:
                last_close = None
                price_1y = None
                price_2y = None
                price_5y = None
            except Exception:
                last_close = None
                price_1y = None
                price_2y = None
                price_5y = None
            price_change = None
            price_change_1y = None
            price_change_2y = None
            price_change_5y = None
            if current_price is not None and last_close is not None and last_close != 0:
                price_change = (current_price / last_close) - 1
            if current_price is not None and price_1y is not None and price_1y != 0:
                price_change_1y = (current_price / price_1y) - 1
            if current_price is not None and price_2y is not None and price_2y != 0:
                price_change_2y = (current_price / price_2y) - 1
            if current_price is not None and price_5y is not None and price_5y != 0:
                price_change_5y = (current_price / price_5y) - 1
            dividend_data.append({
                'Ticker': ticker,
                'Company': abbr,
                'Dividend Yield': yield_val,
                'P/E Ratio': pe_ratio,
                'Price to Book': price_to_book,
                'Profit Margin (%)': profit_margin * 100 if profit_margin is not None else None,
                'Price Change (%)': price_change * 100 if price_change is not None else None,
                '1Y Change (%)': price_change_1y * 100 if price_change_1y is not None else None,
                '2Y Change (%)': price_change_2y * 100 if price_change_2y is not None else None,
                '5Y Change (%)': price_change_5y * 100 if price_change_5y is not None else None
            })
        df_div = pd.DataFrame(dividend_data)
        # Replace inf/-inf with 10000, round all numbers to 1 decimal, ensure numeric columns are float
        num_cols = ['Dividend Yield', 'P/E Ratio', 'Price to Book', 'Profit Margin (%)', 'Price Change (%)', '1Y Change (%)', '2Y Change (%)', '5Y Change (%)']
        for col in num_cols:
            if col in df_div.columns:
                df_div[col] = pd.to_numeric(df_div[col], errors='coerce')
                df_div[col] = df_div[col].replace([float('inf'), float('-inf')], 10000)
                df_div[col] = df_div[col].round(1)
        st.subheader('Dividend, P/E, and Price Change Table')
        st.dataframe(df_div, use_container_width=True, height=20*35)


