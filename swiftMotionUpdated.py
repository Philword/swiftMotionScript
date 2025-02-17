import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# Constants
TOTAL_TOKENS = 100_000_0  # 100M security tokens
BASE_GROWTH = {
    'Core Business': [2.5, 3.1, 3.8, 4.6, 5.4],
    'Startup Portfolio': [6.0, 7.2, 8.64, 10.37, 12.44],
    'BTC Treasury': [2.0, 4.0, 6.5, 10.0, 15.0],
    'Utility Token MCap': [10.0, 12.4, 15.2, 18.48, 22.16],
    'STO Proceeds': [1.5, 2.5, 3.0, 2.5, 0.5]
}

# Streamlit App Configuration
st.set_page_config(layout="wide")
st.title("SwiftMotion Advanced STO Simulator")

# ==============================================================================
# Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.header("💰 Investment Parameters")
    investment = st.number_input("Investment Amount ($)", 1000, 10_000_000, 100_000, 5000)
    tokens_to_buy = st.slider("Tokens to Purchase", 1000, 1_000_000, 10_000, 1000)
    token_maturity = st.selectbox("Token Maturity", ["Short-term (5%)", "Medium-term (10%)", "Long-term (15%)"])

    st.header("🏢 Company Parameters")
    scenario = st.selectbox("Market Scenario", ["Bull", "Base", "Bear"], index=1)
    profit_margin = st.slider("Profit Margin (%)", 10, 50, 20)
    leverage_strategy = st.selectbox("Leverage Strategy", ["No Leverage", "2x Leverage"])
    btc_growth = st.slider("BTC Growth Boost (%)", -50, 200, 0)

    st.header("📈 Growth Adjustments")
    core_growth = st.slider("Core Business Growth (%)", -20, 100, 0)
    startup_growth = st.slider("Startup Portfolio Growth (%)", -20, 100, 0)
    token_growth = st.slider("Token Appreciation (%)", 0, 300, 0)

    st.header("🪙 Token Parameters")
    initial_token_price = st.number_input("Initial Token Price ($)", 0.01, 100.0, 1.0, 0.01)

    st.header("💸 Profit Allocation")
    profit_to_tokens = st.slider("Profit to Token Holders (%)", 10, 50, 30)
    profit_to_reserves = st.slider("Profit to Reserves (%)", 5, 20, 12)

    utility_token_bonus = st.checkbox("Include Utility Token Bonus (5% ROI)")

# ==============================================================================
# Simulation Engine
# ==============================================================================
class CompanySimulator:
    def __init__(self):
        self.years = [2025, 2026, 2027, 2028, 2029]
        self.profit_shares = {
            'Short-term (5%)': 0.05,
            'Medium-term (10%)': 0.10,
            'Long-term (15%)': 0.15
        }
        self.token_allocations = {
            'Short-term (5%)': 0.3,
            'Medium-term (10%)': 0.5,
            'Long-term (15%)': 0.2
        }

    def calculate_valuation(self, leverage):
        df = pd.DataFrame({'Year': self.years})

        # Apply growth adjustments
        df['Core Business'] = self.apply_growth(BASE_GROWTH['Core Business'], core_growth/100)
        df['Startup Portfolio'] = self.apply_growth(BASE_GROWTH['Startup Portfolio'], startup_growth/100)
        df['BTC Treasury'] = self.apply_growth(BASE_GROWTH['BTC Treasury'], btc_growth/100)
        df['Utility Token MCap'] = self.apply_growth(BASE_GROWTH['Utility Token MCap'], token_growth/100)
        df['STO Proceeds'] = BASE_GROWTH['STO Proceeds']

        # BTC Price scenarios
        btc_prices = {
            'Bull': [280780, 420021, 610305, 800000, 1000000],
            'Base': [200000, 280000, 350000, 400000, 450000],
            'Bear': [120000, 150000, 180000, 200000, 220000]
        }
        df['BTC Price'] = np.array(btc_prices[scenario]) * (1 + btc_growth/100)

        # Calculate leveraged BTC value
        leverage_factor = 2.0 if leverage_strategy == "2x Leverage" else 1.0
        df['BTC Value'] = (df['BTC Treasury'] * 1e6 / df['BTC Price']) * df['BTC Price'] * leverage_factor / 1e6

        # Total company valuation
        df['Total Valuation'] = (df['Core Business'] + df['Startup Portfolio'] +
                                df['Utility Token MCap'] + df['STO Proceeds'] + df['BTC Value'])

        return df

    def apply_growth(self, base_values, growth_rate):
        return [v * (1 + growth_rate) ** i for i, v in enumerate(base_values)]

    def calculate_returns(self, df, investor_tokens, token_maturity, profit_to_tokens, utility_token_bonus):
        results = []
        total_valuation = df['Total Valuation'].values
        token_mcaps = df['Utility Token MCap'].values

        profit_share = self.profit_shares.get(token_maturity, 0.10)
        token_allocation = self.token_allocations.get(token_maturity, 0.5)

        for i, year in enumerate(self.years):
            profit = total_valuation[i] * (profit_margin / 100) * 1e6
            token_profit = profit * (profit_to_tokens / 100)
            class_profit = token_profit * profit_share

            total_class_tokens = TOTAL_TOKENS * token_allocation
            investor_share = investor_tokens / total_class_tokens
            dividend = class_profit * investor_share

            # Quarterly Calculations
            for quarter in range(1, 5):
                quarterly_dividend = dividend / 4
                token_price = (token_mcaps[i] * 1e6) / TOTAL_TOKENS

                if quarter > 1 and i > 0:
                    token_price *= (1 + np.random.uniform(0.01, 0.03))

                total_value = (token_price * investor_tokens) + (quarterly_dividend * 4)
                roi = (total_value / investment - 1) * 100

                if utility_token_bonus:
                    roi *= 1.05

                results.append({
                    'Year': year,
                    'Quarter': quarter,
                    'Token Price': token_price,
                    'Dividend per Token': quarterly_dividend / investor_tokens if investor_tokens > 0 else 0,
                    'Total Dividend': quarterly_dividend,
                    'Company Valuation': total_valuation[i],
                    'ROI': roi
                })

        return pd.DataFrame(results)

# ==============================================================================
# Simulation Execution
# ==============================================================================
simulator = CompanySimulator()
leverage = 2.0 if leverage_strategy == "2x Leverage" else 1.0
valuation_df = simulator.calculate_valuation(leverage)
returns_df = simulator.calculate_returns(
    valuation_df,
    tokens_to_buy,
    token_maturity,
    profit_to_tokens,
    utility_token_bonus
)

# Post-process results
returns_df['Token Value'] = tokens_to_buy * returns_df['Token Price']
returns_df['Cumulative Dividends'] = returns_df.groupby(['Year'])['Total Dividend'].cumsum()
returns_df['Total Value'] = returns_df['Token Value'] + returns_df['Cumulative Dividends']
returns_df['Annualized ROI'] = ((returns_df['Total Value'] / investment) ** (1/4) - 1) * 100

# ==============================================================================
# Results Visualization
# ==============================================================================
st.header("🚀 Projected Investment Performance")

# Key Metrics
final_roi = returns_df.iloc[-1]['ROI']
annualized_roi = returns_df.iloc[-1]['Annualized ROI']

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final ROI", f"{final_roi:.1f}%")
col2.metric("Annualized Return", f"{annualized_roi:.1f}%")
col3.metric("2029 Token Value", f"${returns_df.iloc[-1]['Token Value']:,.0f}")
col4.metric("Total Dividends", f"${returns_df.iloc[-1]['Cumulative Dividends']:,.0f}")

# Main Chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=returns_df['Year'],
    y=returns_df['Total Value'],
    name="Total Value",
    line=dict(color='#00CC96', width=4)
))
fig.add_trace(go.Bar(
    x=returns_df['Year'],
    y=returns_df['Total Dividend'],
    name="Annual Dividends",
    marker_color='#EF553B'
))
fig.update_layout(
    title="Investment Growth Projection",
    xaxis_title="Year",
    yaxis_title="Value ($)",
    hovermode="x unified",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# Tokenomics Section
st.header("🔍 Detailed Tokenomics Analysis")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Token Price Appreciation")
    fig = px.line(returns_df, x='Year', y='Token Price',
                 labels={'Token Price': 'Price ($)'},
                 markers=True)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Company Valuation Growth")
    fig = px.area(valuation_df, x='Year', y='Total Valuation',
                 labels={'Total Valuation': 'Valuation ($M)'})
    st.plotly_chart(fig, use_container_width=True)



### Data Export
#csv = returns_df.to_csv().encode('utf-8')
#st.download_button(
#    "📥 Download Full Report",
#    data=csv,
#    file_name="swiftmotion_investment_report.csv",
#    mime="text/csv"
#)
