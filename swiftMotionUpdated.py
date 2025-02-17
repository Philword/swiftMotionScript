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
    tokens_to_buy = st.slider("Tokens to Purchase", 1000, 1_000_000_000, 10_000, 1000)
    token_type = st.selectbox("Token Class",
                            ["Short-term (5%)", "Medium-term (10%)", "Long-term (15%)"],
                            index=2)

    st.header("🏢 Company Parameters")
    scenario = st.selectbox("Market Scenario", ["Bull", "Base", "Bear"], index=1)
    profit_margin = st.slider("Profit Margin (%)", 10, 50, 20)
    leverage = st.slider("BTC Leverage", 1.0, 5.0, 2.0) if scenario == "Bull" else 1.0
    btc_growth = st.slider("BTC Growth Boost (%)", -50, 200, 0)

    st.header("📈 Growth Adjustments")
    core_growth = st.slider("Core Business Growth (%)", -20, 100, 0)
    startup_growth = st.slider("Startup Portfolio Growth (%)", -20, 100, 0)
    token_growth = st.slider("Token Appreciation (%)", 0, 300, 0)

# ==============================================================================
# Simulation Engine
# ==============================================================================
class CompanySimulator:
    def __init__(self):
        self.years = [2025, 2026, 2027, 2028, 2029]
        self.profit_shares = {'Short-term (5%)': 0.05, 'Medium-term (10%)': 0.10, 'Long-term (15%)': 0.15}
        self.token_allocations = {'Short-term (5%)': 0.3, 'Medium-term (10%)': 0.5, 'Long-term (15%)': 0.2}

    def calculate_valuation(self):
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
        df['BTC Value'] = (df['BTC Treasury'] * 1e6 / df['BTC Price']) * df['BTC Price'] * leverage / 1e6

        # Total company valuation
        df['Total Valuation'] = (df['Core Business'] + df['Startup Portfolio'] +
                                df['Utility Token MCap'] + df['STO Proceeds'] + df['BTC Value'])

        return df

    def apply_growth(self, base_values, growth_rate):
        return [v * (1 + growth_rate) ** i for i, v in enumerate(base_values)]

    def calculate_returns(self, df, investor_tokens):
        results = []
        total_valuation = df['Total Valuation'].values
        token_mcaps = df['Utility Token MCap'].values

        for i, year in enumerate(self.years):
            profit = total_valuation[i] * (profit_margin / 100) * 1e6
            token_profit = profit * 0.30
            class_profit = token_profit * self.profit_shares[token_type]

            total_class_tokens = TOTAL_TOKENS * self.token_allocations[token_type]
            investor_share = investor_tokens / total_class_tokens
            dividend = class_profit * investor_share

            token_price = (token_mcaps[i] * 1e6) / TOTAL_TOKENS

            # Quarterly Calculations
            quarterly_results = []
            for quarter in range(1, 5):  # 4 quarters per year
                quarterly_dividend = dividend / 4  # Distribute dividends evenly
                quarterly_results.append({
                    'Year': year,
                    'Quarter': quarter,
                    'Token Price': token_price,
                    'Dividend per Token': quarterly_dividend / investor_tokens if investor_tokens > 0 else 0,
                    'Total Dividend': quarterly_dividend,
                    'Company Valuation': total_valuation[i]
                })
            results.extend(quarterly_results) # Add the quarterly results to the main results

        return pd.DataFrame(results)

# ==============================================================================
# Simulation Execution
# ==============================================================================
simulator = CompanySimulator()
valuation_df = simulator.calculate_valuation()
returns_df = simulator.calculate_returns(valuation_df, tokens_to_buy)

# Calculate investor returns
returns_df['Token Value'] = tokens_to_buy * returns_df['Token Price']
returns_df['Cumulative Dividends'] = returns_df.groupby(['Year'])['Total Dividend'].cumsum() # cumulative dividends by year
returns_df['Total Value'] = returns_df['Token Value'] + returns_df['Cumulative Dividends']
returns_df['ROI'] = (returns_df['Total Value'] / investment - 1) * 100
returns_df['Annualized ROI'] = ((returns_df.groupby('Year')['Total Value'].last() / investment) ** (1/4) - 1) * 100 * 100 # Annualized ROI by year

# ==============================================================================
# Results Visualization
# ==============================================================================
st.header("🚀 Projected Investment Performance")

# Key Metrics
current_year = 2025
final_roi = returns_df.iloc[-1]['ROI']
annualized_roi = ((returns_df.iloc[-1]['Total Value'] / investment) ** (1/4) - 1) * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Final ROI", f"{final_roi:.1f}%")
col2.metric("Annualized Return", f"{annualized_roi:.1f}%")
col3.metric("2029 Token Value", f"${returns_df.iloc[-1]['Token Value']:,.0f}")
col4.metric("Total Dividends", f"${returns_df.iloc[-1]['Cumulative Dividends']:,.0f}")

# Main Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=returns_df['Year'], y=returns_df['Total Value'],
              name="Total Value", line=dict(color='#00CC96', width=4)))
fig.add_trace(go.Bar(x=returns_df['Year'], y=returns_df['Total Dividend'],
              name="Annual Dividends", marker_color='#EF553B'))
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

# Detailed Data
st.subheader("📊 Annual Performance Breakdown")
st.dataframe(
    returns_df.style.format({
        'Token Price': "${:.2f}",
        'Dividend per Token': "${:.4f}",
        'Total Dividend': "${:,.0f}",
        'Token Value': "${:,.0f}",
        'Cumulative Dividends': "${:,.0f}",
        'Total Value': "${:,.0f}",
        'ROI': "{:.1f}%"
    })
)

# Scenario Analysis
st.header("🌐 Comparative Scenario Analysis")
scenarios = ['Bull', 'Base', 'Bear']
scenario_returns = []

for scenario in scenarios:
    temp_sim = CompanySimulator()
    temp_df = temp_sim.calculate_valuation()
    temp_returns = temp_sim.calculate_returns(temp_df, tokens_to_buy)

    # ***KEY CHANGE: Calculate Total Value here***
    temp_returns['Token Value'] = tokens_to_buy * temp_returns['Token Price']
    temp_returns['Cumulative Dividends'] = temp_returns['Total Dividend'].cumsum()
    temp_returns['Total Value'] = temp_returns['Token Value'] + temp_returns['Cumulative Dividends']  # This was missing!


    scenario_returns.append(temp_returns.iloc[-1]['Total Value'])

fig = px.bar(x=scenarios, y=scenario_returns,
             labels={'x': 'Scenario', 'y': 'Final Value'},
             color=scenarios,
             color_discrete_sequence=['#00CC96', '#636EFA', '#EF553B'])
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# Data Export
# ==============================================================================
# csv = returns_df.to_csv().encode('utf-8')
# st.download_button(
#     "📥 Download Full Report",
#     data=csv,
#     file_name="swiftmotion_investment_report.csv",
#     mime="text/csv"
# )
