import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Base Growth Config (in millions)
BASE_GROWTH = {
    'Core Business': [2.5, 3.1, 3.8, 4.6, 5.4],
    'Startup Portfolio': [6.0, 7.2, 8.64, 10.37, 12.44],
    'BTC Treasury': [2.0, 4.0, 6.5, 10.0, 15.0],
    'Utility Token MCap': [10.0, 12.4, 15.2, 18.48, 22.16],
    'STO Proceeds': [1.5, 2.5, 3.0, 2.5, 0.5]
}

# Streamlit App Configuration
st.set_page_config(layout="wide")
st.title("🚀 SwiftMotion Pro Investment Simulator")

# ==============================================================================
# Sidebar Controls
# ==============================================================================
with st.sidebar:
    st.header("💰 Core Investment")
    initial_investment = st.number_input("Initial Investment ($)", 1000, 10_000_000, 100000, 1000)

    st.header("⚙️ Growth (Annual %)")
    core_business_growth = st.slider("Core Business Growth", -10, 50, 20)  # Negative allowed
    startup_portfolio_growth = st.slider("Startup Portfolio Growth", -50, 50, 20) # Negative allowed
    btc_treasury_growth = st.slider("BTC Treasury Growth", -10, 50, 25)  # Negative allowed
    sto_proceeds_growth = st.slider("STO Proceeds Growth", -10, 50, 0)  # Negative allowed

    profit_margin = st.slider("Profit Margin (%)", 5, 50, 25)
    scenario = st.selectbox("Market Scenario (BTC)", ["Bull", "Base", "Bear"], index=1)

    st.header("📊 Display Options")
    show_valuation = st.checkbox("Show Valuation Breakdown", True)
    show_dividends = st.checkbox("Show Dividend Analysis", True)


# ==============================================================================
# Simulation Engine
# ==============================================================================
class InvestmentSimulator:
    def __init__(self, investment):
        self.years = [2025, 2026, 2027, 2028, 2029]
        self.total_supply = 100_000_000_000  # Fixed token supply
        self.initial_investment = investment
        self.base_market_cap = investment * 10  # Initial valuation

    def calculate_valuation(self):
        df = pd.DataFrame({'Year': self.years})

        # Portfolio Growth Calculations (using sliders)
        df['Core Business'] = [2.5 * (1 + core_business_growth/100)**i for i in range(len(self.years))]
        df['Startup Portfolio'] = [6.0 * (1 + startup_portfolio_growth/100)**i for i in range(len(self.years))]
        df['BTC Treasury'] = [2.0 * (1 + btc_treasury_growth/100)**i for i in range(len(self.years))]
        df['STO Proceeds'] = [1.5 * (1 + sto_proceeds_growth/100)**i for i in range(len(self.years))] # Apply growth to STO


        # BTC calculations (same as before)
        btc_prices = {
            'Bull': [280780, 420021, 610305, 800000, 1000000],
            'Base': [200000, 280000, 350000, 400000, 450000],
            'Bear': [120000, 150000, 180000, 200000, 220000]
        }
        df['BTC Value'] = [btc * 1.5 for btc in btc_prices[scenario]]  # BTC leverage

        # Total Valuation
        df['Total Valuation'] = (
            df['Core Business'] + df['Startup Portfolio'] + df['BTC Value'] + df['STO Proceeds']
        )

        # Token price calculations
        df['Token Price'] = df['Total Valuation'] * 1e6 / self.total_supply

        # Generate quarterly prices with smooth interpolation
        quarterly_prices = []
        for i in range(len(df)-1):
            for q in range(4):
                quarterly_prices.append(
                    df['Token Price'].iloc[i] +
                    (df['Token Price'].iloc[i+1] - df['Token Price'].iloc[i]) * (q+1)/4
                )
        quarterly_prices.append(df['Token Price'].iloc[-1])  # Last year

        # Create quarterly DataFrame
        quarters = [f"{y} Q{q+1}" for y in self.years for q in range(4)][:len(quarterly_prices)]
        self.valuation_df = pd.DataFrame({
            'Period': quarters,
            'Token Price': quarterly_prices
        })

        return df

    def calculate_returns(self, investment):
        # Calculate initial token allocation
        initial_price = self.valuation_df['Token Price'].iloc[0]
        tokens = investment / initial_price

        # Generate returns data
        returns = []
        total_dividends = 0
        for idx, row in self.valuation_df.iterrows():
            # Profit calculation (5% of token market cap)
            profit = (row['Token Price'] * self.total_supply) * (profit_margin/100)
            dividend = (profit * 0.30) * (tokens/self.total_supply)  # 30% profit sharing

            total_dividends += dividend
            token_value = tokens * row['Token Price']

            returns.append({
                'Period': row['Period'],
                'Token Value': token_value,
                'Dividends': dividend,
                'Total Value': token_value + total_dividends,
                'Token Price': row['Token Price']
            })

        return pd.DataFrame(returns)

# ==============================================================================
# Simulation Execution
# ==============================================================================
simulator = InvestmentSimulator(initial_investment)
valuation_df = simulator.calculate_valuation()
returns_df = simulator.calculate_returns(initial_investment)

# ==============================================================================
# Visualization
# ==============================================================================
st.header("📈 Investment Performance Overview")

# Main metrics
current_value = returns_df['Total Value'].iloc[-1]
total_dividends = returns_df['Dividends'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Final Value", f"${current_value:,.0f}",
           f"{((current_value/initial_investment)-1)*100:.1f}% ROI")
col2.metric("Total Dividends", f"${total_dividends:,.0f}",
           f"Annual Yield: {total_dividends/initial_investment/4*100:.1f}%")
col3.metric("Final Token Price", f"${returns_df['Token Price'].iloc[-1]:.2f}",
           f"{((returns_df['Token Price'].iloc[-1]/returns_df['Token Price'].iloc[0])-1)*100:.1f}% Growth")

# Main chart
fig = px.area(returns_df, x='Period', y='Total Value',
             title="Total Investment Value Development")
fig.add_scatter(x=returns_df['Period'], y=returns_df['Token Value'],
               mode='lines', name='Token Value')
st.plotly_chart(fig, use_container_width=True)

# Valuation Analysis
if show_valuation:
    st.subheader("🏢 Company Valuation Breakdown")
    valuation_melted = valuation_df.melt(
        id_vars='Year',
        value_vars=['Core Business', 'Startup Portfolio', 'BTC Value', 'STO Proceeds'],  # Include all relevant columns
        var_name='Asset',
        value_name='Value'
    )

    fig = px.bar(valuation_melted, x='Year', y='Value', color='Asset', title="Business Unit Valuation")
    st.plotly_chart(fig, use_container_width=True)

# Dividend Analysis
if show_dividends:
    st.subheader("💵 Dividend Cash Flow")
    fig = px.bar(returns_df, x='Period', y='Dividends',
                title="Quarterly Dividend Payments")
    st.plotly_chart(fig, use_container_width=True)

# Token Price Development
st.subheader("📈 Token Price Growth")
fig = px.line(returns_df, x='Period', y='Token Price',
             markers=True, title="Token Price Development")
st.plotly_chart(fig, use_container_width=True)

# Data Export
st.sidebar.download_button(
    "📥 Export Simulation Data",
    data=returns_df.to_csv().encode('utf-8'),
    file_name="investment_simulation.csv",
    mime="text/csv"
)
