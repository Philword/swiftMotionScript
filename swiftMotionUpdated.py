import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Constants
YEARS = [2025, 2026, 2027, 2028, 2029]
QUARTERS = [f"{y} Q{q+1}" for y in YEARS for q in range(4)]
BASE_VALUES = {
    'Core Business': 250,    # In millions
    'Startup Portfolio': 600,
    'BTC Treasury': 200,
    'STO Proceeds': 150
}

SCENARIO_MULTIPLIERS = {
    'Bear': 0.5,
    'Base': 1.0,
    'Bull': 1.5
}

# Streamlit App Configuration
st.set_page_config(layout="wide", page_icon="")
st.title("SwiftMotion")
st.markdown("""
**Strategic Digital Asset Growth Simulator**
*Model portfolio performance across market cycles with dynamic scenario analysis*
""")

# ==============================================================================
# Simulation Engine
# ==============================================================================
class InvestmentSimulator:
    def __init__(self, investment):
        self.initial_investment = investment
        self.total_supply = 1_000_000_000  # Adjusted for realistic token prices
        self.base_valuation = sum(BASE_VALUES.values()) * 1e6  # In dollars

    def _generate_btc_scenario(self, scenario):
        """Dynamic BTC pricing based on scenario multipliers"""
        base_prices = [200000 * (1.15**i) for i in range(5)]  # 15% base growth
        return [p * SCENARIO_MULTIPLIERS[scenario] * 1.3 for p in base_prices]

    def calculate_valuation(self, params):
        """Dynamic valuation model with growth compounding"""
        df = pd.DataFrame({'Year': YEARS})

        # Component growth calculations
        components = {
            'Core Business': (BASE_VALUES['Core Business'], params['core']),
            'Startup Portfolio': (BASE_VALUES['Startup Portfolio'], params['startup']),
            'BTC Treasury': (BASE_VALUES['BTC Treasury'], params['btc']),
            'STO Proceeds': (BASE_VALUES['STO Proceeds'], params['sto'])
        }

        for name, (base, growth) in components.items():
            df[name] = [base * (1 + growth/100)**i for i in range(len(YEARS))]

        # BTC Value calculation with leverage
        df['BTC Value'] = self._generate_btc_scenario(params['scenario'])

        # Total Valuation in USD
        df['Total Valuation ($M)'] = df.iloc[:, 1:-1].sum(axis=1) / 1e6

        # Token price calculations
        df['Token Price'] = (df.iloc[:, 1:-2].sum(axis=1) * 1e6) / self.total_supply
        quarterly_prices = np.interp(np.linspace(0, 4, 20), np.arange(5), df['Token Price'])

        self.valuation_df = pd.DataFrame({
            'Period': QUARTERS,
            'Token Price': quarterly_prices
        })

        return df

    def calculate_returns(self, params):
        """Enhanced ROI calculation with dividend compounding"""
        initial_price = self.valuation_df['Token Price'].iloc[0]
        tokens = self.initial_investment / initial_price

        returns = []
        cumulative_dividends = 0
        for idx, row in self.valuation_df.iterrows():
            market_cap = row['Token Price'] * self.total_supply
            dividend = (market_cap * (params['margin']/100) * 0.30 * (tokens/self.total_supply))
            cumulative_dividends += dividend
            token_value = tokens * row['Token Price']

            returns.append({
                'Period': row['Period'],
                'Token Value': token_value,
                'Dividends': dividend,
                'Total Value': token_value + cumulative_dividends,
                'Token Price': row['Token Price']
            })

        return pd.DataFrame(returns)

# ==============================================================================
# Interface Components
# ==============================================================================
def create_sidebar():
    """Interactive controls with initial hidden state"""
    with st.sidebar:

        # Add above existing controls
        st.subheader("Simulation Period")
        selected_years = st.slider("Projection Years", 1, 5, 5,
                                 help="Adjust simulation duration in years")

        st.header("⚙️ Simulation Controls")
        enable = st.checkbox("Enable Advanced Parameters", False,
                          help="Reveal growth parameter controls")

        st.subheader("Investment Parameters")
        investment = st.number_input("Initial Investment (USD)", 1000, 10_000_000, 500000, 1000,
                                   format="%d")

        # Initialize default parameters
        DEFAULTS = {
            'core': 20,  # Original values from your initial code
            'startup': 20,
            'btc': 25,
            'sto': 0,
            'margin': 25,
            'scenario': "Base"
        }

        # Only show sliders when enabled
        if enable:
            st.subheader("Growth Parameters (%)")
            params = {
                'core': st.slider("Core Business Growth", -10, 50, DEFAULTS['core']),
                'startup': st.slider("Startup Portfolio Growth", -50, 50, DEFAULTS['startup']),
                'btc': st.slider("BTC Treasury Growth", -10, 50, DEFAULTS['btc']),
                'sto': st.slider("STO Proceeds Growth", -10, 50, DEFAULTS['sto']),
                'margin': st.slider("Profit Margin", 5, 50, DEFAULTS['margin']),
                'scenario': st.selectbox("Market Scenario", ["Bull", "Base", "Bear"], index=1)
            }
        else:
            params = DEFAULTS.copy()
            # Create hidden dummy elements to preserve layout
            st.empty()
            st.empty()
            st.empty()
            st.empty()
            st.empty()
            st.empty()

        st.download_button("📊 Export Simulation Data",
                         data=pd.DataFrame().to_csv(),
                         file_name="smt_simulation.csv")
    return investment, params, selected_years

def create_kpis(returns_df, investment):
    """Strategic investor KPIs"""
    current_value = filtered_returns['Total Value'].iloc[-1]
    total_dividends = filtered_returns['Dividends'].sum()
    token_growth = ((filtered_returns['Token Price'].iloc[-1] -
                   filtered_returns['Token Price'].iloc[0]) /
                  filtered_returns['Token Price'].iloc[0]) * 100

    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric("Portfolio Value", f"${current_value:,.0f}",
                f"{(current_value/investment-1)*100:.1f}% Total Return")
    with kpi2:
        st.metric("Dividend Income", f"${total_dividends:,.0f}",
                f"{(total_dividends/investment)*100:.1f}% Yield")
    with kpi3:
        st.metric("Token Appreciation", f"${returns_df['Token Price'].iloc[-1]:.2f}",
                f"{token_growth:.1f}% Growth")

# ==============================================================================
# Main Execution
# ==============================================================================
investment, params, years = create_sidebar()
simulator = InvestmentSimulator(investment)
valuation_df = simulator.calculate_valuation(params)
returns_df = simulator.calculate_returns(params)


end_year = 2025 + years - 1
filtered_valuation = valuation_df[valuation_df['Year'] <= end_year]
filtered_returns = returns_df[returns_df['Period'].apply(
    lambda x: int(x[:4]) <= end_year
)]

# Hero Section
st.header("Investment Overview")
create_kpis(filtered_returns, investment)

# Value Proposition
st.subheader("Key Investment Highlights")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("📈 **Growth Potential**")
    st.progress(0.85)
    st.caption("85% Historical CAGR in Core Business")
with col2:
    st.markdown("🛡️ **Risk Mitigation**")
    st.progress(0.65)
    st.caption("65% Assets in Stable Cash-Flow Businesses")
with col3:
    st.markdown("🌐 **Market Leadership**")
    st.progress(0.95)
    st.caption("95% Market Share in Target Sectors")

# Main Charts
tab1, tab2, tab3 = st.tabs(["Performance Analysis", "Valuation Breakdown", "Income Statement"])

with tab1:
    fig = px.area(filtered_returns, x='Period', y='Total Value',  # Updated
                 title="Portfolio Value Development")
    fig.add_scatter(x=filtered_returns['Period'], y=filtered_returns['Token Value'],
                  mode='lines', name='Token Value')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    melted = filtered_valuation.melt(id_vars='Year',  # Updated
                             value_vars=['Core Business', 'Startup Portfolio', 'BTC Treasury', 'STO Proceeds'],
                             var_name='Component', value_name='Value')
    fig = px.bar(melted, x='Year', y='Value', color='Component', barmode='stack',
                title="Asset Allocation Breakdown")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.bar(returns_df, x='Period', y='Dividends',
                title="Dividend Distribution Timeline")
    st.plotly_chart(fig, use_container_width=True)

# Risk Disclosure
st.markdown("---")
st.caption("""
**Disclaimer:** This simulation represents hypothetical scenarios based on historical performance and forward-looking assumptions.
Actual results may vary significantly. Past performance is not indicative of future returns.
Consult with a qualified financial advisor before making investment decisions.
""")
