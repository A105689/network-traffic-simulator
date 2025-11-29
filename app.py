"""
Network Traffic Simulation - Educational Version 2.0
Updates:
- Increased default Simulation Time (100 -> 1000) for better convergence.
- Added "Convergence" explainer in results.
- Fixed Poisson parameter visibility.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import json

from simulation_engine import (
    NetworkSimulator, 
    SimulationConfig, 
    DistributionType,
    compute_mmc_theoretical,
    compute_confidence_interval,
    run_replications,
    run_comparative_analysis,
    perform_chi_square_test
)

st.set_page_config(
    page_title="Network Traffic Simulator",
    page_icon="🎓",
    layout="wide"
)

st.title("Network Traffic Simulation 🎓")
st.markdown("### Discrete Event Simulation (Event-Scheduling Approach)")

# --- EDUCATIONAL: Concept Guide ---
with st.expander("📘 Concept Guide: Why do simulated results vary?"):
    st.markdown("""
    **1. Transient vs. Steady State:**
    * Simulations start with **0 packets** (Transient State).
    * Theoretical formulas assume the system has run forever (Steady State).
    * *Fix:* Run the simulation longer (e.g., Time > 1000) or use a **Warm-up Period**.

    **2. Stochastic Variation:**
    * M/M/1 systems are highly variable. Short runs (e.g., Time=100) are like rolling a die 5 times; the average might not be 3.5.
    * *Fix:* Use **Statistical Validation** mode to average 20+ runs.
    """)

st.markdown("---")

# --- Mode Selection ---
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Simulation", "Comparative Analysis", "Statistical Validation", "Input Analysis (Goodness of Fit)"],
    index=0
)

# --- Helper: Distribution Widget ---
def get_distribution_params(prefix: str, dist_type: DistributionType, default_rate: float = 5.0):
    params = {}
    if dist_type == DistributionType.EXPONENTIAL:
        params['rate'] = st.slider(f"{prefix} Rate (λ)", 0.1, 50.0, default_rate, 0.1, key=f"{prefix}_rate", help="Average number of events per time unit.")
        params['mean'] = 1/params['rate']
    elif dist_type == DistributionType.NORMAL:
        params['mean'] = st.slider(f"{prefix} Mean", 0.01, 2.0, 1/default_rate, 0.01, key=f"{prefix}_mean")
        params['std'] = st.slider(f"{prefix} Std Dev", 0.001, 0.5, 0.05, 0.001, key=f"{prefix}_std")
        params['rate'] = 1/params['mean']
    elif dist_type == DistributionType.UNIFORM:
        params['min'] = st.slider(f"{prefix} Min", 0.01, 1.0, 0.1, 0.01, key=f"{prefix}_min")
        params['max'] = st.slider(f"{prefix} Max", 0.02, 2.0, 0.3, 0.01, key=f"{prefix}_max")
        params['mean'] = (params['min'] + params['max']) / 2
        params['rate'] = 1/params['mean']
    elif dist_type == DistributionType.WEIBULL:
        params['shape'] = st.slider(f"{prefix} Shape (k)", 0.5, 5.0, 2.0, 0.1, key=f"{prefix}_shape")
        params['scale'] = st.slider(f"{prefix} Scale (λ)", 0.01, 2.0, 1/default_rate, 0.01, key=f"{prefix}_scale")
    elif dist_type == DistributionType.GAMMA:
        params['shape'] = st.slider(f"{prefix} Shape (k)", 0.5, 10.0, 2.0, 0.1, key=f"{prefix}_shape")
        params['scale'] = st.slider(f"{prefix} Scale (θ)", 0.01, 1.0, 1/(default_rate * 2), 0.01, key=f"{prefix}_scale")
    elif dist_type == DistributionType.LOGNORMAL:
        params['mean'] = st.slider(f"{prefix} μ (log-mean)", -2.0, 2.0, np.log(1/default_rate), 0.1, key=f"{prefix}_mean")
        params['std'] = st.slider(f"{prefix} σ (log-std)", 0.1, 2.0, 0.5, 0.1, key=f"{prefix}_std")
    elif dist_type == DistributionType.POISSON:
        params['mean'] = st.slider(f"{prefix} Lambda (λ)", 0.1, 50.0, default_rate, 0.1, key=f"{prefix}_lambda", help="Mean number of events occurring in the interval.")
        params['rate'] = params['mean']
    
    defaults = {'rate': default_rate, 'mean': 1/default_rate, 'std': 0.05, 'min': 0.1, 'max': 0.3, 'shape': 2.0, 'scale': 1/default_rate}
    for k, v in defaults.items():
        if k not in params: params[k] = v
    return params

# ==========================================
# MODE 1: SINGLE SIMULATION
# ==========================================
if mode == "Single Simulation":
    with st.sidebar:
        st.header("Simulation Parameters")
        
        with st.expander("RNG & Initialization", expanded=True):
            use_lcg = st.checkbox("Use Custom LCG", True, help="Use Linear Congruential Generator for course requirement")
            lcg_params = {'a': 16807, 'c': 0, 'm': 2147483647}
            if use_lcg:
                st.caption("LCG: $X_{i+1} = (aX_i + c) \mod m$")
                lcg_c1, lcg_c2, lcg_c3 = st.columns(3)
                lcg_params['a'] = lcg_c1.number_input("a", 1, value=16807)
                lcg_params['c'] = lcg_c2.number_input("c", 0, value=0)
                lcg_params['m'] = lcg_c3.number_input("m", 1, value=2147483647)
            
            warmup_time = st.number_input("Warm-up (T0)", 0.0, 500.0, 0.0, help="Initial period to discard to remove bias.")
            random_seed = st.number_input("Random Seed", 1, 99999, 42)
        
        st.subheader("System Configuration")
        num_servers = st.number_input("Servers (c)", 1, 50, 1)
        # UPDATED DEFAULT: 1000.0 instead of 100.0 for better convergence
        sim_time = st.number_input("Sim Time", 10.0, 10000.0, 1000.0, help="Longer time = Better convergence to theoretical values.")
        capacity = st.number_input("Queue Cap (0=Inf)", 0, 1000, 50)
        
        st.markdown("---")
        st.subheader("Arrival Process")
        arr_dist = st.selectbox("Distribution", [d.value for d in DistributionType], index=0, key="arr_dist")
        arr_params = get_distribution_params("Arrival", DistributionType(arr_dist), 5.0)
        
        st.markdown("---")
        st.subheader("Service Process")
        svc_dist = st.selectbox("Distribution", [d.value for d in DistributionType], index=0, key="svc_dist")
        svc_params = get_distribution_params("Service", DistributionType(svc_dist), 8.0)
        
        st.markdown("---")
        rho_est = arr_params['rate'] / (num_servers * svc_params['rate']) if svc_params['rate'] > 0 else 0
        st.metric("Estimated Traffic (ρ)", f"{rho_est:.3f}", delta_color="inverse" if rho_est >= 1 else "normal")
        if rho_est >= 1: st.warning("⚠️ System is unstable (ρ ≥ 1). Queue will grow infinitely.")
        
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        cfg = SimulationConfig(
            use_lcg=use_lcg,
            lcg_a=lcg_params['a'] if use_lcg else 16807,
            lcg_c=lcg_params['c'] if use_lcg else 0,
            lcg_m=lcg_params['m'] if use_lcg else 2147483647,
            warmup_time=warmup_time,
            random_seed=random_seed,
            num_servers=num_servers,
            simulation_time=sim_time,
            queue_capacity=capacity,
            arrival_distribution=DistributionType(arr_dist),
            arrival_rate=arr_params['rate'],
            arrival_mean=arr_params['mean'],
            arrival_std=arr_params['std'],
            arrival_min=arr_params['min'],
            arrival_max=arr_params['max'],
            arrival_shape=arr_params['shape'],
            arrival_scale=arr_params['scale'],
            service_distribution=DistributionType(svc_dist),
            service_rate=svc_params['rate'],
            service_mean=svc_params['mean'],
            service_std=svc_params['std'],
            service_min=svc_params['min'],
            service_max=svc_params['max'],
            service_shape=svc_params['shape'],
            service_scale=svc_params['scale']
        )
        
        with st.spinner("Simulating..."):
            sim = NetworkSimulator(cfg)
            stats = sim.run()
        
        # Dashboard
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Throughput", f"{stats['throughput']:.3f}/sec", help="Packets processed per second")
        m2.metric("Avg Queue (Lq)", f"{stats['average_queue_length']:.4f}", help="Avg packets waiting in line")
        m3.metric("Avg Wait (Wq)", f"{stats['average_waiting_time']:.4f}s", help="Avg time a packet waits")
        m4.metric("Utilization", f"{stats['server_utilization']:.2%}", help="% of time servers are busy")
        
        tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📋 Detailed Stats", "📐 Theoretical Comparison"])
        
        with tab1:
            ts_df = sim.get_time_series_dataframe()
            if not ts_df.empty:
                st.subheader("System Dynamics")
                fig_ts = px.line(ts_df, x='time', y=['queue_length', 'servers_busy'], 
                                title="Queue Length & Busy Servers Over Time")
                st.plotly_chart(fig_ts, use_container_width=True)
            
            c_viz1, c_viz2 = st.columns(2)
            if stats['waiting_times']:
                with c_viz1:
                    fig_w = px.histogram(x=stats['waiting_times'], nbins=30, title="Wait Time Distribution", labels={'x': 'Time (s)'})
                    st.plotly_chart(fig_w, use_container_width=True)
            if stats['system_times']:
                with c_viz2:
                    fig_s = px.histogram(x=stats['system_times'], nbins=30, title="System Time Distribution", labels={'x': 'Time (s)'}, color_discrete_sequence=['green'])
                    st.plotly_chart(fig_s, use_container_width=True)

        with tab2:
            st.dataframe(pd.DataFrame({
                'Metric': ['Total Arrivals', 'Total Departures', 'Drops', 'Avg Queue', 'Avg Wait', 'Max Wait', 'StdDev Wait'],
                'Value': [stats['total_arrivals'], stats['total_departures'], stats['total_drops'],
                    f"{stats['average_queue_length']:.4f}", f"{stats['average_waiting_time']:.4f}",
                    f"{stats['max_waiting_time']:.4f}", f"{stats['std_waiting_time']:.4f}"]
            }), hide_index=True, use_container_width=True)
            
            if num_servers > 1:
                st.subheader("Server Load Balance")
                st.dataframe(sim.get_server_stats_dataframe(), hide_index=True)
                
        with tab3:
            if arr_dist == "Exponential" and svc_dist == "Exponential":
                theo = compute_mmc_theoretical(arr_params['rate'], svc_params['rate'], num_servers)
                if theo['stable']:
                    st.markdown("#### M/M/c Theoretical vs Simulated")
                    st.latex(r"L_q = \frac{P_0 ((\lambda/\mu)^c) \rho}{c! (1-\rho)^2}")
                    
                    comp_df = pd.DataFrame({
                        "Metric": ["Lq (Queue Len)", "Wq (Wait Time)", "Rho (Util)"],
                        "Theoretical": [theo['Lq'], theo['Wq'], theo['rho']],
                        "Simulated": [stats['average_queue_length'], stats['average_waiting_time'], stats['server_utilization']],
                        "Abs Error": [
                            abs(stats['average_queue_length']-theo['Lq']),
                            abs(stats['average_waiting_time']-theo['Wq']),
                            abs(stats['server_utilization']-theo['rho'])
                        ]
                    })
                    st.dataframe(comp_df, hide_index=True)
                    
                    # EDUCATIONAL TIP
                    if sim_time < 500 and abs(stats['average_waiting_time']-theo['Wq']) > 0.05:
                        st.info("💡 **Tip:** Your simulated result differs from theory. Try increasing **Simulation Time** (>1000) or enabling **Warm-up** to reduce initialization bias.")
                else:
                    st.error("Theoretical System Unstable (Rho >= 1)")
            else:
                st.info("Theoretical comparison only available for M/M/c (Exponential/Exponential)")

        with st.expander("📂 Event Logs & Export"):
            log_df = sim.get_event_log_dataframe()
            st.dataframe(log_df.head(200), use_container_width=True)
            st.download_button("Download CSV", log_df.to_csv(index=False), "event_log.csv", "text/csv")

# ==========================================
# MODE 2: COMPARATIVE ANALYSIS
# ==========================================
elif mode == "Comparative Analysis":
    st.header("Compare Configurations")
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("System A")
        s_a = st.number_input("Servers A", 1, 10, 1)
        r_a = st.number_input("Service Rate A", 0.1, 20.0, 8.0)
    with col_b:
        st.subheader("System B")
        s_b = st.number_input("Servers B", 1, 10, 2)
        r_b = st.number_input("Service Rate B", 0.1, 20.0, 4.0)
    lam = st.number_input("Common Arrival Rate", 1.0, 20.0, 5.0)
    
    if st.button("Run Comparison"):
        cfg_a = SimulationConfig(num_servers=s_a, service_rate=r_a, arrival_rate=lam)
        cfg_b = SimulationConfig(num_servers=s_b, service_rate=r_b, arrival_rate=lam)
        res = run_comparative_analysis([cfg_a, cfg_b])
        
        metrics = ['average_queue_length', 'average_waiting_time', 'server_utilization']
        names = ['Avg Queue (Lq)', 'Avg Wait (Wq)', 'Utilization (ρ)']
        fig = make_subplots(rows=1, cols=3, subplot_titles=names)
        for i, m in enumerate(metrics):
            fig.add_trace(go.Bar(x=['System A', 'System B'], y=[res[0][m], res[1][m]], name=names[i]), row=1, col=i+1)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODE 3: STATISTICAL VALIDATION
# ==========================================
elif mode == "Statistical Validation":
    st.header("Confidence Intervals")
    st.markdown("Run multiple **Replications** to reduce variance and find the true mean.")
    n_reps = st.number_input("Replications", 5, 100, 20)
    conf = st.slider("Confidence Level", 0.8, 0.99, 0.95)
    
    if st.button("Run Validation"):
        cfg = SimulationConfig(num_servers=1, arrival_rate=4.0, service_rate=5.0)
        with st.spinner("Running replications..."):
            res = run_replications(cfg, n_reps, conf)
            
        st.success(f"Wait Time CI: [{res['Wq']['lower']:.4f}, {res['Wq']['upper']:.4f}]")
        fig = px.histogram(res['Wq']['values'], nbins=10, title="Distribution of Means")
        fig.add_vline(x=res['Wq']['mean'], line_color='red')
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODE 4: INPUT ANALYSIS (ENHANCED)
# ==========================================
elif mode == "Input Analysis (Goodness of Fit)":
    st.header("Input Modeling (Chi-Square Test)")
    st.markdown("""
    **Goal:** Test if your observed data fits an **Exponential Distribution**.
    * **$H_0$ (Null Hypothesis):** Data follows Exponential distribution.
    * **$H_1$ (Alternative Hypothesis):** Data does NOT follow Exponential distribution.
    """)
    
    f = st.file_uploader("Upload CSV (Single Column)", type="csv")
    if f:
        data = pd.read_csv(f).iloc[:,0].tolist()
        mean_val = np.mean(data)
        st.write(f"**Observed Mean:** {mean_val:.4f}")
        
        res = perform_chi_square_test(data, "Exponential", mean_val)
        
        if 'error' in res:
            st.error(res['error'])
        else:
            # 1. Summary Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Chi-Square Stat ($X^2$)", f"{res['chi2']:.4f}")
            c2.metric("Critical Value", f"{res['critical']:.4f}")
            c3.metric("P-Value", f"{res['p_value']:.4f}")
            
            # 2. Result Interpretation
            if res['reject']:
                st.error(f"**Result: Reject $H_0$.** (P-Value < 0.05). The data does NOT look Exponential.")
            else:
                st.success(f"**Result: Fail to Reject $H_0$.** (P-Value >= 0.05). The data fits Exponential.")

            # 3. Detailed Table (Educational)
            st.subheader("Goodness of Fit Table")
            
            # Construct DataFrame for table
            rows = []
            obs = res['observed']
            exp = res['expected']
            edges = res['bin_edges']
            
            for i in range(len(obs)):
                lower = edges[i]
                upper = edges[i+1] if i+1 < len(edges) else float('inf')
                contribution = ((obs[i] - exp[i])**2) / exp[i]
                
                rows.append({
                    "Bin Range": f"{lower:.2f} - {upper:.2f}",
                    "Observed ($O_i$)": obs[i],
                    "Expected ($E_i$)": f"{exp[i]:.2f}",
                    "Contribution $\\frac{(O-E)^2}{E}$": f"{contribution:.4f}"
                })
            
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            
            # 4. Visual Comparison
            st.subheader("Observed vs Expected Frequency")
            
            # Prepare data for plotting
            bin_labels = [r['Bin Range'] for r in rows]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=bin_labels, y=obs, name='Observed', marker_color='blue'))
            fig.add_trace(go.Bar(x=bin_labels, y=exp, name='Expected', marker_color='orange'))
            
            fig.update_layout(barmode='group', xaxis_title="Bin Intervals", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
