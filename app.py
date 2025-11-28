"""
Network Traffic Simulation - Merged & Enhanced
Discrete Event Simulation for SWE627 Course
Supports M/M/1 and M/M/c queuing systems with multiple distributions
Includes: Single Sim, Comparison, Validation, and Input Analysis

ADEL2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from simulation_engine import (
    NetworkSimulator, SimulationConfig, DistributionType, 
    perform_chi_square_test, compute_mmc_theoretical, 
    compute_confidence_interval, run_replications
)

st.set_page_config(layout="wide", page_title="SWE 627 Simulator")

st.title("SWE 627: Network Traffic Simulation")
st.markdown("### Discrete Event Simulation (Event-Scheduling Approach)")

# --- Mode Selection (Added "Input Analysis") ---
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Simulation", "Comparative Analysis", "Statistical Validation", "Input Analysis (Goodness of Fit)"]
)

# --- HELPER: Param Widget ---
def get_dist_params(prefix, dist_type):
    params = {}
    if dist_type == DistributionType.EXPONENTIAL:
        rate = st.number_input(f"{prefix} Rate (λ)", 0.1, 50.0, 5.0, key=f"{prefix}_r")
        params['rate'] = rate
        params['mean'] = 1/rate
    elif dist_type == DistributionType.POISSON:
        lam = st.number_input(f"{prefix} Lambda", 1.0, 50.0, 5.0, key=f"{prefix}_l")
        params['lam'] = lam
        params['mean'] = lam
    elif dist_type == DistributionType.NORMAL:
        mean = st.number_input(f"{prefix} Mean", 0.1, 100.0, 10.0, key=f"{prefix}_m")
        std = st.number_input(f"{prefix} Std Dev", 0.1, 20.0, 2.0, key=f"{prefix}_s")
        params['mean'] = mean; params['std'] = std
    elif dist_type == DistributionType.UNIFORM:
        mn = st.number_input(f"{prefix} Min", 0.0, 50.0, 1.0, key=f"{prefix}_min")
        mx = st.number_input(f"{prefix} Max", 0.1, 100.0, 5.0, key=f"{prefix}_max")
        params['min'] = mn; params['max'] = mx
        params['mean'] = (mn+mx)/2
    else:
        # Default fallback for complex ones
        st.info(f"{prefix} using standard parameters (Mean=1.0)")
        params['mean'] = 1.0
        params['shape'] = 2.0
        params['scale'] = 1.0
    return params

# ==========================================
# MODE 1: SINGLE SIMULATION
# ==========================================
if mode == "Single Simulation":
    with st.sidebar:
        st.header("System Config")
        
        # NEW: LCG & Warmup
        st.subheader("RNG & Time")
        use_lcg = st.checkbox("Use Custom LCG", True, help="Use Linear Congruential Generator")
        seed = st.number_input("Seed", 1, 999999, 12345)
        sim_time = st.number_input("Total Time", 10.0, 5000.0, 100.0)
        warmup = st.number_input("Warm-up Time (T0)", 0.0, 1000.0, 0.0, help="Initial data deletion")
        
        st.subheader("Resources")
        n_servers = st.number_input("Servers (c)", 1, 50, 1)
        capacity = st.number_input("Queue Cap", 0, 1000, 100)
        
        st.markdown("---")
        st.subheader("Arrival Process")
        arr_type = st.selectbox("Arr. Dist", [d.value for d in DistributionType], index=0)
        arr_params = get_dist_params("Arr", DistributionType(arr_type))
        
        st.markdown("---")
        st.subheader("Service Process")
        svc_type = st.selectbox("Svc. Dist", [d.value for d in DistributionType], index=0)
        svc_params = get_dist_params("Svc", DistributionType(svc_type))
        
        run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        # Setup Config
        cfg = SimulationConfig(
            use_lcg=use_lcg, random_seed=seed,
            simulation_time=sim_time, warmup_time=warmup,
            num_servers=n_servers, queue_capacity=capacity,
            arrival_distribution=DistributionType(arr_type),
            service_distribution=DistributionType(svc_type)
        )
        # Apply params
        for k,v in arr_params.items(): setattr(cfg, f"arrival_{k}", v)
        for k,v in svc_params.items(): setattr(cfg, f"service_{k}", v)
        
        # Run
        sim = NetworkSimulator(cfg)
        res = sim.run()
        
        # --- Display Results ---
        st.header("Simulation Results")
        
        # KPI Row
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Throughput", f"{res['throughput']:.3f}/s")
        k2.metric("Avg Queue (Lq)", f"{res['average_queue_length']:.4f}")
        k3.metric("Avg Wait (Wq)", f"{res['average_waiting_time']:.4f}s")
        k4.metric("Utilization", f"{res['server_utilization']:.2%}")
        
        # Charts
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Queue Length Over Time")
            ts_df = sim.get_time_series_dataframe()
            if not ts_df.empty:
                st.line_chart(ts_df, x='time', y='queue_length')
        
        with c2:
            st.subheader("Wait Time Distribution")
            if res['waiting_times']:
                fig = px.histogram(res['waiting_times'], nbins=30, title="Histogram")
                st.plotly_chart(fig, use_container_width=True)
                
        # Theoretical Comparison (M/M/c)
        if arr_type == "Exponential" and svc_type == "Exponential":
            theo = compute_mmc_theoretical(arr_params['rate'], svc_params['rate'], n_servers)
            if theo['stable']:
                st.info(f"**Theoretical M/M/{n_servers}:** Lq = {theo['Lq']:.4f} | Wq = {theo['Wq']:.4f} | ρ = {theo['rho']:.3f}")
            else:
                st.warning("Theoretical System Unstable (ρ >= 1)")

        # Logs
        with st.expander("Event Log (Detailed Trace)"):
            st.dataframe(sim.get_event_log_dataframe())

# ==========================================
# MODE 2: COMPARATIVE ANALYSIS
# ==========================================
elif mode == "Comparative Analysis":
    st.header("Compare Configurations")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("System A")
        srv_a = st.number_input("Servers A", 1, 10, 1)
        mu_a = st.number_input("Service Rate A", 0.1, 20.0, 8.0)
    with c2:
        st.subheader("System B")
        srv_b = st.number_input("Servers B", 1, 10, 2)
        mu_b = st.number_input("Service Rate B", 0.1, 20.0, 4.0)
    
    lam = st.number_input("Common Arrival Rate", 0.1, 20.0, 5.0)
    
    if st.button("Compare"):
        # Run A
        cfg_a = SimulationConfig(num_servers=srv_a, service_rate=mu_a, arrival_rate=lam)
        sim_a = NetworkSimulator(cfg_a); res_a = sim_a.run()
        
        # Run B
        cfg_b = SimulationConfig(num_servers=srv_b, service_rate=mu_b, arrival_rate=lam)
        sim_b = NetworkSimulator(cfg_b); res_b = sim_b.run()
        
        comp_df = pd.DataFrame({
            "Metric": ["Avg Queue", "Avg Wait", "Utilization"],
            "System A": [res_a['average_queue_length'], res_a['average_waiting_time'], res_a['server_utilization']],
            "System B": [res_b['average_queue_length'], res_b['average_waiting_time'], res_b['server_utilization']]
        })
        st.table(comp_df)

# ==========================================
# MODE 3: STATISTICAL VALIDATION
# ==========================================
elif mode == "Statistical Validation":
    st.header("Confidence Intervals")
    reps = st.number_input("Replications", 5, 100, 20)
    alpha = st.slider("Confidence Level", 0.8, 0.99, 0.95)
    
    if st.button("Run Validation"):
        cfg = SimulationConfig(random_seed=12345)
        with st.spinner("Running replications..."):
            res = run_replications(cfg, reps)
            
        mean_wq, lo, hi = compute_confidence_interval(res['Wq'], alpha)
        
        st.metric("Mean Wait Time", f"{mean_wq:.4f}")
        st.success(f"{int(alpha*100)}% CI: [{lo:.4f}, {hi:.4f}]")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=res['Wq'], mode='markers', name='Replication Mean'))
        fig.add_hline(y=mean_wq, line_color='red')
        fig.add_hrect(y0=lo, y1=hi, opacity=0.2, fillcolor="red")
        st.plotly_chart(fig)

# ==========================================
# MODE 4: INPUT ANALYSIS (NEW)
# ==========================================
elif mode == "Input Analysis (Goodness of Fit)":
    st.header("Input Modeling")
    st.markdown("Upload observed data to test against theoretical distributions.")
    
    up_file = st.file_uploader("Upload CSV (1 column of numbers)", type="csv")
    
    if up_file:
        df = pd.read_csv(up_file)
        data = df.iloc[:,0].tolist()
        
        st.subheader("Descriptive Stats")
        st.write(pd.DataFrame(data).describe().T)
        
        fig = px.histogram(data, nbins=20, title="Observed Histogram")
        st.plotly_chart(fig)
        
        st.subheader("Chi-Square Test")
        mean_obs = np.mean(data)
        
        # Test against Exponential
        res = perform_chi_square_test(data, "Exponential", mean_obs)
        if 'error' in res:
            st.error(res['error'])
        else:
            c1, c2 = st.columns(2)
            c1.metric("Chi-Square Stat", f"{res['chi2']:.4f}")
            c1.metric("Critical Value", f"{res['critical']:.4f}")
            c2.metric("P-Value", f"{res['p_value']:.4f}")
            
            if res['reject']:
                st.error("Reject Null Hypothesis: Data is likely NOT Exponential.")
            else:
                st.success("Fail to Reject Null: Data could be Exponential.")
