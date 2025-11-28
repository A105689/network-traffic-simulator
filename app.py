"""
Network Traffic Simulation - Ultimate Merged Version
Features: 
- Rich Dashboard (KPIs, Charts, Logs, Export)
- Academic Requirements (LCG, Warm-up, Chi-Square)
- Comparative Analysis
- Statistical Validation

ADEL2025

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
    page_icon="🔄",
    layout="wide"
)

st.title("Network Traffic Simulation")
st.markdown("### Discrete Event Simulation (Event-Scheduling Approach)")

# --- Mode Selection ---
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Simulation", "Comparative Analysis", "Statistical Validation", "Input Analysis (Goodness of Fit)"],
    index=0
)

st.markdown("---")

# --- Helper: Distribution Widget ---
def get_distribution_params(prefix: str, dist_type: DistributionType, default_rate: float = 5.0):
    """Dynamic widget generation based on distribution type"""
    params = {}
    
    if dist_type == DistributionType.EXPONENTIAL:
        params['rate'] = st.slider(f"{prefix} Rate (λ)", 0.1, 50.0, default_rate, 0.1, key=f"{prefix}_rate")
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
    
    # Fill defaults for unused
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
        
        # --- NEW: Academic Requirements Section ---
        with st.expander("RNG & Initialization", expanded=True):
            use_lcg = st.checkbox("Use Custom LCG", True, help="Use Linear Congruential Generator instead of NumPy")
            warmup_time = st.number_input("Warm-up Period (T0)", 0.0, 500.0, 0.0, help="Time to run before collecting stats")
            random_seed = st.number_input("Random Seed", 1, 99999, 42)
        
        st.subheader("System Configuration")
        num_servers = st.number_input("Number of Servers (c)", 1, 50, 1)
        sim_time = st.number_input("Simulation Time", 10.0, 10000.0, 100.0)
        capacity = st.number_input("Queue Capacity (0=Inf)", 0, 1000, 50)
        
        st.markdown("---")
        st.subheader("Arrival Process")
        arr_dist = st.selectbox("Distribution", [d.value for d in DistributionType], index=0, key="arr_dist_box")
        arr_params = get_distribution_params("Arrival", DistributionType(arr_dist), 5.0)
        
        st.markdown("---")
        st.subheader("Service Process")
        svc_dist = st.selectbox("Distribution", [d.value for d in DistributionType], index=0, key="svc_dist_box")
        svc_params = get_distribution_params("Service", DistributionType(svc_dist), 8.0)
        
        st.markdown("---")
        # Traffic Intensity Calc
        rho_est = arr_params['rate'] / (num_servers * svc_params['rate']) if svc_params['rate'] > 0 else 0
        st.metric("Estimated Traffic (ρ)", f"{rho_est:.3f}")
        
        run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

    if run_btn:
        # Create Config
        cfg = SimulationConfig(
            use_lcg=use_lcg,
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
        
        # --- RESULTS DASHBOARD ---
        
        # 1. Config Summary
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Arrival:** {arr_dist} (Mean={arr_params['mean']:.2f})")
        c2.info(f"**Service:** {svc_dist} (Mean={svc_params['mean']:.2f})")
        c3.info(f"**System:** M/G/{num_servers}/{capacity} (Warmup={warmup_time})")
        
        # 2. Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Throughput", f"{stats['throughput']:.3f}/sec")
        m2.metric("Avg Queue (Lq)", f"{stats['average_queue_length']:.4f}")
        m3.metric("Avg Wait (Wq)", f"{stats['average_waiting_time']:.4f}s")
        m4.metric("Utilization", f"{stats['server_utilization']:.2%}")
        
        # 3. Tabs
        tab1, tab2, tab3 = st.tabs(["Visualization", "Detailed Stats", "Theoretical Comparison"])
        
        with tab1:
            # Time Series
            ts_df = sim.get_time_series_dataframe()
            if not ts_df.empty:
                st.subheader("System State Over Time")
                fig_ts = px.line(ts_df, x='time', y=['queue_length', 'servers_busy'], 
                                title="Queue Length & Busy Servers")
                st.plotly_chart(fig_ts, use_container_width=True)
            
            # Histograms
            c_viz1, c_viz2 = st.columns(2)
            if stats['waiting_times']:
                with c_viz1:
                    fig_w = px.histogram(x=stats['waiting_times'], nbins=30, title="Waiting Time Distribution",
                                       labels={'x': 'Time (s)'})
                    st.plotly_chart(fig_w, use_container_width=True)
            if stats['system_times']:
                with c_viz2:
                    fig_s = px.histogram(x=stats['system_times'], nbins=30, title="System Time Distribution",
                                       labels={'x': 'Time (s)'}, color_discrete_sequence=['green'])
                    st.plotly_chart(fig_s, use_container_width=True)

        with tab2:
            st.subheader("Performance Metrics")
            perf_data = {
                'Metric': ['Total Arrivals', 'Total Departures', 'Drops', 'Avg Queue', 'Avg Wait', 'Max Wait', 'StdDev Wait'],
                'Value': [
                    stats['total_arrivals'], stats['total_departures'], stats['total_drops'],
                    f"{stats['average_queue_length']:.4f}", f"{stats['average_waiting_time']:.4f}",
                    f"{stats['max_waiting_time']:.4f}", f"{stats['std_waiting_time']:.4f}"
                ]
            }
            st.dataframe(pd.DataFrame(perf_data), hide_index=True)
            
            if num_servers > 1:
                st.subheader("Server Load Balance")
                st.dataframe(sim.get_server_stats_dataframe(), hide_index=True)
                
        with tab3:
            if arr_dist == "Exponential" and svc_dist == "Exponential":
                theo = compute_mmc_theoretical(arr_params['rate'], svc_params['rate'], num_servers)
                if theo['stable']:
                    st.success(f"Theoretical M/M/{num_servers} Stable")
                    comp_df = pd.DataFrame({
                        "Metric": ["Lq", "Wq", "L", "W", "Rho"],
                        "Theoretical": [theo['Lq'], theo['Wq'], theo['L'], theo['W'], theo['rho']],
                        "Simulated": [stats['average_queue_length'], stats['average_waiting_time'], 
                                     stats['average_system_length'], stats['average_system_time'], stats['server_utilization']],
                        "Error %": [
                            f"{abs(stats['average_queue_length']-theo['Lq'])/theo['Lq']*100:.1f}%" if theo['Lq']>0 else "-",
                            f"{abs(stats['average_waiting_time']-theo['Wq'])/theo['Wq']*100:.1f}%" if theo['Wq']>0 else "-",
                            "-", "-", "-"
                        ]
                    })
                    st.dataframe(comp_df, hide_index=True)
                else:
                    st.error("Theoretical System Unstable (Rho >= 1)")
            else:
                st.info("Theoretical comparison only available for M/M/c (Exponential/Exponential)")

        # 4. Logs & Export
        with st.expander("Event Logs & Export"):
            log_df = sim.get_event_log_dataframe()
            st.dataframe(log_df.head(200), use_container_width=True)
            
            ec1, ec2, ec3 = st.columns(3)
            with ec1:
                st.download_button("Download CSV", log_df.to_csv(index=False), "event_log.csv", "text/csv")
            with ec2:
                json_d = sim.export_to_json()
                st.download_button("Download JSON", json_d, "sim_data.json", "application/json")

# ==========================================
# MODE 2: COMPARATIVE ANALYSIS
# ==========================================
elif mode == "Comparative Analysis":
    st.header("Compare Configurations")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Config A")
        s_a = st.number_input("Servers A", 1, 10, 1)
        r_a = st.number_input("Service Rate A", 0.1, 20.0, 8.0)
    with col_b:
        st.subheader("Config B")
        s_b = st.number_input("Servers B", 1, 10, 2)
        r_b = st.number_input("Service Rate B", 0.1, 20.0, 4.0)
        
    lam = st.number_input("Common Arrival Rate", 1.0, 20.0, 5.0)
    
    if st.button("Compare"):
        cfg_a = SimulationConfig(num_servers=s_a, service_rate=r_a, arrival_rate=lam)
        cfg_b = SimulationConfig(num_servers=s_b, service_rate=r_b, arrival_rate=lam)
        
        res = run_comparative_analysis([cfg_a, cfg_b])
        
        # Visualization
        metrics = ['average_queue_length', 'average_waiting_time', 'server_utilization']
        names = ['Avg Queue', 'Avg Wait', 'Utilization']
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=names)
        
        for i, m in enumerate(metrics):
            fig.add_trace(go.Bar(x=['Config A', 'Config B'], y=[res[0][m], res[1][m]], name=names[i]), row=1, col=i+1)
            
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(pd.DataFrame(res)[['num_servers', 'service_rate', 'average_waiting_time', 'average_queue_length']])

# ==========================================
# MODE 3: STATISTICAL VALIDATION
# ==========================================
elif mode == "Statistical Validation":
    st.header("Confidence Intervals (Replications)")
    
    n_reps = st.number_input("Replications", 5, 100, 20)
    conf = st.slider("Confidence Level", 0.8, 0.99, 0.95)
    
    # Simple Config for validation
    v_servers = st.number_input("Servers", 1, 10, 1)
    v_lam = st.number_input("Arrival Rate", 1.0, 10.0, 4.0)
    v_mu = st.number_input("Service Rate", 1.0, 10.0, 5.0)
    
    if st.button("Run Validation"):
        cfg = SimulationConfig(num_servers=v_servers, arrival_rate=v_lam, service_rate=v_mu)
        with st.spinner("Running replications..."):
            res = run_replications(cfg, n_reps, conf)
            
        st.subheader("Results")
        st.success(f"Avg Wait Time: {res['Wq']['mean']:.4f} ± {(res['Wq']['upper']-res['Wq']['mean']):.4f}")
        
        # Histogram of means
        fig = px.histogram(res['Wq']['values'], nbins=10, title="Distribution of Wait Times across Replications")
        fig.add_vline(x=res['Wq']['mean'], line_color='red', annotation_text="Mean")
        fig.add_vline(x=res['Wq']['lower'], line_dash='dash', annotation_text="Lower CI")
        fig.add_vline(x=res['Wq']['upper'], line_dash='dash', annotation_text="Upper CI")
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# MODE 4: INPUT ANALYSIS (New)
# ==========================================
elif mode == "Input Analysis (Goodness of Fit)":
    st.header("Input Modeling (Chi-Square Test)")
    st.markdown("Upload observed data to test if it fits an **Exponential** distribution.")
    
    f = st.file_uploader("Upload CSV (Single Column)", type="csv")
    if f:
        data = pd.read_csv(f).iloc[:,0].tolist()
        st.write(pd.DataFrame(data).describe().T)
        
        mean_val = np.mean(data)
        res = perform_chi_square_test(data, "Exponential", mean_val)
        
        c1, c2 = st.columns(2)
        c1.metric("Chi-Square Stat", f"{res['chi2']:.4f}")
        c2.metric("P-Value", f"{res['p_value']:.4f}")
        
        if res['reject']:
            st.error("Reject Null: Data is likely NOT Exponential")
        else:
            st.success("Fail to Reject: Data fits Exponential")
            
        fig = px.histogram(data, nbins=20, title="Observed Data Histogram")
        st.plotly_chart(fig, use_container_width=True)
