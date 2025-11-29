"""
Network Traffic Simulation - Educational & Educational Friendly Version
Features: 
- Concept Guides & Tooltips
- Transparent Chi-Square Analysis
- Academic Rigor (LCG, Warm-up)

Run with: streamlit run app.py
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
    run_comparative_analysis
)

st.set_page_config(
    page_title="Network Traffic Simulator",
    page_icon="🎓",
    layout="wide"
)

st.title("Network Traffic Simulation 🎓")
st.markdown("### Discrete Event Simulation (Event-Scheduling Approach)")

# --- EDUCATIONAL: Concept Guide ---
with st.expander("📘 Concept Guide: Click to learn the basics"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **1. Kendall's Notation (A/B/c):**
        * **A**: Arrival Process (e.g., **M** = Markovian/Exponential).
        * **B**: Service Process (e.g., **M** = Markovian, **G** = General).
        * **c**: Number of Servers.
        
        **2. Key Metrics:**
        * $L_q$: Average number of packets in the **Queue**.
        * $W_q$: Average time a packet waits in the **Queue**.
        * $\\rho$ (Traffic Intensity): $\\lambda / (c \\cdot \\mu)$.
        """)
    with c2:
        st.markdown("""
        **3. Random Number Generation (RNG):**
        * **LCG**: A formula $X_{i+1} = (aX_i + c) \mod m$ used to generate pseudo-random numbers.
        
        **4. Stability Condition:**
        * If $\\rho \\ge 1$, the system is **unstable** (queue grows infinitely).
        * If $\\rho < 1$, the system is **stable**.
        
        **5. Statistical Variance:**
        * M/M/1 queues have high variance. Short simulations may deviate from theory.
        * Use **Running Average** in visualization to see convergence.
        """)

st.markdown("---")

# --- Mode Selection ---
mode = st.sidebar.radio(
    "Select Mode",
    ["Single Simulation", "Comparative Analysis", "Statistical Validation"],
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
            
            warmup_time = st.number_input("Warm-up (T0)", 0.0, 500.0, 20.0, help="Initial period to discard to remove bias.")
            random_seed = st.number_input("Random Seed", 1, 99999, 42)
        
        st.subheader("System Configuration")
        num_servers = st.number_input("Servers (c)", 1, 50, 1)
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
                st.subheader("System Dynamics & Convergence")
                
                # Plot with Dual Y-axis or Overlay
                fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
                
                # 1. Queue Length (Gray area)
                fig_ts.add_trace(go.Scatter(
                    x=ts_df['time'], y=ts_df['queue_length'], 
                    name="Instant Queue Length", mode='lines', line=dict(color='rgba(100,100,100,0.3)')
                ), secondary_y=False)
                
                # 2. Running Average (Red line)
                fig_ts.add_trace(go.Scatter(
                    x=ts_df['time'], y=ts_df['running_avg_lq'], 
                    name="Running Avg (Lq)", mode='lines', line=dict(color='red', width=3)
                ), secondary_y=False)
                
                # 3. Theoretical Line (if applicable)
                if arr_dist == "Exponential" and svc_dist == "Exponential":
                    theo = compute_mmc_theoretical(arr_params['rate'], svc_params['rate'], num_servers)
                    if theo['stable']:
                        fig_ts.add_hline(y=theo['Lq'], line_dash="dash", line_color="blue", annotation_text=f"Theory: {theo['Lq']:.2f}")

                fig_ts.update_layout(title="Queue Length vs Running Average", xaxis_title="Time", yaxis_title="Queue Length")
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
                    if sim_time < 500 and abs(stats['average_queue_length']-theo['Lq']) > 0.1:
                        st.info("💡 **Why the difference?** Short simulations have high variance. Look at the **Running Avg** line in the Visualization tab. Does it settle?")
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
    st.markdown("Compare how **Servers**, **Distributions (Variance)**, and **Queue Capacity** affect performance.")

    # Global settings for fair comparison
    with st.expander("Global Simulation Settings", expanded=False):
        lam = st.number_input("Common Arrival Rate (λ)", 0.1, 50.0, 5.0, help="Average arrivals per second (Poisson)")
        sim_time_comp = st.number_input("Simulation Time", 100.0, 10000.0, 1000.0, help="Duration of simulation")

    col_a, col_b = st.columns(2)

    def config_ui_column(col, name, key_suffix):
        with col:
            st.subheader(name)
            # 1. Servers
            num_s = st.number_input("Servers", 1, 50, 1, key=f"srv_{key_suffix}")
            # 2. Capacity
            q_cap = st.number_input("Queue Capacity", 0, 5000, 50, key=f"cap_{key_suffix}", help="0 = Infinite")
            
            # 3. Variance / Distribution
            dist_opt = st.selectbox("Service Distribution", 
                                  ["Exponential (High Variance)", "Uniform (Medium Variance)", "Deterministic (No Variance)"], 
                                  key=f"dist_{key_suffix}")
            
            if "Exponential" in dist_opt:
                mu = st.number_input("Service Rate (μ)", 0.1, 50.0, 8.0, key=f"mu_{key_suffix}")
                return SimulationConfig(
                    num_servers=num_s, queue_capacity=q_cap, simulation_time=sim_time_comp,
                    arrival_rate=lam, service_distribution=DistributionType.EXPONENTIAL, service_rate=mu
                )
            elif "Uniform" in dist_opt:
                mean_s = st.number_input("Mean Service Time", 0.01, 2.0, 0.125, key=f"mean_{key_suffix}")
                spread = st.slider("Spread (%)", 0, 100, 20, key=f"spread_{key_suffix}", help="Range around mean")
                half_width = mean_s * (spread/100)
                return SimulationConfig(
                    num_servers=num_s, queue_capacity=q_cap, simulation_time=sim_time_comp,
                    arrival_rate=lam, service_distribution=DistributionType.UNIFORM, 
                    service_min=mean_s-half_width, service_max=mean_s+half_width
                )
            else: # Deterministic
                mean_s = st.number_input("Constant Service Time", 0.01, 2.0, 0.125, key=f"const_{key_suffix}")
                return SimulationConfig(
                    num_servers=num_s, queue_capacity=q_cap, simulation_time=sim_time_comp,
                    arrival_rate=lam, service_distribution=DistributionType.NORMAL, 
                    service_mean=mean_s, service_std=0.000001 # Practically deterministic
                )

    cfg_a = config_ui_column(col_a, "System A", "A")
    cfg_b = config_ui_column(col_b, "System B", "B")
    
    if st.button("Run Comparison", type="primary"):
        with st.spinner("Running simulations..."):
            res = run_comparative_analysis([cfg_a, cfg_b])
        
        # 1. Bar Charts (Averages)
        metrics = ['average_queue_length', 'average_waiting_time', 'server_utilization', 'drop_rate']
        names = ['Avg Queue (Lq)', 'Avg Wait (Wq)', 'Utilization (ρ)', 'Drop Rate (%)']
        
        # Adjust data for plotting (convert drop rate to %)
        res[0]['drop_rate'] = res[0]['drop_rate'] * 100
        res[1]['drop_rate'] = res[1]['drop_rate'] * 100
        
        st.subheader("1. Performance Metrics (Averages)")
        fig = make_subplots(rows=1, cols=4, subplot_titles=names)
        colors = ['#1f77b4', '#ff7f0e'] # Blue, Orange
        
        for i, m in enumerate(metrics):
            fig.add_trace(go.Bar(
                x=['System A', 'System B'], 
                y=[res[0][m], res[1][m]], 
                name=names[i],
                marker_color=colors,
                showlegend=False
            ), row=1, col=i+1)
        st.plotly_chart(fig, use_container_width=True)
        
        # 2. Time Series Comparison (Queue Evolution)
        st.subheader("2. Queue Length Evolution (Time Series)")
        st.caption("Visualizing how the queue builds up and clears over time.")
        
        ts_a = res[0]['time_series_df']
        ts_b = res[1]['time_series_df']
        
        fig_ts = go.Figure()
        # Downsample for performance if too large
        step_a = max(1, len(ts_a)//2000)
        step_b = max(1, len(ts_b)//2000)
        
        fig_ts.add_trace(go.Scatter(x=ts_a['time'][::step_a], y=ts_a['queue_length'][::step_a], 
                                   name="System A (Queue)", line=dict(color='#1f77b4', width=1)))
        fig_ts.add_trace(go.Scatter(x=ts_b['time'][::step_b], y=ts_b['queue_length'][::step_b], 
                                   name="System B (Queue)", line=dict(color='#ff7f0e', width=1)))
        
        fig_ts.update_layout(xaxis_title="Time", yaxis_title="Queue Length", height=400)
        st.plotly_chart(fig_ts, use_container_width=True)

        # 3. Box Plots (Variability)
        st.subheader("3. Wait Time Consistency (Box Plots)")
        st.caption("Lower boxes mean faster service. Taller boxes mean unpredictable (variable) service.")
        
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(y=res[0]['waiting_times'], name="System A", marker_color='#1f77b4'))
        fig_box.add_trace(go.Box(y=res[1]['waiting_times'], name="System B", marker_color='#ff7f0e'))
        fig_box.update_layout(yaxis_title="Waiting Time (s)", showlegend=False, height=400)
        st.plotly_chart(fig_box, use_container_width=True)

# ==========================================
# MODE 3: STATISTICAL VALIDATION
# ==========================================
elif mode == "Statistical Validation":
    st.header("Confidence Intervals & Validation")
    
    with st.expander("📘 What is this?", expanded=True):
        st.markdown("""
        **Why do we need this?**
        One simulation run is just one "sample". If you run it again, you get a different result.
        To trust our data, we run the simulation **N times (Replications)** and calculate an interval where the true mean likely lies.
        
        **Confidence Interval (CI):**
        An estimated range of values which is likely to include an unknown population parameter (the true mean).
        * *95% CI* means if we repeated this experiment 100 times, the true mean would be inside the calculated interval 95 times.
        """)

    col_conf, col_viz = st.columns([1, 2])

    with col_conf:
        st.subheader("Configuration")
        n_reps = st.number_input("Replications (N)", 5, 200, 30, help="More reps = Narrower CI (More precision)")
        conf = st.slider("Confidence Level", 0.80, 0.99, 0.95, help="Standard is 0.95")
        
        metric_to_validate = st.selectbox("Metric to Validate", ["Wait Time (Wq)", "Queue Length (Lq)", "Utilization (ρ)"], index=0)
        
        st.markdown("---")
        st.markdown("**System Settings**")
        v_servers = st.number_input("Servers", 1, 10, 1)
        v_lam = st.number_input("Arrival Rate", 0.1, 20.0, 4.0)
        v_mu = st.number_input("Service Rate", 0.1, 20.0, 5.0)
        v_sim_time = st.number_input("Sim Time per Rep", 100.0, 5000.0, 500.0)

    if st.button("Run Validation Analysis", type="primary"):
        cfg = SimulationConfig(num_servers=v_servers, arrival_rate=v_lam, service_rate=v_mu, simulation_time=v_sim_time)
        
        with st.spinner(f"Running {n_reps} replications..."):
            res = run_replications(cfg, n_reps, conf)
        
        # Map selection to internal key
        key_map = {"Wait Time (Wq)": "Wq", "Queue Length (Lq)": "Lq", "Utilization (ρ)": "utilization"}
        sel_key = key_map[metric_to_validate]
        data = res[sel_key]
        
        with col_viz:
            st.subheader(f"Validation Results: {metric_to_validate}")
            
            # KPI Cards
            k1, k2, k3 = st.columns(3)
            k1.metric("Mean of Means", f"{data['mean']:.4f}")
            k2.metric("CI Width (Precision)", f"± {(data['upper']-data['mean']):.4f}")
            k3.metric("Standard Deviation", f"{data['std']:.4f}")
            
            st.success(f"**95% Confidence Interval:** [{data['lower']:.4f}, {data['upper']:.4f}]")
            
            # Graph 1: Scatter of Means
            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                y=data['values'], 
                mode='markers', 
                name='Replication Result',
                marker=dict(color='rgba(0, 100, 255, 0.6)', size=8)
            ))
            # Add Mean Line
            fig_scatter.add_hline(y=data['mean'], line_color="red", line_width=2, annotation_text="Mean")
            # Add CI Box
            fig_scatter.add_hrect(y0=data['lower'], y1=data['upper'], line_width=0, fillcolor="red", opacity=0.1, annotation_text="95% CI Region")
            
            fig_scatter.update_layout(title="Variation Across Replications", xaxis_title="Replication ID", yaxis_title=metric_to_validate)
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Graph 2: Convergence (Educational)
            # Calculate cumulative mean to show convergence
            cumulative_means = [np.mean(data['values'][:i+1]) for i in range(len(data['values']))]
            fig_conv = go.Figure()
            fig_conv.add_trace(go.Scatter(y=cumulative_means, mode='lines', name='Cumulative Mean'))
            fig_conv.add_hline(y=data['mean'], line_dash="dash", line_color="gray", annotation_text="Final Mean")
            fig_conv.update_layout(title="Law of Large Numbers: Convergence of Mean", xaxis_title="Number of Replications Included", yaxis_title="Cumulative Mean")
            st.plotly_chart(fig_conv, use_container_width=True)
