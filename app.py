"""
Network Traffic Simulation Engine using Event-Scheduling Approach
Discrete Event Simulation for SWE627 Course
Supports M/M/1 and M/M/c queuing systems with multiple distributions

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
    run_comparative_analysis
)

st.set_page_config(
    page_title="Network Traffic Simulator",
    page_icon="🔄",
    layout="wide"
)

st.title("Network Traffic Simulation")
st.markdown("### Event-Scheduling Approach for Discrete Event Simulation")

mode = st.radio(
    "Select Mode",
    ["Single Simulation", "Comparative Analysis", "Statistical Validation"],
    horizontal=True
)

st.markdown("---")

def get_distribution_params(prefix: str, dist_type: DistributionType, default_rate: float = 5.0):
    """Get distribution parameters based on type"""
    params = {}
    
    if dist_type == DistributionType.EXPONENTIAL:
        params['rate'] = st.slider(
            f"{prefix} Rate",
            min_value=0.1, max_value=30.0, value=default_rate, step=0.1,
            key=f"{prefix}_rate",
            help="Average events per time unit"
        )
        st.info(f"Mean: {1/params['rate']:.4f}")
        params['mean'] = 1/params['rate']
        params['std'] = 0.05
        params['min'] = 0.1
        params['max'] = 0.3
        params['shape'] = 2.0
        params['scale'] = 1/params['rate']
        
    elif dist_type == DistributionType.NORMAL:
        params['mean'] = st.slider(
            f"{prefix} Mean",
            min_value=0.01, max_value=2.0, value=1/default_rate, step=0.01,
            key=f"{prefix}_mean"
        )
        params['std'] = st.slider(
            f"{prefix} Std Dev",
            min_value=0.001, max_value=0.5, value=0.05, step=0.001,
            key=f"{prefix}_std"
        )
        params['rate'] = 1/params['mean']
        params['min'] = 0.1
        params['max'] = 0.3
        params['shape'] = 2.0
        params['scale'] = params['mean']
        
    elif dist_type == DistributionType.UNIFORM:
        params['min'] = st.slider(
            f"{prefix} Min",
            min_value=0.01, max_value=1.0, value=0.1, step=0.01,
            key=f"{prefix}_min"
        )
        params['max'] = st.slider(
            f"{prefix} Max",
            min_value=0.02, max_value=2.0, value=0.3, step=0.01,
            key=f"{prefix}_max"
        )
        params['mean'] = (params['min'] + params['max']) / 2
        params['rate'] = 1/params['mean']
        params['std'] = 0.05
        params['shape'] = 2.0
        params['scale'] = params['mean']
        
    elif dist_type == DistributionType.WEIBULL:
        params['shape'] = st.slider(
            f"{prefix} Shape (k)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.1,
            key=f"{prefix}_shape",
            help="Shape parameter (k > 1 for increasing failure rate)"
        )
        params['scale'] = st.slider(
            f"{prefix} Scale (λ)",
            min_value=0.01, max_value=2.0, value=1/default_rate, step=0.01,
            key=f"{prefix}_scale"
        )
        from scipy.special import gamma as gamma_func
        params['mean'] = params['scale'] * gamma_func(1 + 1/params['shape'])
        params['rate'] = 1/params['mean'] if params['mean'] > 0 else default_rate
        params['std'] = 0.05
        params['min'] = 0.1
        params['max'] = 0.3
        st.info(f"Mean: {params['mean']:.4f}")
        
    elif dist_type == DistributionType.GAMMA:
        params['shape'] = st.slider(
            f"{prefix} Shape (k)",
            min_value=0.5, max_value=10.0, value=2.0, step=0.1,
            key=f"{prefix}_shape",
            help="Shape parameter (number of events)"
        )
        params['scale'] = st.slider(
            f"{prefix} Scale (θ)",
            min_value=0.01, max_value=1.0, value=1/(default_rate * 2), step=0.01,
            key=f"{prefix}_scale",
            help="Scale parameter (average time between events)"
        )
        params['mean'] = params['shape'] * params['scale']
        params['rate'] = 1/params['mean'] if params['mean'] > 0 else default_rate
        params['std'] = 0.05
        params['min'] = 0.1
        params['max'] = 0.3
        st.info(f"Mean: {params['mean']:.4f}")
        
    elif dist_type == DistributionType.LOGNORMAL:
        params['mean'] = st.slider(
            f"{prefix} μ (log-mean)",
            min_value=-2.0, max_value=2.0, value=np.log(1/default_rate), step=0.1,
            key=f"{prefix}_mean",
            help="Mean of the underlying normal distribution"
        )
        params['std'] = st.slider(
            f"{prefix} σ (log-std)",
            min_value=0.1, max_value=2.0, value=0.5, step=0.1,
            key=f"{prefix}_std",
            help="Std dev of the underlying normal distribution"
        )
        actual_mean = np.exp(params['mean'] + params['std']**2/2)
        params['rate'] = 1/actual_mean
        params['min'] = 0.1
        params['max'] = 0.3
        params['shape'] = 2.0
        params['scale'] = actual_mean
        st.info(f"Actual Mean: {actual_mean:.4f}")
        
    elif dist_type == DistributionType.POISSON:
        params['mean'] = st.slider(
            f"{prefix} Lambda (λ)",
            min_value=1.0, max_value=20.0, value=max(1.0, 1/default_rate), step=1.0,
            key=f"{prefix}_lambda",
            help="Mean number of events/time units (Integer generation)"
        )
        params['rate'] = 1/params['mean']
        params['std'] = np.sqrt(params['mean'])
        params['min'] = 0.1
        params['max'] = 0.3
        params['shape'] = 2.0
        params['scale'] = params['mean']
        st.info(f"Mean: {params['mean']:.4f} (Variance: {params['mean']:.4f})")
    
    else:
        params = {'rate': default_rate, 'mean': 1/default_rate, 'std': 0.05, 
                  'min': 0.1, 'max': 0.3, 'shape': 2.0, 'scale': 1/default_rate}
    
    return params


if mode == "Single Simulation":
    with st.sidebar:
        st.header("Simulation Parameters")
        
        st.subheader("System Configuration")
        
        num_servers = st.number_input(
            "Number of Servers (c)",
            min_value=1, max_value=20, value=1, step=1,
            help="M/M/1 for c=1, M/M/c for c>1"
        )
        
        simulation_time = st.number_input(
            "Simulation Time",
            min_value=10.0, max_value=10000.0, value=100.0, step=10.0
        )
        
        queue_capacity = st.number_input(
            "Queue Capacity (0 = unlimited)",
            min_value=0, max_value=1000, value=50, step=5
        )
        
        random_seed = st.number_input(
            "Random Seed",
            min_value=1, max_value=99999, value=42, step=1
        )
        
        st.markdown("---")
        st.subheader("Arrival Process")
        
        arrival_dist = st.selectbox(
            "Distribution",
            options=[d.value for d in DistributionType],
            index=0,
            key="arrival_dist"
        )
        arrival_distribution = DistributionType(arrival_dist)
        arrival_params = get_distribution_params("Arrival", arrival_distribution, 5.0)
        
        st.markdown("---")
        st.subheader("Service Process")
        
        service_dist = st.selectbox(
            "Distribution",
            options=[d.value for d in DistributionType],
            index=0,
            key="service_dist"
        )
        service_distribution = DistributionType(service_dist)
        service_params = get_distribution_params("Service", service_distribution, 8.0)
        
        st.markdown("---")
        
        rho = arrival_params['rate'] / (num_servers * service_params['rate'])
        st.metric("Traffic Intensity (ρ)", f"{rho:.3f}")
        
        if rho >= 1:
            st.warning("ρ ≥ 1: System unstable")
        elif rho > 0.8:
            st.info("ρ > 0.8: Heavily loaded")
        else:
            st.success("ρ < 0.8: Stable")
        
        queue_model = f"M/M/{num_servers}" if num_servers > 1 else "M/M/1"
        st.info(f"Queue Model: {queue_model}")
        
        st.markdown("---")
        run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)
    
    config = SimulationConfig(
        arrival_distribution=arrival_distribution,
        arrival_rate=arrival_params['rate'],
        arrival_mean=arrival_params['mean'],
        arrival_std=arrival_params['std'],
        arrival_min=arrival_params['min'],
        arrival_max=arrival_params['max'],
        arrival_shape=arrival_params['shape'],
        arrival_scale=arrival_params['scale'],
        service_distribution=service_distribution,
        service_rate=service_params['rate'],
        service_mean=service_params['mean'],
        service_std=service_params['std'],
        service_min=service_params['min'],
        service_max=service_params['max'],
        service_shape=service_params['shape'],
        service_scale=service_params['scale'],
        num_servers=num_servers,
        queue_capacity=queue_capacity,
        simulation_time=simulation_time,
        random_seed=int(random_seed)
    )
    
    if 'simulation_run' not in st.session_state:
        st.session_state.simulation_run = False
        st.session_state.stats = None
        st.session_state.simulator = None
    
    if run_simulation:
        with st.spinner("Running simulation..."):
            simulator = NetworkSimulator(config)
            stats = simulator.run()
            st.session_state.simulation_run = True
            st.session_state.stats = stats
            st.session_state.simulator = simulator
        st.success("Simulation completed!")
    
    if st.session_state.simulation_run and st.session_state.simulator:
        simulator = st.session_state.simulator
        stats = st.session_state.stats
        
        st.header("Configuration Summary")
        
        config_col1, config_col2, config_col3 = st.columns(3)
        
        with config_col1:
            st.markdown("**Arrival Process**")
            st.table(pd.DataFrame({
                'Parameter': ['Distribution', 'Rate (λ)', 'Mean'],
                'Value': [
                    arrival_distribution.value,
                    f"{arrival_params['rate']:.3f}",
                    f"{arrival_params['mean']:.4f}"
                ]
            }))
        
        with config_col2:
            st.markdown("**Service Process**")
            st.table(pd.DataFrame({
                'Parameter': ['Distribution', 'Rate (μ)', 'Mean'],
                'Value': [
                    service_distribution.value,
                    f"{service_params['rate']:.3f}",
                    f"{service_params['mean']:.4f}"
                ]
            }))
        
        with config_col3:
            st.markdown("**System Parameters**")
            st.table(pd.DataFrame({
                'Parameter': ['Servers (c)', 'Queue Capacity', 'Traffic (ρ)'],
                'Value': [
                    str(num_servers),
                    str(queue_capacity) if queue_capacity > 0 else "Unlimited",
                    f"{rho:.4f}"
                ]
            }))
        
        st.markdown("---")
        st.header("Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Arrivals", stats['total_arrivals'])
            st.metric("Total Departures", stats['total_departures'])
            st.metric("Packets Dropped", stats['total_drops'])
        
        with col2:
            st.metric("Avg Queue Length (Lq)", f"{stats['average_queue_length']:.4f}")
            st.metric("Avg System Length (L)", f"{stats['average_system_length']:.4f}")
            st.metric("Server Utilization (ρ)", f"{stats['server_utilization']:.4f}")
        
        with col3:
            st.metric("Avg Waiting Time (Wq)", f"{stats['average_waiting_time']:.4f}")
            st.metric("Avg System Time (W)", f"{stats['average_system_time']:.4f}")
            st.metric("Avg Service Time", f"{stats['average_service_time']:.4f}")
        
        with col4:
            st.metric("Throughput", f"{stats['throughput']:.4f}")
            st.metric("Drop Rate", f"{stats['drop_rate']:.4%}")
            st.metric("Max Waiting Time", f"{stats['max_waiting_time']:.4f}")
        
        if num_servers > 1:
            st.markdown("---")
            st.header("Per-Server Statistics")
            server_df = simulator.get_server_stats_dataframe()
            st.dataframe(server_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.header("Detailed Summary Tables")
        
        summary_tab1, summary_tab2, summary_tab3 = st.tabs([
            "Performance Metrics", 
            f"Theoretical Comparison (M/M/{num_servers})", 
            "Distribution Statistics"
        ])
        
        with summary_tab1:
            st.subheader("Performance Metrics Table")
            performance_data = {
                'Metric': [
                    'Total Arrivals', 'Total Departures', 'Packets Dropped',
                    'Packets Remaining', 'Number of Servers',
                    'Average Queue Length (Lq)', 'Average System Length (L)',
                    'Average Waiting Time (Wq)', 'Average System Time (W)',
                    'Average Service Time', 'Server Utilization',
                    'Throughput', 'Drop Rate',
                    'Max Waiting Time', 'Max System Time',
                    'Std Dev Waiting Time', 'Std Dev System Time'
                ],
                'Value': [
                    stats['total_arrivals'], stats['total_departures'], stats['total_drops'],
                    stats['packets_remaining'], stats['num_servers'],
                    round(stats['average_queue_length'], 6), round(stats['average_system_length'], 6),
                    round(stats['average_waiting_time'], 6), round(stats['average_system_time'], 6),
                    round(stats['average_service_time'], 6), round(stats['server_utilization'], 6),
                    round(stats['throughput'], 6), f"{stats['drop_rate']:.4%}",
                    round(stats['max_waiting_time'], 6), round(stats['max_system_time'], 6),
                    round(stats['std_waiting_time'], 6), round(stats['std_system_time'], 6)
                ]
            }
            st.dataframe(pd.DataFrame(performance_data), use_container_width=True, hide_index=True)
        
        with summary_tab2:
            st.subheader(f"Theoretical vs Simulated (M/M/{num_servers})")
            
            if arrival_distribution == DistributionType.EXPONENTIAL and \
               service_distribution == DistributionType.EXPONENTIAL:
                theo = compute_mmc_theoretical(arrival_params['rate'], service_params['rate'], num_servers)
                
                if theo['stable']:
                    comparison_data = {
                        'Metric': ['Queue Length (Lq)', 'System Length (L)', 
                                  'Waiting Time (Wq)', 'System Time (W)', 
                                  'Server Utilization (ρ)'],
                        f'Theoretical M/M/{num_servers}': [
                            round(theo['Lq'], 6), round(theo['L'], 6),
                            round(theo['Wq'], 6), round(theo['W'], 6),
                            round(theo['rho'], 6)
                        ],
                        'Simulated': [
                            round(stats['average_queue_length'], 6),
                            round(stats['average_system_length'], 6),
                            round(stats['average_waiting_time'], 6),
                            round(stats['average_system_time'], 6),
                            round(stats['server_utilization'], 6)
                        ],
                        'Difference': [
                            round(stats['average_queue_length'] - theo['Lq'], 6),
                            round(stats['average_system_length'] - theo['L'], 6),
                            round(stats['average_waiting_time'] - theo['Wq'], 6),
                            round(stats['average_system_time'] - theo['W'], 6),
                            round(stats['server_utilization'] - theo['rho'], 6)
                        ],
                        'Error %': [
                            f"{abs(stats['average_queue_length'] - theo['Lq']) / theo['Lq'] * 100:.2f}%" if theo['Lq'] > 0 else "N/A",
                            f"{abs(stats['average_system_length'] - theo['L']) / theo['L'] * 100:.2f}%" if theo['L'] > 0 else "N/A",
                            f"{abs(stats['average_waiting_time'] - theo['Wq']) / theo['Wq'] * 100:.2f}%" if theo['Wq'] > 0 else "N/A",
                            f"{abs(stats['average_system_time'] - theo['W']) / theo['W'] * 100:.2f}%" if theo['W'] > 0 else "N/A",
                            f"{abs(stats['server_utilization'] - theo['rho']) / theo['rho'] * 100:.2f}%" if theo['rho'] > 0 else "N/A"
                        ]
                    }
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                else:
                    st.warning("Traffic intensity ρ ≥ 1: Theoretical values undefined (unstable)")
            else:
                st.info("Theoretical comparison only available for M/M/c (Exponential distributions)")
        
        with summary_tab3:
            st.subheader("Distribution Statistics")
            packet_df = simulator.get_packet_table_dataframe()
            
            if not packet_df.empty:
                waiting_times = [float(w) for w in packet_df['Waiting Time'] if w != 'N/A']
                system_times = [float(s) for s in packet_df['System Time'] if s != 'N/A']
                
                if waiting_times and system_times:
                    dist_stats = {
                        'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 
                                     '25th Pctl', '75th Pctl', '90th Pctl', '95th Pctl'],
                        'Waiting Time': [
                            len(waiting_times),
                            round(np.mean(waiting_times), 6),
                            round(np.median(waiting_times), 6),
                            round(np.std(waiting_times), 6),
                            round(min(waiting_times), 6),
                            round(max(waiting_times), 6),
                            round(np.percentile(waiting_times, 25), 6),
                            round(np.percentile(waiting_times, 75), 6),
                            round(np.percentile(waiting_times, 90), 6),
                            round(np.percentile(waiting_times, 95), 6),
                        ],
                        'System Time': [
                            len(system_times),
                            round(np.mean(system_times), 6),
                            round(np.median(system_times), 6),
                            round(np.std(system_times), 6),
                            round(min(system_times), 6),
                            round(max(system_times), 6),
                            round(np.percentile(system_times, 25), 6),
                            round(np.percentile(system_times, 75), 6),
                            round(np.percentile(system_times, 90), 6),
                            round(np.percentile(system_times, 95), 6),
                        ]
                    }
                    st.dataframe(pd.DataFrame(dist_stats), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.header("Visualization Dashboard")
        
        ts_df = simulator.get_time_series_dataframe()
        
        if not ts_df.empty:
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.subheader("Queue Length Over Time")
                fig_queue = px.line(
                    ts_df, x='time', y='queue_length',
                    labels={'time': 'Simulation Time', 'queue_length': 'Queue Length'}
                )
                fig_queue.update_layout(height=400)
                st.plotly_chart(fig_queue, use_container_width=True)
            
            with viz_col2:
                st.subheader("Packets in System Over Time")
                fig_system = px.line(
                    ts_df, x='time', y='packets_in_system',
                    labels={'time': 'Simulation Time', 'packets_in_system': 'Packets'}
                )
                fig_system.update_layout(height=400)
                fig_system.update_traces(line_color='green')
                st.plotly_chart(fig_system, use_container_width=True)
            
            packet_df = simulator.get_packet_table_dataframe()
            
            if not packet_df.empty:
                viz_col3, viz_col4 = st.columns(2)
                
                with viz_col3:
                    st.subheader("Waiting Time Distribution")
                    waiting_times = [float(w) for w in packet_df['Waiting Time'] if w != 'N/A']
                    if waiting_times:
                        fig_waiting = px.histogram(x=waiting_times, nbins=30,
                            labels={'x': 'Waiting Time', 'y': 'Frequency'})
                        fig_waiting.update_layout(height=400)
                        fig_waiting.update_traces(marker_color='orange')
                        st.plotly_chart(fig_waiting, use_container_width=True)
                
                with viz_col4:
                    st.subheader("System Time Distribution")
                    system_times = [float(s) for s in packet_df['System Time'] if s != 'N/A']
                    if system_times:
                        fig_system_time = px.histogram(x=system_times, nbins=30,
                            labels={'x': 'System Time', 'y': 'Frequency'})
                        fig_system_time.update_layout(height=400)
                        fig_system_time.update_traces(marker_color='purple')
                        st.plotly_chart(fig_system_time, use_container_width=True)
            
            st.subheader("Server Utilization Over Time")
            if 'server_utilization' in ts_df.columns:
                window_size = max(1, len(ts_df) // 50)
                ts_df['util_smooth'] = ts_df['server_utilization'].rolling(window=window_size, min_periods=1).mean()
                
                fig_util = go.Figure()
                fig_util.add_trace(go.Scatter(
                    x=ts_df['time'], y=ts_df['util_smooth'],
                    mode='lines', name='Utilization',
                    line=dict(color='red', width=2)
                ))
                fig_util.add_hline(y=rho, line_dash="dash", line_color="blue",
                                 annotation_text=f"Theoretical ρ = {rho:.3f}")
                fig_util.update_layout(
                    xaxis_title='Simulation Time', yaxis_title='Utilization',
                    height=400, yaxis=dict(range=[0, 1.1])
                )
                st.plotly_chart(fig_util, use_container_width=True)
        
        st.markdown("---")
        st.header("Event Log Tables")
        
        table_tab1, table_tab2 = st.tabs(["Event Schedule Log", "Packet Details"])
        
        with table_tab1:
            st.subheader("Event-Scheduling Log")
            event_df = simulator.get_event_log_dataframe()
            
            if not event_df.empty:
                st.info(f"Total events: {len(event_df)}")
                
                num_events = st.slider("Events to display", 10, min(500, len(event_df)), 
                                       min(100, len(event_df)), 10)
                display_opt = st.radio("Display:", ["First N", "Last N", "All"], horizontal=True)
                
                if display_opt == "First N":
                    display_df = event_df.head(num_events)
                elif display_opt == "Last N":
                    display_df = event_df.tail(num_events)
                else:
                    display_df = event_df
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        with table_tab2:
            st.subheader("Packet-Level Statistics")
            packet_df = simulator.get_packet_table_dataframe()
            
            if not packet_df.empty:
                st.info(f"Total completed packets: {len(packet_df)}")
                
                num_packets = st.slider("Packets to display", 10, min(500, len(packet_df)),
                                        min(100, len(packet_df)), 10, key="pkt_slider")
                pkt_display = st.radio("Display:", ["First N", "Last N", "All"], 
                                       horizontal=True, key="pkt_radio")
                
                if pkt_display == "First N":
                    display_pkt = packet_df.head(num_packets)
                elif pkt_display == "Last N":
                    display_pkt = packet_df.tail(num_packets)
                else:
                    display_pkt = packet_df
                
                st.dataframe(display_pkt, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.header("Export Data")
        
        export_col1, export_col2, export_col3, export_col4 = st.columns(4)
        
        with export_col1:
            csv_events = simulator.get_event_log_dataframe().to_csv(index=False)
            st.download_button("Download Event Log (CSV)", csv_events, 
                              "event_log.csv", "text/csv")
        
        with export_col2:
            csv_packets = simulator.get_packet_table_dataframe().to_csv(index=False)
            st.download_button("Download Packets (CSV)", csv_packets,
                              "packets.csv", "text/csv")
        
        with export_col3:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                simulator.get_event_log_dataframe().to_excel(writer, sheet_name='Events', index=False)
                simulator.get_packet_table_dataframe().to_excel(writer, sheet_name='Packets', index=False)
                simulator.get_server_stats_dataframe().to_excel(writer, sheet_name='Servers', index=False)
                pd.DataFrame(performance_data).to_excel(writer, sheet_name='Summary', index=False)
            st.download_button("Download All (Excel)", buffer.getvalue(),
                              "simulation_results.xlsx", 
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        with export_col4:
            json_data = simulator.export_to_json()
            st.download_button("Download for Replay (JSON)", json_data,
                              "simulation_replay.json", "application/json")

    else:
        st.info("Configure parameters in the sidebar and click 'Run Simulation'")
        
        st.markdown("""
        ### About This Simulation
        
        This tool implements **Discrete Event Simulation (DES)** using the **Event-Scheduling Approach** to model network traffic for probability courses.
        
        #### Key Features:
        - **M/M/1 and M/M/c Systems**: Single or multiple server configurations
        - **Multiple Distributions**: Exponential, Normal, Uniform, Weibull, Gamma, Log-normal, Poisson
        - **Complete Event Tracing**: Full simulation event log with packet tracking
        - **Theoretical Comparison**: Compare simulated results with analytical M/M/c values
        - **Statistical Analysis**: Confidence intervals, percentiles, distribution statistics
        
        #### Queue Notation:
        - **M/M/1**: Single server, Markovian arrivals, Markovian service
        - **M/M/c**: c servers, Markovian arrivals, Markovian service
        - **Traffic Intensity**: ρ = λ/(cμ) where λ is arrival rate, μ is service rate
        """)


elif mode == "Comparative Analysis":
    st.header("Comparative Analysis Mode")
    st.markdown("Compare multiple simulation configurations side-by-side")
    
    num_configs = st.number_input("Number of Configurations", min_value=2, max_value=5, value=2)
    
    configs = []
    config_cols = st.columns(num_configs)
    
    for i, col in enumerate(config_cols):
        with col:
            st.subheader(f"Config {i+1}")
            
            servers = st.number_input("Servers", 1, 10, 1, key=f"comp_servers_{i}")
            arr_rate = st.number_input("Arrival Rate (λ)", 0.1, 20.0, 5.0, 0.1, key=f"comp_arr_{i}")
            svc_rate = st.number_input("Service Rate (μ)", 0.1, 30.0, 8.0, 0.1, key=f"comp_svc_{i}")
            sim_time = st.number_input("Sim Time", 10.0, 1000.0, 100.0, 10.0, key=f"comp_time_{i}")
            
            rho_i = arr_rate / (servers * svc_rate)
            st.metric("ρ", f"{rho_i:.3f}")
            
            configs.append(SimulationConfig(
                num_servers=servers,
                arrival_rate=arr_rate,
                service_rate=svc_rate,
                simulation_time=sim_time,
                random_seed=42 + i
            ))
    
    if st.button("Run Comparative Analysis", type="primary"):
        with st.spinner("Running simulations..."):
            results = run_comparative_analysis(configs)
        
        st.success("Analysis complete!")
        
        st.subheader("Comparison Results")
        
        comparison_df = pd.DataFrame({
            'Configuration': [f"Config {i+1}" for i in range(len(results))],
            'Servers': [r['num_servers'] for r in results],
            'Traffic (ρ)': [round(r['traffic_intensity'], 4) for r in results],
            'Avg Queue (Lq)': [round(r['average_queue_length'], 4) for r in results],
            'Avg Wait (Wq)': [round(r['average_waiting_time'], 4) for r in results],
            'Utilization': [round(r['server_utilization'], 4) for r in results],
            'Throughput': [round(r['throughput'], 4) for r in results],
            'Drop Rate': [f"{r['drop_rate']:.2%}" for r in results]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        st.subheader("Visual Comparison")
        
        metrics = ['average_queue_length', 'average_waiting_time', 'server_utilization', 'throughput']
        metric_names = ['Avg Queue Length', 'Avg Waiting Time', 'Server Utilization', 'Throughput']
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=metric_names)
        
        colors = px.colors.qualitative.Set1[:len(results)]
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            row = idx // 2 + 1
            col = idx % 2 + 1
            values = [r[metric] for r in results]
            configs_names = [f"Config {i+1}" for i in range(len(results))]
            
            fig.add_trace(
                go.Bar(x=configs_names, y=values, marker_color=colors, name=name, showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)


elif mode == "Statistical Validation":
    st.header("Statistical Validation Mode")
    st.markdown("Run multiple replications to compute confidence intervals")
    
    with st.sidebar:
        st.header("Validation Parameters")
        
        num_replications = st.number_input("Number of Replications", 5, 100, 30)
        confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
        
        st.markdown("---")
        st.subheader("Base Configuration")
        
        v_servers = st.number_input("Servers", 1, 10, 1, key="v_servers")
        v_arr_rate = st.number_input("Arrival Rate (λ)", 0.1, 20.0, 5.0, key="v_arr")
        v_svc_rate = st.number_input("Service Rate (μ)", 0.1, 30.0, 8.0, key="v_svc")
        v_sim_time = st.number_input("Simulation Time", 10.0, 1000.0, 100.0, key="v_time")
        v_seed = st.number_input("Base Seed", 1, 99999, 42, key="v_seed")
        
        st.markdown("---")
        run_validation = st.button("Run Validation", type="primary", use_container_width=True)
    
    if run_validation:
        config = SimulationConfig(
            num_servers=v_servers,
            arrival_rate=v_arr_rate,
            service_rate=v_svc_rate,
            simulation_time=v_sim_time,
            random_seed=v_seed
        )
        
        with st.spinner(f"Running {num_replications} replications..."):
            ci_results = run_replications(config, num_replications, confidence_level)
        
        st.success("Validation complete!")
        
        st.subheader(f"Confidence Intervals ({confidence_level*100:.0f}%)")
        
        ci_df = pd.DataFrame({
            'Metric': ['Queue Length (Lq)', 'System Length (L)', 'Waiting Time (Wq)',
                      'System Time (W)', 'Utilization', 'Throughput', 'Drop Rate'],
            'Mean': [round(ci_results[m]['mean'], 6) for m in 
                    ['Lq', 'L', 'Wq', 'W', 'utilization', 'throughput', 'drop_rate']],
            'Std Dev': [round(ci_results[m]['std'], 6) for m in 
                       ['Lq', 'L', 'Wq', 'W', 'utilization', 'throughput', 'drop_rate']],
            f'CI Lower ({confidence_level*100:.0f}%)': [round(ci_results[m]['lower'], 6) for m in 
                    ['Lq', 'L', 'Wq', 'W', 'utilization', 'throughput', 'drop_rate']],
            f'CI Upper ({confidence_level*100:.0f}%)': [round(ci_results[m]['upper'], 6) for m in 
                    ['Lq', 'L', 'Wq', 'W', 'utilization', 'throughput', 'drop_rate']],
            'CI Width': [round(ci_results[m]['upper'] - ci_results[m]['lower'], 6) for m in 
                        ['Lq', 'L', 'Wq', 'W', 'utilization', 'throughput', 'drop_rate']]
        })
        st.dataframe(ci_df, use_container_width=True, hide_index=True)
        
        theo = compute_mmc_theoretical(v_arr_rate, v_svc_rate, v_servers)
        if theo['stable']:
            st.subheader("Theoretical Comparison")
            st.markdown(f"Comparing simulated CI with theoretical M/M/{v_servers} values")
            
            theo_comparison = pd.DataFrame({
                'Metric': ['Queue Length (Lq)', 'System Length (L)', 
                          'Waiting Time (Wq)', 'System Time (W)'],
                'Theoretical': [round(theo['Lq'], 6), round(theo['L'], 6),
                               round(theo['Wq'], 6), round(theo['W'], 6)],
                'Simulated Mean': [round(ci_results['Lq']['mean'], 6), round(ci_results['L']['mean'], 6),
                                  round(ci_results['Wq']['mean'], 6), round(ci_results['W']['mean'], 6)],
                'In CI?': [
                    'Yes' if ci_results['Lq']['lower'] <= theo['Lq'] <= ci_results['Lq']['upper'] else 'No',
                    'Yes' if ci_results['L']['lower'] <= theo['L'] <= ci_results['L']['upper'] else 'No',
                    'Yes' if ci_results['Wq']['lower'] <= theo['Wq'] <= ci_results['Wq']['upper'] else 'No',
                    'Yes' if ci_results['W']['lower'] <= theo['W'] <= ci_results['W']['upper'] else 'No',
                ]
            })
            st.dataframe(theo_comparison, use_container_width=True, hide_index=True)
        
        st.subheader("Replication Distributions")
        
        fig = make_subplots(rows=2, cols=2, 
                           subplot_titles=['Queue Length (Lq)', 'Waiting Time (Wq)', 
                                          'Utilization', 'Throughput'])
        
        metrics_to_plot = ['Lq', 'Wq', 'utilization', 'throughput']
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 2 + 1
            col = idx % 2 + 1
            values = ci_results[metric]['values']
            
            fig.add_trace(
                go.Histogram(x=values, nbinsx=15, name=metric, showlegend=False),
                row=row, col=col
            )
            
            fig.add_vline(x=ci_results[metric]['mean'], line_dash="solid", 
                         line_color="red", row=row, col=col)
            fig.add_vline(x=ci_results[metric]['lower'], line_dash="dash",
                         line_color="blue", row=row, col=col)
            fig.add_vline(x=ci_results[metric]['upper'], line_dash="dash",
                         line_color="blue", row=row, col=col)
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Network Traffic Simulation - Event-Scheduling Approach | Probability Course Tool"
    "</div>",
    unsafe_allow_html=True
)