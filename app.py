import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

st.set_page_config(page_title="TCP Simulation: Slow Start, CUBIC & AIMD", layout="wide")

st.title("Advanced TCP Flow & Buffer Simulator")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Network Parameters")

link_bw_mbps = st.sidebar.slider("Bottleneck Bandwidth (Mbps)", 10, 10000, 1000, step=10)
rtt_ms = st.sidebar.slider("Base RTT (ms)", 5, 200, 50)
packet_size_bytes = 1500

# BDP Calculation
rtt_sec = rtt_ms / 1000.0
packets_per_sec = (link_bw_mbps * 1_000_000) / (packet_size_bytes * 8)
bdp_packets = int(packets_per_sec * rtt_sec)

st.sidebar.markdown(f"**Calculated BDP:** ~`{bdp_packets}` packets")

st.sidebar.header("TCP Parameters")
algo = st.sidebar.selectbox("Congestion Control Algorithm", ["CUBIC", "AIMD (Reno)"])

# iOS Default Window Calculation
# 131,072 bytes / 1500 bytes per packet = ~87 packets
ios_default_window_pkts = int(131072 / packet_size_bytes)

max_window_packets = st.sidebar.number_input(
    "Max Window Limit (rwnd in pkts)", 
    value=ios_default_window_pkts, 
    step=10
)

num_flows = st.sidebar.slider("Number of TCP Flows", 1, 20, 3)

# Calculate BDP / sqrt(N) for the default buffer size
default_buffer = int(bdp_packets / np.sqrt(num_flows)) if num_flows > 0 else bdp_packets
buffer_size = st.sidebar.slider("Buffer Size (packets)", 0, bdp_packets * 5, default_buffer)
sim_steps = st.sidebar.slider("Simulation Steps (RTTs)", 100, 1000, 300)

def pkts_to_mbps(pkts):
    return (pkts * packet_size_bytes * 8) / 1_000_000 / rtt_sec

# --- SIMULATION LOGIC ---
def run_simulation(bdp, buffer_capacity, flows_count, steps, algorithm, max_rwnd):
    cwnds = np.ones(flows_count) 
    ssthresh = np.full(flows_count, max_rwnd, dtype=float)
    
    w_max = np.zeros(flows_count)
    time_since_drop = np.zeros(flows_count)
    C_cubic = 0.4
    beta_cubic = 0.7
    
    history = []
    
    for step in range(steps):
        # Enforce max window limit (rwnd cap)
        cwnds = np.minimum(cwnds, max_rwnd)
        total_inflight = np.sum(cwnds)
        
        # Calculate Queue and Goodput
        if total_inflight <= bdp:
            queue_occupancy = 0
            goodput_packets = total_inflight
        else:
            queue_occupancy = total_inflight - bdp
            goodput_packets = bdp 
            
        # Check for Buffer Overflow
        dropped = False
        if queue_occupancy > buffer_capacity:
            dropped = True
            queue_occupancy = buffer_capacity 
            
            for i in range(flows_count):
                if random.random() < 0.5: 
                    if algorithm == "CUBIC":
                        w_max[i] = cwnds[i]
                        ssthresh[i] = max(2.0, cwnds[i] * beta_cubic)
                        cwnds[i] = ssthresh[i]
                        time_since_drop[i] = 0
                    else: 
                        ssthresh[i] = max(2.0, cwnds[i] / 2.0)
                        cwnds[i] = ssthresh[i]
        else:
            # Growth Phase
            for i in range(flows_count):
                if cwnds[i] < ssthresh[i]:
                    cwnds[i] *= 2
                    if cwnds[i] > ssthresh[i]:
                        cwnds[i] = ssthresh[i]
                else:
                    if algorithm == "CUBIC":
                        time_since_drop[i] += rtt_sec
                        K = np.cbrt((w_max[i] * (1 - beta_cubic)) / C_cubic)
                        w_target = C_cubic * (time_since_drop[i] - K)**3 + w_max[i]
                        cwnds[i] = max(cwnds[i] + 1/cwnds[i], w_target)
                    else: 
                        cwnds[i] += 1
            
        state = {
            "Step": step,
            "Total Inflight": total_inflight,
            "Queue Occupancy": queue_occupancy,
            "Throughput (Mbps)": pkts_to_mbps(total_inflight), 
            "Goodput (Mbps)": pkts_to_mbps(goodput_packets),   
            "Drop Event": 1 if dropped else 0
        }
        
        for i in range(flows_count):
            state[f"Flow {i+1} cwnd"] = cwnds[i]
            
        history.append(state)
        
    return pd.DataFrame(history)

df = run_simulation(bdp_packets, buffer_size, num_flows, sim_steps, algo, max_window_packets)

# --- VISUALIZATION ---
st.header("Simulation Results")

# Calculate metrics
avg_goodput = df['Goodput (Mbps)'].mean()
avg_utilization = (avg_goodput / link_bw_mbps) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Average Link Utilization", f"{avg_utilization:.2f} %")
col2.metric("Total Drop Events", df["Drop Event"].sum())
col3.metric("Peak Queue Occupancy", f"{df['Queue Occupancy'].max():.0f} pkts")

# Plot 1: Throughput vs Goodput
st.subheader("Throughput vs. Goodput")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["Step"], df["Throughput (Mbps)"], label="Throughput (Attempted Send)", color="blue", alpha=0.6)
ax1.plot(df["Step"], df["Goodput (Mbps)"], label="Goodput (Actual Delivery)", color="green", linewidth=2)
ax1.axhline(y=link_bw_mbps, color="red", linestyle="--", alpha=0.5, label="Link Capacity")
ax1.set_xlabel("Time (RTT Steps)")
ax1.set_ylabel("Bandwidth (Mbps)")
ax1.legend()
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# Plot 2: Queue Occupancy
st.subheader("Router Queue Occupancy")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.fill_between(df["Step"], df["Queue Occupancy"], color="orange", alpha=0.5, label="Queue Occupancy")
ax2.axhline(y=buffer_size, color="red", linestyle="--", alpha=0.5, label="Buffer Capacity")
ax2.set_xlabel("Time (RTT Steps)")
ax2.set_ylabel("Packets in Queue")
ax2.legend()
st.pyplot(fig2)

# Plot 3: Individual TCP Flows (cwnd)
st.subheader(f"TCP Congestion Windows ({algo})")
fig3, ax3 = plt.subplots(figsize=(10, 4))
for i in range(num_flows):
    ax3.plot(df["Step"], df[f"Flow {i+1} cwnd"], label=f"Flow {i+1}", alpha=0.8)
ax3.axhline(y=max_window_packets, color="purple", linestyle=":", alpha=0.5, label="Max Window (rwnd)")
ax3.set_xlabel("Time (RTT Steps)")
ax3.set_ylabel("cwnd (Packets)")
ax3.legend()
ax3.grid(True, alpha=0.3)
st.pyplot(fig3)
