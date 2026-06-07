import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

st.set_page_config(page_title="TCP Elephants & Mice Simulator", layout="wide")

st.title("Dynamic Traffic: Elephant vs. Mice Flows")
st.markdown("""
Real networks are not steady-state. They are a mix of long-lived **Elephant Flows** (large downloads) 
and short-lived, highly bursty **Mice Flows** (web API calls). This simulation introduces dynamic flow arrivals.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Network Parameters")
nic_bw_mbps = st.sidebar.slider("Host NIC Line Rate (Mbps)", 100, 25000, 10000, step=100)
link_bw_mbps = st.sidebar.slider("Bottleneck Avg Rate (Mbps)", 10, 10000, 1000, step=10)

if link_bw_mbps > nic_bw_mbps:
    link_bw_mbps = nic_bw_mbps

rtt_ms = st.sidebar.slider("Base RTT (ms)", 5, 200, 50)
packet_size_bytes = 1500

# 1. Calculate BDP
rtt_sec = rtt_ms / 1000.0
packets_per_sec = (link_bw_mbps * 1_000_000) / (packet_size_bytes * 8)
bdp_packets = int(packets_per_sec * rtt_sec)
bdp_mbytes = bdp_packets * packet_size_bytes /1e6

# Print BDP on the Sidebar
st.sidebar.markdown(f"### **Calculated BDP:** `{bdp_packets}` packets  {bdp_mbytes} MB")
st.sidebar.markdown("---")

# 2. Get Traffic Profile FIRST
st.sidebar.header("Traffic Profile (Elephants & Mice)")
num_elephants = st.sidebar.number_input("Elephant Flows (Always Active)", 0, 10, 1)
num_mice_pool = st.sidebar.number_input("Mice Flow Pool (Max Concurrent)", 0, 50, 15)

mice_prob = st.sidebar.slider("Mouse Arrival Probability (per RTT)", 0.0, 1.0, 0.2)
mice_size_pkts = st.sidebar.slider("Mouse File Size (Avg Packets)", 10, 500, 100)

st.sidebar.markdown("---")

# 3. Calculate Stanford Buffer Default and Render Slider
st.sidebar.header("Router Settings")
if num_elephants > 0:
    stanford_default = int(bdp_packets / np.sqrt(num_elephants))
else:
    # Fallback if Elephants = 0 to prevent division by zero
    stanford_default = bdp_packets

buffer_size_Mbytes = st.sidebar.slider(
    "Router Buffer Size (MBytes)", 
    min_value=0.0, 
    max_value=float(bdp_packets * packet_size_bytes / 1e6 * 5), 
    value=float(stanford_default * packet_size_bytes / 1e6),
    step=0.1  # Allows you to smoothly scroll through fractions of a Megabyte
)
buffer_size = int(buffer_size_Mbytes * 1E6 / packet_size_bytes)
st.sidebar.markdown("---")

# 4. TCP Parameters
st.sidebar.header("TCP Parameters")
algo = st.sidebar.selectbox("Congestion Control", ["CUBIC", "AIMD (Reno)"])
sim_steps = st.sidebar.slider("Simulation Steps (RTTs)", 100, 1000, 300)
max_window_packets = st.sidebar.number_input("Max Window Limit (rwnd in pkts)", value=bdp_packets * 2, step=100)

def pkts_to_mbps(pkts):
    return (pkts * packet_size_bytes * 8) / 1_000_000 / rtt_sec

# --- SIMULATION LOGIC ---
def run_simulation():
    total_flows = num_elephants + num_mice_pool
    
    # State tracking
    cwnds = np.zeros(total_flows)
    active = np.array([True]*num_elephants + [False]*num_mice_pool)
    remaining_pkts = np.array([np.inf]*num_elephants + [0.0]*num_mice_pool)
    
    # Initialize Elephants
    for i in range(num_elephants):
        cwnds[i] = 10 # Linux/iOS default initcwnd
        
    ssthresh = np.full(total_flows, max_window_packets, dtype=float)
    w_max = np.zeros(total_flows)
    time_since_drop = np.zeros(total_flows)
    C_cubic = 0.4
    beta_cubic = 0.7
    burst_factor = max(0, 1.0 - (link_bw_mbps / nic_bw_mbps))
    
    history = []
    
    for step in range(sim_steps):
        # 1. Handle Mice Arrivals
        for i in range(num_elephants, total_flows):
            if not active[i]:
                if random.random() < mice_prob:
                    active[i] = True
                    # Randomize file size around the average
                    remaining_pkts[i] = max(10, random.gauss(mice_size_pkts, mice_size_pkts/4))
                    cwnds[i] = 10 # Start with initcwnd
                    ssthresh[i] = max_window_packets
                    w_max[i] = 0
                    time_since_drop[i] = 0

        # Enforce max window
        cwnds = np.minimum(cwnds, max_window_packets)
        
        # Calculate in-flight packets (ONLY from active flows)
        total_inflight = np.sum(cwnds[active]) if np.any(active) else 0
        
        # Fluid vs Burst queue math
        if total_inflight <= bdp_packets:
            fluid_queue = 0
            goodput_packets = total_inflight
        else:
            fluid_queue = total_inflight - bdp_packets
            goodput_packets = bdp_packets 
            
        burst_spike = total_inflight * burst_factor
        instant_queue = fluid_queue + burst_spike
            
        # 2. Check Drops
        dropped = False
        if instant_queue > buffer_size:
            dropped = True
            overflow = instant_queue - buffer_size
            goodput_packets = max(0, goodput_packets - overflow)
            
            for i in range(total_flows):
                if active[i] and random.random() < 0.5: 
                    if algo == "CUBIC":
                        w_max[i] = cwnds[i]
                        ssthresh[i] = max(2.0, cwnds[i] * beta_cubic)
                        cwnds[i] = ssthresh[i]
                        time_since_drop[i] = 0
                    else: 
                        ssthresh[i] = max(2.0, cwnds[i] / 2.0)
                        cwnds[i] = ssthresh[i]
        else:
            # 3. Growth Phase
            for i in range(total_flows):
                if active[i]:
                    if cwnds[i] < ssthresh[i]: # Slow Start
                        cwnds[i] *= 2
                        if cwnds[i] > ssthresh[i]: cwnds[i] = ssthresh[i]
                    else: # Congestion Avoidance
                        if algo == "CUBIC":
                            time_since_drop[i] += rtt_sec
                            K = np.cbrt((w_max[i] * (1 - beta_cubic)) / C_cubic)
                            w_target = C_cubic * (time_since_drop[i] - K)**3 + w_max[i]
                            cwnds[i] = max(cwnds[i] + 1/cwnds[i], w_target)
                        else: 
                            cwnds[i] += 1
        
        # 4. Data Transmission & Mice Departures
        for i in range(total_flows):
            if active[i]:
                remaining_pkts[i] -= cwnds[i]
                if remaining_pkts[i] <= 0 and i >= num_elephants:
                    active[i] = False
                    cwnds[i] = 0
            
        state = {
            "Step": step,
            "Active Flows": np.sum(active),
            "Fluid Queue (Avg)": fluid_queue,
            "Instant Queue (Burst)": instant_queue,
            "Throughput (Mbps)": pkts_to_mbps(total_inflight), 
            "Goodput (Mbps)": pkts_to_mbps(goodput_packets),   
            "Drop Event": 1 if dropped else 0
        }
        history.append(state)
        
    return pd.DataFrame(history)

df = run_simulation()

# --- VISUALIZATION ---
st.header("Simulation Results: Dynamic Traffic Profile")

col1, col2, col3 = st.columns(3)
col1.metric("Avg Goodput", f"{df['Goodput (Mbps)'].mean():.2f} Mbps")
col2.metric("Total Drop Events", df["Drop Event"].sum())
col3.metric("Peak Active Flows", df["Active Flows"].max())

# Plot 1: Active Connections over Time
st.subheader("Active TCP Connections")
fig1, ax1 = plt.subplots(figsize=(10, 2))
ax1.plot(df["Step"], df["Active Flows"], color="purple", drawstyle="steps-mid")
ax1.set_xlabel("Time (RTT Steps)")
ax1.set_ylabel("Concurrent Flows")
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# Plot 2: Throughput vs Goodput
st.subheader("Throughput vs. Goodput")
fig2, ax2 = plt.subplots(figsize=(10, 3))
ax2.plot(df["Step"], df["Throughput (Mbps)"], label="Attempted Throughput", color="blue", alpha=0.6)
ax2.plot(df["Step"], df["Goodput (Mbps)"], label="Actual Goodput", color="green", linewidth=2)
ax2.axhline(y=link_bw_mbps, color="red", linestyle="--", alpha=0.5, label="Bottleneck Capacity")
ax2.set_xlabel("Time (RTT Steps)")
ax2.set_ylabel("Bandwidth (Mbps)")
ax2.legend()
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# Plot 3: Router Buffer Spikes
st.subheader("Router Buffer: The Impact of Mice Bursting")
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(df["Step"], df["Fluid Queue (Avg)"], color="orange", linewidth=2, label="Fluid Queue")
ax3.fill_between(df["Step"], df["Fluid Queue (Avg)"], df["Instant Queue (Burst)"], color="purple", alpha=0.3, label="Microburst Queue Spike")
ax3.axhline(y=buffer_size, color="red", linestyle="--", alpha=0.8, label="Buffer Capacity")
ax3.set_xlabel("Time (RTT Steps)")
ax3.set_ylabel("Packets in Queue")
ax3.legend()
st.pyplot(fig3)
