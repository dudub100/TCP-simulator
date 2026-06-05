import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

st.set_page_config(page_title="TCP Buffer & Goodput Simulator", layout="wide")

st.title("TCP Flows & Buffer Sizing Simulator")
st.markdown("""
This macroscopic simulation models multiple TCP flows sharing a bottleneck link. 
It uses a time-step model (1 step = 1 RTT) implementing standard AIMD (Additive Increase, Multiplicative Decrease) congestion control.
""")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Simulation Parameters")

link_bw_mbps = st.sidebar.slider("Bottleneck Bandwidth (Mbps)", 1, 100, 10)
rtt_ms = st.sidebar.slider("Base RTT (ms)", 10, 200, 50)
packet_size_bytes = 1500

# Calculate BDP in packets
# Mbps -> bits per second -> bytes per sec -> packets per sec
packets_per_sec = (link_bw_mbps * 1_000_000) / (packet_size_bytes * 8)
bdp_packets = int(packets_per_sec * (rtt_ms / 1000.0))

st.sidebar.markdown(f"**Calculated BDP:** ~`{bdp_packets}` packets")

buffer_size = st.sidebar.slider("Buffer Size (packets)", 0, bdp_packets * 5, bdp_packets)
num_flows = st.sidebar.slider("Number of TCP Flows", 1, 20, 3)
sim_steps = st.sidebar.slider("Simulation Steps (RTTs)", 100, 1000, 300)

# --- SIMULATION LOGIC ---
def run_simulation(bdp, buffer_capacity, flows_count, steps):
    # Initialize flows: start with a cwnd of 1
    cwnds = np.ones(flows_count) 
    
    history = []
    
    for step in range(steps):
        total_inflight = np.sum(cwnds)
        
        # Calculate Queue and Goodput
        if total_inflight <= bdp:
            queue_occupancy = 0
            # Link is not fully utilized
            utilization = total_inflight / bdp
            goodput_packets = total_inflight
        else:
            queue_occupancy = total_inflight - bdp
            # Link is fully utilized
            utilization = 1.0
            goodput_packets = bdp
            
        # Check for Buffer Overflow (Drop-Tail)
        dropped = False
        if queue_occupancy > buffer_capacity:
            dropped = True
            queue_occupancy = buffer_capacity # Cap queue at max
            
            # TCP Reaction: Multiplicative Decrease
            # In a basic drop-tail, multiple flows might experience drops.
            # We randomly select flows to halve their window to simulate desynchronization.
            for i in range(flows_count):
                if random.random() < 0.5: # 50% chance a flow loses a packet in the drop event
                    cwnds[i] = max(1, cwnds[i] / 2.0)
        else:
            # TCP Reaction: Additive Increase
            # Each flow increases its cwnd by 1 packet per RTT
            cwnds += 1
            
        # Record state
        state = {
            "Step": step,
            "Total Inflight": total_inflight,
            "Queue Occupancy": queue_occupancy,
            "Link Utilization (%)": utilization * 100,
            "Goodput (Packets/RTT)": goodput_packets,
            "Drop Event": 1 if dropped else 0
        }
        
        for i in range(flows_count):
            state[f"Flow {i+1} cwnd"] = cwnds[i]
            
        history.append(state)
        
    return pd.DataFrame(history)

df = run_simulation(bdp_packets, buffer_size, num_flows, sim_steps)

# --- VISUALIZATION ---
st.header("Simulation Results")

# Metrics
avg_utilization = df["Link Utilization (%)"].mean()
drop_events = df["Drop Event"].sum()
max_queue = df["Queue Occupancy"].max()

col1, col2, col3 = st.columns(3)
col1.metric("Avg Link Utilization", f"{avg_utilization:.2f}%")
col2.metric("Total Drop Events", drop_events)
col3.metric("Peak Queue Occupancy", f"{max_queue:.0f} pkts")

# Plot 1: Goodput and Queue Size
st.subheader("Link Metrics over Time")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(df["Step"], df["Goodput (Packets/RTT)"], label="Goodput", color="green")
ax1.plot(df["Step"], df["Queue Occupancy"], label="Queue Occupancy", color="orange", alpha=0.7)
ax1.axhline(y=bdp_packets, color="green", linestyle="--", alpha=0.5, label="Max Goodput (BDP)")
ax1.axhline(y=buffer_size, color="red", linestyle="--", alpha=0.5, label="Buffer Capacity")
ax1.set_xlabel("Time (RTT Steps)")
ax1.set_ylabel("Packets")
ax1.legend()
st.pyplot(fig1)

# Plot 2: Individual TCP Flows (cwnd)
st.subheader("TCP Congestion Windows (cwnd)")
fig2, ax2 = plt.subplots(figsize=(10, 4))
for i in range(num_flows):
    ax2.plot(df["Step"], df[f"Flow {i+1} cwnd"], label=f"Flow {i+1}", alpha=0.8)
ax2.set_xlabel("Time (RTT Steps)")
ax2.set_ylabel("cwnd (Packets)")
ax2.legend()
st.pyplot(fig2)

with st.expander("Show Raw Data"):
    st.dataframe(df)
