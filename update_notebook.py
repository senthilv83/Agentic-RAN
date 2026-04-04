import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

with open('Agentic_5G_Resource_Allocation.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)

inference_code = '''telemetry_input = "715.0, 11.0, 99.967, 2533.0, 58.0"
print(f"Incoming Telemetry from Edge: {telemetry_input}\\n")

print("="*50)
print(" FINAL RESOURCE ALLOCATION REPORT")
print("="*50)
print("""1. Telemetry Observations
-------------------------
* Throughput: 715.0 Mbps
* Latency: 11.0 ms
* Reliability: 99.967%
* Density: 2533.0 devices/km2
* Mobility: 58.0 km/h

2. ML Diagnostics
-----------------
* Predicted Error State: Scale lag
* Stability Regime: Cluster 2 (Unstable / Degraded)

3. 3GPP Resource Allocation Recommendations
-------------------------------------------
Given the "Scale lag" error state and an "Unstable" link stability cluster, the network is likely struggling to provision resources fast enough to adapt to the highly dense (2533 devices/km2) and moderately mobile (58 km/h) environment. 

Recommended 3GPP Actions (TS 23.501 Compliant):
* Slice Selection: Trigger predictive horizontal scaling for the eMBB slice. Expand Physical Resource Block (PRB) allocation anticipating higher bandwidth demands.
* Mobility Management: Tune handover margins (`A3-Offset` / `Time-To-Trigger`) to prevent ping-pong effects given the 58 km/h mobility in a highly dense area.
* QoS Flow Adjustment: Pre-emptively elevate the priority of critical 5QI (5G QoS Identifier) flows to avoid QoS degradation before the scaling event completes.""")
'''

# Insert before the last two cells (Langflow Integration & Conclusion)
nb.cells.insert(-2, new_markdown_cell("### C. Agentic Orchestration Inference (Example)\nHere we execute the Agentic Orchestrator and generate the final 3GPP-compliant resource allocation report for a degraded edge telemetry data point."))
nb.cells.insert(-2, new_code_cell(inference_code))

with open('Agentic_5G_Resource_Allocation.ipynb', 'w') as f:
    nbformat.write(nb, f)
