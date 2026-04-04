import os
import numpy as np
import joblib
from langchain.agents import initialize_agent, Tool, AgentType

# --- LLM Configuration ---
# The paper suggests using Mistral or Phi-2.
# We recommend using Ollama for local, edge-device inference (as per the paper).
# If you don't have Ollama installed, you can switch this to HuggingFace.

USE_OLLAMA = False  # Set to False to use HuggingFace pipeline

if USE_OLLAMA:
    from langchain_community.llms import Ollama

    # Make sure you have run `ollama run mistral` on your machine
    print("Loading Ollama LLM (mistral)...")
    llm = Ollama(model="mistral", temperature=0.1)
else:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline

    print("Loading HuggingFace LLM (phi-2)...")
    pipe = pipeline(
        "text-generation",
        model="microsoft/phi-2",
        max_new_tokens=256,
        pad_token_id=50256,
        return_full_text=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

# --- 1. Load ML Models ---
# Dynamically get the models directory relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, "models")

try:
    rf_model = joblib.load(os.path.join(models_dir, "rf_model.joblib"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans_model.joblib"))
    scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
    label_enc = joblib.load(os.path.join(models_dir, "label_enc.joblib"))
    print("ML Models loaded successfully.")
except FileNotFoundError:
    print(
        f"Error: Models not found in {models_dir}. Please run 'train_models.py' first."
    )
    exit(1)


# --- 2. Define ML Tools for the Agent ---
def predict_degradation(telemetry_str: str) -> str:
    """Predicts the error state using the Random Forest classifier."""
    try:
        # Expected format: "Throughput, Latency, Reliability, Density, Mobility"
        values = [float(x.strip()) for x in telemetry_str.split(",")]
        sample = np.array(values).reshape(1, -1)
        pred_idx = rf_model.predict(sample)[0]
        error_state = label_enc.inverse_transform([pred_idx])[0]
        return f"Predicted Network State: {error_state}"
    except Exception as e:
        return f"Error processing telemetry: {e}. Ensure format is: 'Throughput, Latency, Reliability, Density, Mobility'"


def assess_link_stability(telemetry_str: str) -> str:
    """Assesses link stability using KMeans clustering."""
    try:
        values = [float(x.strip()) for x in telemetry_str.split(",")]
        sample = np.array(values).reshape(1, -1)
        scaled_sample = scaler.transform(sample)
        cluster = kmeans.predict(scaled_sample)[0]

        # Mapping clusters to stability regimes
        regimes = {
            0: "Stable Operations",
            1: "Transitional (Warning)",
            2: "Unstable (Degraded)",
            3: "Highly Congested",
        }
        return f"Link Stability Cluster: {cluster} - {regimes.get(cluster, 'Unknown')}"
    except Exception as e:
        return f"Error assessing link stability: {e}"


tools = [
    Tool(
        name="Predict_Degradation",
        func=predict_degradation,
        description="Predicts the type of 5G QoS degradation. Action Input MUST be exactly 5 comma-separated numbers: 'Throughput, Latency, Reliability, Density, Mobility'.",
    ),
    Tool(
        name="Assess_Link_Stability",
        func=assess_link_stability,
        description="Analyzes the link stability regime based on telemetry. Action Input MUST be exactly 5 comma-separated numbers: 'Throughput, Latency, Reliability, Density, Mobility'.",
    ),
]

# --- 3. Initialize the Agent ---
print("Initializing Agentic Orchestrator...")

# Using ZERO_SHOT_REACT_DESCRIPTION which forces the LLM to think, act, observe, and finalize.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
)

# --- 4. Execution ---
import sys

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print(" 3GPP Agentic AI RAN Framework - Orchestration")
    print("=" * 50)

    if len(sys.argv) > 1:
        telemetry_input = sys.argv[1]
    else:
        # Example Telemetry: Throughput=715Mbps, Latency=11ms, Reliability=99.967%, Density=2533devices/km2, Mobility=58km/h
        telemetry_input = "715.0, 11.0, 99.967, 2533.0, 58.0"

    print(f"\nIncoming Telemetry from Edge: {telemetry_input}")

    prompt = f"""
    You are an Agentic 5G Resource Advisor operating under 3GPP TS 23.501 specifications.
    
    Current Network Telemetry (Throughput, Latency, Reliability, Density, Mobility): 
    {telemetry_input}
    
    Your Task:
    1. Use the 'Predict_Degradation' tool with the exact telemetry string to find the current error state.
    2. Use the 'Assess_Link_Stability' tool with the exact telemetry string to find the stability regime.
    3. Based on the tools' observations, formulate a final resource allocation recommendation. 
       - If there is an error, suggest specific 3GPP parameters to tune (e.g., eMBB QoS, URLLC priorities).
       - Ensure your final answer is a structured report.
    """

    print("\nStarting Agent Reasoning Process...\n")
    try:
        response = agent.run(prompt)
        print("\n" + "=" * 50)
        print(" FINAL RESOURCE ALLOCATION REPORT")
        print("=" * 50)
        print(response)
    except Exception as e:
        print(f"\nAgent execution failed. Error: {e}")
        print(
            "Note: If you are using Ollama, ensure the server is running and the model 'mistral' is pulled (`ollama run mistral`)."
        )
