from langflow import CustomComponent
from langchain.tools import Tool
import numpy as np
import joblib


class QoSClassifierComponent(CustomComponent):
    display_name = "5G QoS Classifier (RF)"
    description = "Predicts network degradation state using a Random Forest model."

    def build_config(self):
        return {
            "model_path": {
                "display_name": "RF Model Path",
                "info": "Absolute path to the rf_model.joblib file",
            },
            "encoder_path": {
                "display_name": "Label Encoder Path",
                "info": "Absolute path to the label_enc.joblib file",
            },
        }

    def build(self, model_path: str, encoder_path: str) -> Tool:
        try:
            rf_model = joblib.load(model_path)
            label_enc = joblib.load(encoder_path)
        except Exception as e:
            raise ValueError(f"Failed to load models: {e}")

        def predict_degradation(telemetry_str):
            try:
                values = [float(x.strip()) for x in telemetry_str.split(",")]
                sample = np.array(values).reshape(1, -1)
                pred_idx = rf_model.predict(sample)[0]
                error_state = label_enc.inverse_transform([pred_idx])[0]
                return f"Predicted Network State: {error_state}"
            except Exception as e:
                return f"Error: {e}. Ensure format: 'Throughput, Latency, Reliability, Density, Mobility'"

        return Tool(
            name="Predict_Degradation",
            func=predict_degradation,
            description="Predicts 5G QoS degradation. Input MUST be comma-separated: 'Throughput, Latency, Reliability, Density, Mobility'.",
        )


class LinkStabilityComponent(CustomComponent):
    display_name = "Link Stability Clusterer (KMeans)"
    description = "Assesses link stability regime based on telemetry."

    def build_config(self):
        return {
            "model_path": {
                "display_name": "KMeans Model Path",
                "info": "Absolute path to the kmeans_model.joblib file",
            },
            "scaler_path": {
                "display_name": "Scaler Path",
                "info": "Absolute path to the scaler.joblib file",
            },
        }

    def build(self, model_path: str, scaler_path: str) -> Tool:
        try:
            kmeans = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            raise ValueError(f"Failed to load models: {e}")

        def assess_link_stability(telemetry_str):
            try:
                values = [float(x.strip()) for x in telemetry_str.split(",")]
                sample = np.array(values).reshape(1, -1)
                scaled_sample = scaler.transform(sample)
                cluster = kmeans.predict(scaled_sample)[0]

                regimes = {
                    0: "Stable",
                    1: "Transitional (Warning)",
                    2: "Unstable (Degraded)",
                    3: "Highly Congested",
                }
                return f"Link Stability Cluster: {cluster} - {regimes.get(cluster, 'Unknown')}"
            except Exception as e:
                return f"Error assessing link stability: {e}"

        return Tool(
            name="Assess_Link_Stability",
            func=assess_link_stability,
            description="Analyzes the link stability regime based on telemetry. Input MUST be comma-separated: 'Throughput, Latency, Reliability, Density, Mobility'.",
        )
