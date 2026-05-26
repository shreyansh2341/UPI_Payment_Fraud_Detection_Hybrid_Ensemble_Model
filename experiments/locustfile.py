from locust import HttpUser, task, between

class FraudDetectionUser(HttpUser):
    wait_time = between(0.1, 1.0)
    
    @task(3)
    def check_health(self):
        self.client.get("/health")
        
    @task(2)
    def view_dashboard(self):
        self.client.get("/dashboard")
        
    @task(1)
    def view_analytics(self):
        self.client.get("/analytics/7")

    @task(1)
    def single_inference(self):
        payload = {
            "transaction_type": "paysim",
            "tabular_features": [0.0] * 18,
            "model_version": "v5"
        }
        self.client.post("/predict", json=payload)
