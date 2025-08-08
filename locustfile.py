from locust import HttpUser, task, between

class PenguinUser(HttpUser):
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks

    @task
    def predict(self):
        payload = {
            "bill_length_mm": 45.0,
            "bill_depth_mm": 14.0,
            "flipper_length_mm": 210,
            "body_mass_g": 5000,
            "year": 2007,
            "sex": "male",
            "island": "Torgersen"
        }
        headers = {"Content-Type": "application/json"}
        self.client.post("/predict", json=payload, headers=headers)
