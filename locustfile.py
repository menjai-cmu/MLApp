from locust import HttpUser, task, between
import os

class ChurnPredictionUser(HttpUser):
    """
    Simulates user interactions with the churn prediction API.
    """
    # Specify the base URL for the API, including the correct schema
    host = "http://127.0.0.1:8080"  # Ensure the FastAPI server is running on this address and port

    # Simulate user wait times between tasks (1 to 5 seconds)
    wait_time = between(1, 5)

    @task
    def predict_churn(self):
        """
        Sends a POST request to the /predict-churn endpoint
        with a CSV file for churn prediction.
        """
        # Replace with the correct path to your CSV file
        csv_path = "/ProjectML/FinalML/Telco_customer_churn_Testing.csv"  # Update with actual path

        # Ensure the file exists before sending the request
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                response = self.client.post(
                    "/predict-churn",
                    files={"file": ("test.csv", f, "text/csv")},
                    data={"min-churn": 50, "max-churn": 100}  # Include this line if supported by your API
                )
                if response.status_code == 200:
                    print("Prediction successful:", response.json())
                else:
                    print(f"Error: {response.status_code}, Response: {response.text}")
        else:
            print(f"Error: CSV file not found at {csv_path}. Please check the file path.")

    @task
    def access_docs(self):
        """
        Simulates a GET request to access the Swagger UI documentation.
        """
        response = self.client.get("/docs")
        if response.status_code == 200:
            print("Accessed Swagger UI successfully.")
        else:
            print(f"Error accessing /docs: {response.status_code}, Response: {response.text}")

    @task
    def health_check(self):
        """
        Simulates a GET request to the health check endpoint.
        """
        response = self.client.get("/health")
        if response.status_code == 200:
            print("Health check passed.")
        else:
            print(f"Health check failed: {response.status_code}, Response: {response.text}")

        

