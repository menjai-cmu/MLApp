from locust import HttpUser, task, between
import os

class ChurnPredictionUser(HttpUser):
    """
    Simulates user interactions with the churn prediction API.
    """
    # Explicitly set the base host
    host = "http://localhost:8080/"  # make sure this is your api 

    # Simulate user wait times between tasks (1 to 5 seconds)
    wait_time = between(1, 5)

    @task
    def predict_churn(self):
        """
        Sends a POST request to the /predict-churn endpoint
        with a CSV file and churn probability range.
        """
        csv_path = "/Users/....../LocalHost/Telco_customer_churn_Testing.csv"  # Replace with actual CSV file path
        
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                self.client.post(
                    "/predict-churn",
                    files={"file": ("test.csv", f, "text/csv")},
                    data={"min-churn": 50, "max-churn": 100} # You can comment this line if you are using the old version 
                )
        else:
            print(f"Error: CSV file not found at {csv_path}. Please check the file path.")

    @task
    def access_docs(self):
        """
        Simulates a GET request to access the Swagger UI documentation.
        """
        self.client.get("/docs")

    @task
    def health_check(self):
        """
        Simulates a GET request to the root endpoint for health checks.
        """
        self.client.get("/")
