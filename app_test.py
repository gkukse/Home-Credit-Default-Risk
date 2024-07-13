from locust import HttpUser, task, constant_throughput

# Define test json request
test_application = {
    "EXT_SOURCE_1": 0.543,
    "NAME_EDUCATION_TYPE": 1,
    "AMT_CREDIT": 3,
    "Max_DURATION_DUE_VERSION": 0,
    "YEARS_EMPLOYED": 0.4,
    "EXT_SOURCE_3": 0.01,
    "YEARS_BIRTH": 70,
    "Max_DURATION_DECISION_DRAWING": 367912,
    "EXT_SOURCE_2": 0.681,
    "Min_RATIO_GOODS_PRICE_CREDIT": 1.111,
    "AVG_Risk_Score": 3,
    "Min_DURATION_DECISION_DRAWING": 1,
    "CODE_GENDER_F": 0,
    "YEARS_LAST_PHONE_CHANGE": 0.1,
}


class CreditScoringUser(HttpUser):
    # Means that a user will send 1 request per second
    wait_time = constant_throughput(1)
    
    # Task to be performed (send data & get response)
    @task
    def predict(self):
        self.client.post(
            "/predict",
            json=test_application,
            timeout=1,
        )