import random

from locust import HttpUser, between, task

QUESTIONS = [
    "ما هو سن الرشد في القانون المدني؟",
    "ما حكم بيع ملك الغير؟",
    "متى ينتقل الحق في الملكية في العقار؟",
    "ما هو الأثر الملزم للعقد؟",
    "What is the rule on the binding force of a contract?",
]


class RagUser(HttpUser):
    wait_time = between(1, 3)

    @task(9)
    def ask(self):
        question = random.choice(QUESTIONS)
        with self.client.post(
            "/ask",
            json={"question": question},
            name="/ask",
            timeout=60,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"status {resp.status_code}")
            elif not resp.json().get("answer"):
                resp.failure("empty answer")

    @task(1)
    def health(self):
        self.client.get("/health", name="/health")
