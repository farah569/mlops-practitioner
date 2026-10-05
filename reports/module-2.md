# Module 2 Lab Report: Tracking, Versioning, and Automation

## 1. MLflow Experiment Tracking
We successfully tracked 3 model families (Linear Regression, PyTorch MLP, and XGBoost with a 10-trial hyperparameter sweep).
Here is the MLflow UI sorted by MAE, showing the best performing model:

![alt text](<WhatsApp Image 2026-09-28 at 1.46.19 AM.jpeg>)

## 2. Model Registry & Promotion
We registered the best XGBoost model and promoted it to `Production`. 
The API automatically loaded the new Production model without any code changes or rebuilds.


## 3. GitHub Actions CI Pipeline (The Quality Gate)
Our CI pipeline runs linting, testing (with a mocked MLflow server and a 70% coverage gate), and builds the Docker image. 
Here is the proof of the pipeline catching an error (Red) and then passing successfully (Green):

![alt text](<WhatsApp Image 2026-09-28 at 2.19.44 AM.jpeg>)

![alt text](<WhatsApp Image 2026-10-05 at 2.39.48 AM.jpeg>)

## 4. Infrastructure as Code (Terraform)
We provisioned our Postgres database and MinIO artifact store using Terraform.

**Why must Terraform state files (`.tfstate`) never be committed to Git?**
State files contain sensitive information and plain-text secrets (like database passwords and root users). Committing them to a public Git repository exposes these secrets. Additionally, in a team environment, remote state backends (like AWS S3 with DynamoDB locking) solve the problem of multiple developers trying to modify the infrastructure simultaneously, preventing state corruption.


## 5. Continuous Training (CT) Pipeline
We implemented a GitHub Actions CT pipeline with a human-approval gate for Production deployment.
Here is the successful run showing the manual approval:

![alt text](<WhatsApp Image 2026-10-05 at 2.39.35 AM-2.jpeg>)

### Continuous Training Decision Table
Based on the session, here is how our CT pipeline handles different triggers:

| Trigger Type | Implemented Now? | Handled by Module 5? | Description |
| :--- | :--- | :--- | :--- |
| **Schedule (Cron)** | ✅ YES | No | Retrains weekly (`0 0 * * 0`) to catch gradual data drift. |
| **New Labeled Data** | ✅ YES | No | Triggered via `workflow_dispatch` with a specific data version input. |
| **Data Drift Detected** | ❌ NO | ✅ YES | Will use `repository_dispatch` webhook sent by Evidently in Module 5. |
| **Performance Degradation**| ❌ NO | ✅ YES | Will use `repository_dispatch` webhook sent by Prometheus/Grafana alerts in Module 5. |
