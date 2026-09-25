# CICDFailurePredictor_Cloud_Project_2026

> AI-Based CI/CD Failure Prediction Framework for Enterprise Software Development using Predictive DevOps Analytics

**Course**: BCSE355L — Cloud Architecture Design Project
**Instructor**: Dr. Priya V

## Team

| Student | Reg. No |
|---|---|
| Sahana Ramanathan | 24BIT0618 |
| Yarra Pranav | 24BIT0494 |
| Susendar | 24BIT0081 |

## Abstract

CI/CD pipelines in enterprise teams run thousands of builds daily; a meaningful share fail for reasons unrelated to the actual code change — flaky tests, dependency drift, environment misconfiguration. This project predicts pipeline failure probability *before* execution using historical build data, repository metrics and infrastructure signals and aggregates outcomes into DevOps analytics tracking reliability by repository, team and pipeline stage.

## AWS services

| Service | Purpose |
|---|---|
| Amazon S3 | Store datasets and build logs |
| Amazon RDS | Store structured repository/build metadata |
| Amazon SageMaker | Train and deploy the failure-prediction model |
| AWS Lambda | Event-driven inference on new pipeline triggers |
| Amazon CloudWatch | Monitor pipeline and service health |
| Amazon SNS | Send high-risk failure alerts |
| Amazon QuickSight | Visualize repository/DevOps analytics |

## Repository structure

CICDFailurePredictor_Cloud_Project_2026/
├── README.md
├── LICENSE
├── .gitignore
├── docs/
│ └── Project_Report.docx
├── architecture/
│ ├── AWS_Architecture.png
│ ├── System_Architecture.png
│ └── Workflow.png
├── dataset/
│ ├── raw/
│ ├── processed/
│ └── dataset_description.pdf
├── src/
│ ├── frontend/
│ ├── backend/
│ ├── ml_model/
│ └── aws/
├── results/
│ ├── graphs/
│ ├── screenshots/
│ └── accuracy.xlsx
└── presentation/



## Setup

```bash
git clone https://github.com/sahanaramanathan2024-hash/CICDFailurePredictor_Cloud_Project_2026.git
cd CICDFailurePredictor_Cloud_Project_2026
```

## Status

- [x] Literature survey (15 papers) + research gap analysis
- [x] AWS architecture design (2 diagrams)
- [ ] Dataset finalization
- [ ] ML model training
- [ ] Backend + AWS integration
- [ ] Results & demo

