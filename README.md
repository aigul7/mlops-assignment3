\# Assignment 3: AutoML



\## Overview

This project demonstrates AutoML using two different platforms:

\- \*\*FLAML\*\* (Fast Lightweight AutoML) on Databricks Community Edition

\- \*\*H2O AutoML\*\* running in Docker



Both platforms use the same \*\*athletes dataset\*\* (1000 athletes, binary classification: medal vs no medal).



\## Project Structure

```

mlops-assignment3/

├── databricks/

│   └── Assignment3\_AutoML.ipynb    # FLAML AutoML notebook (Questions 1-8)

├── h2o-docker/

│   ├── Dockerfile                  # Docker configuration

│   ├── h2o\_automl.py              # H2O AutoML script (Question 9)

│   ├── athletes.csv               # Dataset

│   └── requirements.txt           # Python dependencies

└── README.md

```



\## Dataset

\- \*\*athletes.csv\*\* — 1000 athletes with features: age, height, weight, country, sport, years\_experience, training\_hours\_per\_week

\- \*\*Target:\*\* won\_medal (binary: medal won or not)

\- \*\*Distribution:\*\* 450 medal winners (45%), 550 no medal (55%)



\## Part 1: FLAML AutoML on Databricks (Questions 1-8)



\### Platform

Databricks Community Edition with FLAML (the engine behind Databricks AutoML)



\### Results — All Features



| Rank | Model | Validation F1 | Training Time |

|------|-------|---------------|---------------|

| #1 | KNeighbor | 0.5032 | 16.9 sec |

| #2 | LGBM | 0.5020 | 12.21 sec |

| #3 | Random Forest | 0.2171 | 3.17 sec |



\### Top 5 Features (Permutation Importance)

1\. weight (0.0572)

2\. training\_hours\_per\_week (0.0316)

3\. age (0.0246)

4\. country (0.0144)

5\. height (0.0137)



\### Results — Top 3 Features Only (weight, training\_hours\_per\_week, age)



| Rank | Model | Validation F1 | Training Time |

|------|-------|---------------|---------------|

| #1 | XGBoost | 0.5868 | 111.49 sec |

| #2 | LGBM | 0.4624 | 3.55 sec |



\### Comparison with Assignment 1



| Model | Test F1 | Best Estimator |

|-------|---------|----------------|

| Assignment 1 (Manual RF) | 0.4294 | RandomForest |

| FLAML AutoML (All Features) | 0.4868 | KNeighbor |

| FLAML AutoML (Top 3 Features) | 0.4693 | XGBoost |



\*\*AutoML improved F1 score by +13.4% over manual model.\*\*



\### Platform Assessment (Q8)

FLAML on Databricks Community Edition is \*\*FULL-CODE\*\* because:

\- Uses Python API (`AutoML().fit()`) — no GUI

\- Requires code for configuration, execution, and result extraction

\- No drag-and-drop interface available in Community Edition



\## Part 2: H2O AutoML in Docker (Question 9)



\### Setup

```bash

cd h2o-docker

docker build -t h2o-automl .

docker run --name h2o-automl-run -m 4g h2o-automl

```



\### Results — All Features



| Rank | Model | AUC |

|------|-------|-----|

| #1 | GBM\_2 | 0.4926 |

| #2 | XRT\_1 | 0.4909 |

| #3 | XGBoost\_1 | 0.4855 |



\### Top 5 Features (H2O Variable Importance)

1\. height (0.2256)

2\. weight (0.1567)

3\. country (0.1528)

4\. training\_hours\_per\_week (0.1459)

5\. years\_experience (0.1165)



\### Results — Top 3 Features Only (height, weight, country)



| Rank | Model | AUC |

|------|-------|-----|

| #1 | XGBoost\_2 | 0.5158 |

| #2 | XGBoost\_3 | 0.5060 |

| #3 | GBM\_2 | 0.5048 |



\## Tools \& Technologies

\- \*\*Databricks Community Edition\*\* — MLOps platform

\- \*\*FLAML\*\* — Fast Lightweight AutoML library (engine behind Databricks AutoML)

\- \*\*H2O AutoML\*\* — Open-source AutoML framework

\- \*\*Docker\*\* — Containerization for H2O environment

\- \*\*MLflow\*\* — Experiment tracking (integrated in Databricks)

\- \*\*Python\*\* — scikit-learn, pandas, numpy, matplotlib

