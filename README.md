# **E-commerce and Bank Fraud Detection System**

## **Project Description**

This project develops an advanced machine learning system designed to accurately detect fraudulent activities across both **e-commerce transactions** and **bank credit card operations**. By leveraging comprehensive data analysis, sophisticated feature engineering, and state-of-the-art predictive modeling, the system aims to minimize financial losses, enhance transaction security, and provide actionable insights for financial technology companies like Adey Innovations Inc.

## **Business Understanding**

Adey Innovations Inc. faces the critical challenge of improving its fraud detection capabilities. The core problem lies in the **highly imbalanced nature of fraud data**, where legitimate transactions vastly outnumber fraudulent ones. This imbalance necessitates a delicate balance between:

- **Minimizing Financial Losses:** Achieved by reducing **False Negatives** (missed fraudulent transactions).
- **Enhancing User Experience:** Achieved by minimizing **False Positives** (legitimate transactions incorrectly flagged as fraud), which can lead to customer inconvenience and distrust.

This project directly addresses these challenges by building **accurate, robust, and interpretable models** that not only prevent financial losses but also foster trust with customers and financial institutions.

## **Project Overview**

This initiative employs a multi-faceted approach to fraud detection, encompassing the entire machine learning lifecycle:

1. **Data Acquisition & Preprocessing:** Handling diverse transaction datasets, including cleaning, imputation, and ensuring correct data types.
2. **Geolocation & Temporal Feature Engineering:** Enriching transaction data with geographical context derived from IP addresses and extracting crucial time-based patterns (e.g., transaction velocity, time since signup).
3. **Class Imbalance Handling:** Applying advanced over-sampling techniques (like SMOTE) to effectively address the severe class imbalance inherent in fraud datasets, ensuring models learn from minority fraud cases.
4. **Model Development & Experimentation:** Building and rigorously comparing various machine learning models, from interpretable baselines (Logistic Regression, Decision Tree) to powerful ensemble methods (Random Forest, XGBoost, LightGBM).
5. **Rigorous Evaluation:** Assessing model performance using metrics specifically suited for imbalanced data, such as Precision, Recall, F1-score, and ROC-AUC/Precision-Recall AUC, with a strong focus on their business implications.
6. **Model Explainability (XAI):** Utilizing cutting-edge techniques like SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) to interpret model decisions, providing transparency and actionable insights into the key drivers of fraud.
7. **MLflow Integration:** Implementing MLflow for comprehensive experiment tracking, model versioning, and artifact management to ensure reproducibility and streamlined deployment.

## **Key Features**

- **Dual-Dataset Analysis:** Comprehensive fraud detection solutions for both distinct e-commerce and bank credit card transaction datasets.
- **Advanced Data Preprocessing:** Robust pipelines for data cleaning, type conversion, and handling missing values.
- **Rich Feature Engineering:** Creation of sophisticated temporal features (e.g., transaction hour, day of week, time since signup) and behavioral features (e.g., transaction frequency and total/average amount over various time windows per user/device/IP).
- **Geolocation Integration:** Merges IP address data with geographical information to enhance fraud pattern recognition with location-based context.
- **Effective Imbalance Handling:** Implementation of techniques like SMOTE during training to effectively manage highly skewed class distributions.
- **Comparative Modeling:** Evaluation of multiple classification algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM) to identify optimal performers.
- **Explainable AI (XAI):** Integration of SHAP for global feature importance and LIME for local, instance-level prediction explanations, providing transparency and trust in model decisions.
- **MLflow for MLOps:** Utilizes MLflow for end-to-end experiment tracking, model registry, and artifact management, supporting reproducible research and deployment readiness.
- **Robust Evaluation Metrics:** Focuses on business-relevant metrics like Precision, Recall, F1-score, and AUC-ROC/AUC-PR, crucial for imbalanced classification problems.
- **Production-Ready Prediction Script:** A standalone `run_predict.py` script capable of loading trained models and processors directly for new data inference.

## **Business Objectives**

The successful implementation of this project aims to achieve the following for Adey Innovations Inc.:

- **Minimize Financial Losses:** Significantly reduce monetary losses due to undetected fraudulent transactions.
- **Enhance Transaction Security:** Strengthen the overall security posture of both e-commerce and banking platforms, protecting customers and the business.
- **Improve User Trust & Experience:** Decrease false positives, leading to fewer legitimate transactions being blocked and a more seamless, trustworthy experience for users.
- **Derive Actionable Insights:** Provide clear, interpretable reasons behind fraud predictions, enabling fraud analysts and business stakeholders to understand underlying patterns and inform strategic decisions.
- **Optimize Operational Efficiency:** Automate and improve the accuracy of fraud detection, reducing the need for manual review and accelerating response times.

## **Project Structure**

```
.
├── .github/                 # GitHub specific configurations (e.g., Workflows for CI/CD)
│   └── workflows/
│       └── main.yml         # CI/CD pipeline for tests and linting
├── config/                  # Configuration files (e.g., model hyperparameters, data paths)
│   └── config.py
├── scripts/                 # Standalone utility scripts for running pipeline stages
│   ├── run_data_pipeline.py # Orchestrates data loading, preprocessing, and saving processed data
│   ├── run_train.py         # Manages model training, evaluation, MLflow logging, and model export
│   ├── run_predict.py       # Handles loading models and making predictions on new data
│   └── run_interpret.py     # Executes model interpretation (SHAP, LIME)
├── data/                    # Data storage
│   ├── raw/                 # Original, immutable raw datasets (e.g., creditcard.csv, Fraud_Data.csv, IpAddress_to_Country.csv)
│   └── processed/           # Transformed, cleaned, and feature-engineered data ready for modeling
│   └── predictions/         # Output directory for prediction results on new data
├── docs/                    # Project documentation, reports, and insights
│   └── report.md            # Final project report/blog post content (placeholder)
├── exported_models/         # Directory for directly exported best models and processors (.pkl files)
├── mlruns/                  # MLflow tracking directory (local backend for experiment logs and artifacts)
├── notebooks/               # Jupyter notebooks for experimentation, EDA, and prototyping
│   ├── 01_EDA.ipynb      # Comprehensive EDA notebook with conditional data loading
│   ├── 02_Model_Experimentation.ipynb   # Detailed model training, evaluation, and comparison
│   └── 1.1-eda.ipynb                    # Intermediate EDA exploration notebook (retained for reference)
├── src/                     # Core source code for the project
│   ├── __init__.py          # Marks src as a Python package
│   ├── main.py              # Main entry point to run the entire pipeline (future integration)
│   ├── data_processing/     # Modules for data loading, cleaning, and preprocessing
│   │   ├── __init__.py
│   │   ├── loader.py        # Handles loading raw data
│   │   └── preprocessor.py  # Defines data preprocessing pipelines for each dataset
│   ├── eda/                 # Modules for Exploratory Data Analysis strategies
│   │   ├── __init__.py
│   │   ├── data_inspection.py
│   │   ├── univariate_analysis.py
│   │   ├── bivariate_analysis.py
│   │   ├── multivariate_analysis.py
│   │   ├── missing_values_analysis.py
│   │   ├── outlier_analysis.py
│   │   └── temporal_analysis.py
│   ├── feature_engineering/ # Modules for creating new features
│   │   ├── __init__.py
│   │   └── engineer.py      # Custom transformers for feature creation
│   ├── models/              # Modules for model definition, training, and prediction
│   │   ├── __init__.py
│   │   ├── base_model_strategy.py       # Abstract base class for model strategies
│   │   ├── model_trainer.py             # Context class for training models using strategies
│   │   ├── model_evaluator.py           # Functions for evaluating model performance
│   │   ├── model_interpreter.py         # Integrates SHAP and LIME for model explainability
│   │   ├── logistic_regression_strategy.py
│   │   ├── decision_tree_strategy.py
│   │   ├── random_forest_strategy.py
│   │   ├── xgboost_strategy.py
│   │   └── lightgbm_strategy.py         # New LightGBM strategy
│   └── utils/               # Utility functions and helper classes
│       ├── __init__.py
│       └── helpers.py       # General helper functions (e.g., IP conversion, path handling)
├── tests/                   # Test suite (unit and integration tests)
│   ├── unit/                # Unit tests for individual components
│   └── integration/         # Integration tests for combined components/pipeline
├── .env                     # Environment variables (e.g., API keys - kept out of Git)
├── .gitignore               # Specifies intentionally untracked files to ignore
├── Makefile                 # Common development tasks (setup, test, lint, clean)
├── pyproject.toml           # Modern Python packaging configuration (PEP 517/621)
├── README.md                # Project overview, installation, usage (this file)
└── requirements.txt         # Python dependencies

```

## **Technologies Used**

- **Programming Language:** Python 3.8+
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning Frameworks:** `scikit-learn`, `xgboost`, `lightgbm`, `imbalanced-learn`
- **Model Explainability:** `shap`, `lime`
- **Experiment Tracking & MLOps:** `MLflow`
- **Visualization:** `matplotlib`, `seaborn`
- **Version Control:** Git
- **Dependency Management:** `pip` (with `requirements.txt`)
- **Code Quality:** `flake8`, `black` (via `Makefile`)
- **Testing:** `pytest`
- **Serialization:** `cloudpickle`

## **Setup and Installation**

### **Prerequisites**

Ensure you have the following installed on your system:

- Python 3.8 or higher
- Git

### **Steps**

1. **Clone the repository:**
    
    ```
    git clone https://github.com/michaWorku/E-commerce-Bank-Fraud-Detection.git # Replace with your actual repo URL
    cd E-commerce-Bank-Fraud-Detection
    
    ```
    
    *(If you created the project in the current directory, you are already in the project root.)*
    
2. **Create and activate a virtual environment:**
    
    ```
    python3 -m venv venv
    source venv/bin/activate # On Windows: .\venv\Scripts\activate
    
    ```
    
3. **Install dependencies:**
    
    ```
    pip install -r requirements.txt
    
    ```
    
4. **Download Raw Data:**
    - Place `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` into the `data/raw/` directory. (These datasets are typically found on platforms like Kaggle).

## **Usage**

To run the complete data engineering, model training, and prediction pipeline:

1. Run the Data Pipeline (Preprocessing & Feature Engineering):
    
    This script will load raw data, preprocess it, engineer new features, and save the processed data to data/processed/. It also includes a demonstration of class imbalance handling.
    
    ```
    python scripts/run_data_pipeline.py
    
    ```
    
2. Run Model Training & MLflow Experimentation:
    
    This script trains various models, evaluates them, logs experiments to MLflow, registers the best models, and exports them to exported_models/.
    
    ```
    python scripts/run_train.py
    
    ```
    
3. Run Predictions on New Data:
    
    This script loads the best-performing models and their processors directly from exported_models/ and makes predictions on dummy new data, saving the results to data/predictions/.
    
    ```
    python scripts/run_predict.py --generate-dummy-data # Use --generate-dummy-data on first run
    
    ```
    
4. Run Model Interpretation:
    
    This script loads trained models and uses SHAP and LIME to provide insights into global feature importance and local prediction explanations.
    
    ```
    python scripts/run_interpret.py --generate-dummy-data # Use --generate-dummy-data on first run
    
    ```
    
5. Explore Jupyter Notebooks:
    
    Navigate to the notebooks/ directory to explore detailed steps for EDA, feature engineering, and model experimentation:
    
    - `01_EDA.ipynb`: Comprehensive EDA and data understanding.
    - `02_Model_Experimentation.ipynb`: Detailed model training, evaluation, and comparison.
    - `1.1-eda.ipynb` & `1.2-eda.ipynb`: Supplementary EDA explorations.

## **Contributing**

Contributions are highly welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
    
    ```
    git checkout -b feature/your-feature-name
    
    ```
    
3. Make your changes and ensure tests pass.
4. Commit your changes with a descriptive message:
    
    ```
    git commit -m 'feat: Add new feature for X'
    
    ```
    
5. Push your changes to your forked repository:
    
    ```
    git push origin feature/your-feature-name
    
    ```
    
6. Open a Pull Request to the `main` branch of this repository, describing your changes in detail.

## **License**

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).