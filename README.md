[README.md](https://github.com/user-attachments/files/31545665/README.md)
# End-to-End Customer Churn Prediction Pipeline with FastAPI

A production-ready Machine Learning system that identifies customer churn risks using an optimized XGBoost classifier and exposes the model via a high-performance FastAPI endpoint.

## 🚀 Key Features
- **Production Pipeline:** Built using `scikit-learn`'s `ColumnTransformer` and `Pipeline` architectures to completely prevent training data leakage.
- **Imbalance Handling:** Tuned classification thresholds and utilized XGBoost's `scale_pos_weight` to boost minority class evaluation metrics.
- **Business-Optimized Metrics:** Elevated minority class (Churn) **Recall from 0.49 to 0.87**, ensuring 87% of high-risk customers are successfully identified for targeted retention.
- **RESTful API:** Developed a microservice API using FastAPI with automated schema validation using Pydantic models.

## 🛠️ Project Architecture
1. **Data Preprocessing:** Standard Scaling for continuous variables (`tenure`, charges) and One-Hot Encoding for categorical factors.
2. **Model Training:** Gradient boosted decision trees using XGBoost optimized via precision/recall threshold adjustment.
3. **Deployment:** Serialization via `joblib` served synchronously using Uvicorn.

## 💻 Technical Setup

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost fastapi uvicorn joblib pydantic




1. Train the model and export the pipeline:
  
  python main.py

2. Spin up the live serving API:

  python -m uvicorn app:app --reload
  to kill use :- kill -9 $(lsof -t -i:8000)

3. Open http://127.0.0.1:8000/docs in your browser to interactively test user profiles.

## 🏆 Project Complete!
Push this entire repository up to your GitHub profile. You now have a solid project under your belt that covers **data cleaning, handling real-world class imbalance, model evaluation trade-offs, serialization, and deployment engineering**. 

Now that you've mastered an end-to-end tabular data pipeline, would you like to explore another project to add to your stack—perhaps diving into Computer Vision or NLP next?



kill command : 
lsof: Stands for "List Open Files". In Unix-like operating systems, network connections are treated as files.
-i:8000: Tells lsof to filter the results to only show internet/network connections using port 8000.
-t: Stands for "terse". This strips away all columns (like user, command name, and memory usage) and outputs only the raw Process ID (PID) number.
$( ... ): This is a command substitution. It executes the lsof command inside the parentheses first and passes its output directly to the kill command.
kill -9: The kill command sends a termination signal. The -9 flag represents the SIGKILL signal, which forces the operating system to terminate the process immediately without letting it save data, clean up cache, or safely close connections.
