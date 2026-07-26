# 🎬 MovieIQ: Movie Success Prediction using Machine Learning

# 📌 Project Overview

MovieIQ is an end-to-end Data Analysis / Data Science project that predicts whether a movie is likely to become commercially successful using Machine Learning.
The project combines data preprocessing, exploratory data analysis (EDA), statistical hypothesis testing, Random Forest Classification, and an interactive Streamlit dashboard to provide business insights and predictive analytics.

---

# 🎯 Business Problem

The film industry invests millions of dollars in movie production, yet predicting commercial success remains a significant challenge.

Movie studios often rely on historical trends and expert judgment while making investment decisions. A data-driven prediction system can help reduce financial risk by estimating the probability of a movie's success before its release.

This project aims to build a Machine Learning solution capable of predicting movie success based on production and audience-related features.

---

# 🎯 Project Objectives

- Analyze historical movie data.
- Perform data cleaning and preprocessing.
- Explore relationships between movie features.
- Validate hypotheses using statistical tests.
- Build a Random Forest Classification model.
- Predict movie success.
- Develop an interactive Streamlit dashboard.
- Provide business recommendations based on analytical findings.

---

# 📂 Dataset

The dataset contains historical movie information including production, financial, and audience-related attributes.

### Features

- Budget
- Revenue
- Popularity
- Runtime
- Vote Average
- Genres
- Title
- Success (Target Variable)

---

# 🛠 Tech Stack

| Category | Tools |
|----------|-------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly, Matplotlib |
| Machine Learning | Scikit-learn |
| Statistical Analysis | SciPy |
| Dashboard | Streamlit |
| Model Serialization | Joblib |

---

# 🔄 Project Workflow

```text
Business Understanding
        │
        ▼
Data Collection
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Statistical Hypothesis Testing
        │
        ▼
Feature Selection
        │
        ▼
Random Forest Classification
        │
        ▼
Model Evaluation
        │
        ▼
Model Deployment
        │
        ▼
Interactive Streamlit Dashboard
```

---

# 📊 Exploratory Data Analysis

The exploratory analysis focused on understanding the characteristics of successful and unsuccessful movies.

The analysis included:

- Success distribution
- Budget vs Revenue analysis
- Popularity distribution
- Runtime analysis
- Genre analysis
- Vote Average analysis
- Correlation Heatmap

---

# 📈 Statistical Analysis

## Independent Sample T-Test

**Objective**

Determine whether popularity significantly differs between successful and unsuccessful movies.

### Results

| Metric | Value |
|--------|------:|
| T Statistic | 2.0647 |
| P Value | 0.03908 |

**Conclusion**

The null hypothesis was rejected, indicating a statistically significant relationship between popularity and movie success.

---

## Chi-Square Test

**Objective**

Determine whether movie genre is associated with commercial success.

### Results

| Metric | Value |
|--------|------:|
| Chi-Square Statistic | 1.7731 |
| P Value | 0.994569 |

**Conclusion**

The null hypothesis could not be rejected. Genre alone is not significantly associated with movie success.

---

# 🤖 Machine Learning Model

### Model Used

- Random Forest Classifier

### Why Random Forest?

- Handles nonlinear relationships
- Reduces overfitting through ensemble learning
- Works well on structured datasets
- Provides feature importance
- High predictive performance

---

# ⭐ Feature Importance

The model identified the following variables as the most influential predictors of movie success.

- Popularity
- Budget
- Vote Average
- Runtime

Feature importance improves model interpretability and explains which variables contribute most to predictions.

---

# 📱 Streamlit Dashboard

The application contains seven interactive modules.

| Module | Description |
|---------|-------------|
| 🏠 Home | Project overview and KPIs |
| 📊 Data Explorer | Interactive dataset exploration |
| 📈 EDA Dashboard | Visual analysis of movie data |
| 📉 Statistical Analysis | T-Test and Chi-Square results |
| 🤖 Prediction | Predict movie success |
| ⭐ Feature Importance | Explain model predictions |
| 💼 Business Insights | Executive summary and recommendations |

---

# 💡 Key Business Insights

- Popularity is the strongest predictor of movie success.
- Movies with larger budgets generally generate higher revenue.
- Audience ratings positively influence commercial performance.
- Genre alone is not a reliable predictor of success.
- Machine Learning can support better investment decisions in the film industry.

---

# 📈 Business Recommendations

- Increase pre-release marketing to improve popularity.
- Optimize production budgets based on expected returns.
- Monitor audience ratings and reviews.
- Use predictive analytics before approving movie projects.
- Consider multiple production and audience metrics instead of relying solely on genre.

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/MovieIQ.git
```

Navigate to the project

```bash
cd MovieIQ
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python -m streamlit run app.py
```

---

# 📸 Preview


---
<img width="635" height="343" alt="Screenshot 2026-07-25 180258" src="https://github.com/user-attachments/assets/d07d7962-0df9-47a9-87a6-e17502866b1a" />
<img width="397" height="450" alt="newplot (2)" src="https://github.com/user-attachments/assets/741fb70c-f8b2-4c1e-82b6-5112b5d10aa6" />
<img width="810" height="450" alt="newplot (6)" src="https://github.com/user-attachments/assets/dbfe29bd-9614-4714-a8bd-2751470d5154" />
<img width="397" height="450" alt="newplot (5)" src="https://github.com/user-attachments/assets/119ea6e9-aa3e-402b-9766-fc47bad08ed3" />
<img width="397" height="450" alt="newplot (4)" src="https://github.com/user-attachments/assets/bd00f740-d9a4-43c4-81dd-cecca1a3e6ed" />
<img width="397" height="450" alt="newplot (3)" src="https://github.com/user-attachments/assets/515ae005-580b-4284-9681-ad6cddf279a9" />
<img width="640" height="344" alt="Screenshot 2026-07-25 180136" src="https://github.com/user-attachments/assets/5c586291-81f6-4b07-a0d1-21eecc80299b" />
<img width="622" height="349" alt="Screenshot 2026-07-26 135435" src="https://github.com/user-attachments/assets/cf66a5a9-5419-4f27-813b-035b3405a624" />

---

# 🔮 Future Enhancements

- Hyperparameter tuning
- Model comparison with XGBoost and LightGBM
- SHAP-based model explainability
- Cloud deployment
- REST API integration
- Real-time prediction service

---

# 👨‍💻 Author

**Shlok Sundriyal**

Data Analyst | Data Science Enthusiast

**Skills:** Python • SQL • Power BI • Machine Learning • Streamlit • Business Analytics

---

## ⭐ Thank You . If you found this project helpful, consider giving it a star!
