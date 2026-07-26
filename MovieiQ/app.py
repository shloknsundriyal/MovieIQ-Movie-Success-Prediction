# ==========================================================
# 🎬 MovieIQ - Movie Success Prediction Dashboard
# ==========================================================

# -----------------------------
# Import Libraries
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="MovieIQ",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Custom CSS
# -----------------------------
with open("style.css") as css:
    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("movies.csv")

    # Create Success column if missing
    if "Success" not in df.columns:
        df["Success"] = (df["revenue"] > df["budget"]).astype(int)

    return df

df = load_data()

# -----------------------------
# Load Trained Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("random_forest.pkl")

model = load_model()

# -----------------------------
# Global Feature Importance
# -----------------------------
feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(by="Importance", ascending=False)
    .reset_index(drop=True)
)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.image(
    "https://img.icons8.com/fluency/96/movie-projector.png",
    width=80
)

st.sidebar.title("🎬 MovieIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Data Explorer",
        "📈 EDA Dashboard",
        "📉 Statistical Analysis",
        "🤖 Prediction",
        "⭐ Feature Importance",
        "💼 Business Insights"
    ]
)
# ==========================================================
# 🏠 HOME PAGE
# ==========================================================

if page == "🏠 Home":

    # -----------------------------
    # Hero Section
    # -----------------------------
    st.markdown("""
        <h1 style='text-align:center; color:#F8FAFC;'>
            🎬 MovieIQ
        </h1>

        <h3 style='text-align:center; color:#CBD5E1;'>
            Predicting Movie Success using Machine Learning
        </h3>

        <hr>
    """, unsafe_allow_html=True)

    # -----------------------------
    # KPI Metrics
    # -----------------------------
    total_movies = len(df)
    successful_movies = df["Success"].sum()
    success_rate = (successful_movies / total_movies) * 100
    avg_budget = df["budget"].mean() / 1_000_000
    avg_revenue = df["revenue"].mean() / 1_000_000

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🎥 Total Movies", f"{total_movies:,}")
    col2.metric("🏆 Successful Movies", f"{successful_movies:,}")
    col3.metric("💰 Avg Budget", f"${avg_budget:.1f} M")
    col4.metric("💵 Avg Revenue", f"${avg_revenue:.1f} M")

    st.markdown("---")

    # -----------------------------
    # Project Overview
    # -----------------------------
    st.subheader("📌 Project Overview")

    st.write("""
    **MovieIQ** is a Machine Learning application developed to predict whether a movie
    will become commercially successful based on its production characteristics.

    The project combines:

    - 📊 Exploratory Data Analysis (EDA)
    - 📈 Statistical Hypothesis Testing
    - 🤖 Random Forest Classification
    - 📱 Interactive Streamlit Dashboard

    The prediction is based on key attributes such as **Budget, Popularity,
    Runtime, and Vote Average**.
    """)

    st.markdown("---")

    # -----------------------------
    # Dataset Summary
    # -----------------------------
    st.subheader("📂 Dataset Summary")

    c1, c2 = st.columns(2)

    with c1:
        st.write("### Numerical Features")
        st.write(df[["budget", "revenue", "popularity",
                     "runtime", "vote_average"]].describe())

    with c2:
        st.write("### Sample Records")
        st.dataframe(df.head())


# ==========================================================
# 📊 DATA EXPLORER
# ==========================================================

elif page == "📊 Data Explorer":

    st.title("📊 Data Explorer")
    st.markdown("Interactively explore and filter the movie dataset.")

    st.markdown("---")

    # -----------------------------
    # Sidebar Filters
    # -----------------------------
    st.sidebar.subheader("🔍 Filter Movies")

    # Genre Filter
    genres = ["All"] + sorted(df["genres"].dropna().unique().tolist())
    selected_genre = st.sidebar.selectbox(
        "Select Genre",
        genres
    )

    # Budget Filter
    budget_range = st.sidebar.slider(
        "Budget ($)",
        int(df["budget"].min()),
        int(df["budget"].max()),
        (
            int(df["budget"].min()),
            int(df["budget"].max())
        )
    )

    # Popularity Filter
    popularity_range = st.sidebar.slider(
        "Popularity",
        float(df["popularity"].min()),
        float(df["popularity"].max()),
        (
            float(df["popularity"].min()),
            float(df["popularity"].max())
        )
    )

    # Runtime Filter
    runtime_range = st.sidebar.slider(
        "Runtime (Minutes)",
        int(df["runtime"].min()),
        int(df["runtime"].max()),
        (
            int(df["runtime"].min()),
            int(df["runtime"].max())
        )
    )

    # -----------------------------
    # Apply Filters
    # -----------------------------
    filtered_df = df.copy()

    if selected_genre != "All":
        filtered_df = filtered_df[
        filtered_df["genres"].str.contains(selected_genre, case=False, na=False)
    ]

    filtered_df = filtered_df[
        (filtered_df["budget"] >= budget_range[0]) &
        (filtered_df["budget"] <= budget_range[1]) &
        (filtered_df["popularity"] >= popularity_range[0]) &
        (filtered_df["popularity"] <= popularity_range[1]) &
        (filtered_df["runtime"] >= runtime_range[0]) &
        (filtered_df["runtime"] <= runtime_range[1])
    ]

    # -----------------------------
    # KPI Cards
    # -----------------------------
    c1, c2, c3 = st.columns(3)

    c1.metric("Movies Found", len(filtered_df))
    c2.metric("Average Budget", f"${filtered_df['budget'].mean()/1e6:.2f} M")
    c3.metric("Average Revenue", f"${filtered_df['revenue'].mean()/1e6:.2f} M")

    st.markdown("---")

    # -----------------------------
    # Movie Table
    # -----------------------------
    st.subheader("🎥 Filtered Movies")

    st.dataframe(
        filtered_df[
            [
                "title",
                "genres",
                "budget",
                "revenue",
                "popularity",
                "runtime",
                "vote_average",
                "Success"
            ]
        ],
        use_container_width=True
    )

    # -----------------------------
    # Download Button
    # -----------------------------
    csv = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Filtered Dataset",
        data=csv,
        file_name="filtered_movies.csv",
        mime="text/csv"
    )

# ==========================================================
# 📈 EDA DASHBOARD
# ==========================================================

elif page == "📈 EDA Dashboard":

    st.title("📈 Exploratory Data Analysis")
    st.markdown("Visual exploration of the movie dataset.")

    st.markdown("---")

    # -----------------------------
    # Row 1
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:

        fig = px.scatter(
            df,
            x="budget",
            y="revenue",
            color="Success",
            hover_name="title",
            title="Budget vs Revenue",
            color_continuous_scale="Viridis"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        success_counts = df["Success"].value_counts().reset_index()
        success_counts.columns = ["Success", "Count"]

        fig = px.pie(
            success_counts,
            values="Count",
            names="Success",
            title="Movie Success Distribution",
            hole=0.45
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # Row 2
    # -----------------------------
    col3, col4 = st.columns(2)

    with col3:

        genre_count = (
            df["genres"]
            .value_counts()
            .head(10)
            .reset_index()
        )

        genre_count.columns = ["Genre", "Count"]

        fig = px.bar(
            genre_count,
            x="Genre",
            y="Count",
            title="Top 10 Genres"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col4:

        fig = px.box(
            df,
            x="Success",
            y="popularity",
            title="Popularity vs Success"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # Row 3
    # -----------------------------
    col5, col6 = st.columns(2)

    with col5:

        fig = px.box(
            df,
            x="Success",
            y="runtime",
            title="Runtime vs Success"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col6:

        fig = px.box(
            df,
            x="Success",
            y="vote_average",
            title="Vote Average vs Success"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # Correlation Heatmap
    # -----------------------------

    st.subheader("Correlation Heatmap")

    corr = df[
        [
            "budget",
            "revenue",
            "popularity",
            "runtime",
            "vote_average",
            "Success"
        ]
    ].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Correlation Matrix"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 📉 STATISTICAL ANALYSIS
# ==========================================================

elif page == "📉 Statistical Analysis":

    st.title("📉 Statistical Analysis")
    st.markdown("Statistical validation of key business hypotheses.")

    st.markdown("---")

    # -----------------------------
    # T-Test
    # -----------------------------

    st.subheader("📊 Independent Sample T-Test")

    col1, col2 = st.columns(2)

    with col1:

        st.metric("T Statistic", "2.0647")
        st.metric("P-Value", "0.03908")

    with col2:

        st.success("✅ Reject the Null Hypothesis")

        st.write("""
The p-value is **less than 0.05**, indicating a statistically significant
difference between the compared groups.

This suggests that **movie popularity has a significant relationship with movie success.**
""")

    st.markdown("---")

    # -----------------------------
    # Chi-Square Test
    # -----------------------------

    st.subheader("📈 Chi-Square Test")

    col3, col4 = st.columns(2)

    with col3:

        st.metric("Chi-Square Statistic", "1.7731")
        st.metric("P-Value", "0.994569")
        st.metric("Degrees of Freedom", "9")

    with col4:

        st.warning("⚠️ Fail to Reject the Null Hypothesis")

        st.write("""
The p-value is **greater than 0.05**, indicating no statistically significant
association between **Genre** and **Movie Success**.

Genre alone is not a reliable predictor of commercial success.
""")

    st.markdown("---")

    # -----------------------------
    # Business Interpretation
    # -----------------------------

    st.subheader("💼 Business Interpretation")

    st.info("""
### Key Findings

✅ Popularity has a statistically significant impact on movie success.

⚠️ Genre alone does not significantly influence commercial success.

🎯 Marketing efforts that increase audience awareness and popularity
may contribute more to movie success than genre selection alone.
""")

# ==========================================================
# 🤖 MOVIE SUCCESS PREDICTOR
# ==========================================================

elif page == "🤖 Prediction":

    st.title("🤖 Movie Success Predictor")
    st.markdown("Predict whether a movie is likely to be commercially successful.")

    st.markdown("---")

    st.subheader("🎬 Enter Movie Details")

    col1, col2 = st.columns(2)

    with col1:

        budget = st.number_input(
            "Budget ($)",
            min_value=0.0,
            value=50000000.0,
            step=1000000.0,
            format="%.0f"
        )

        popularity = st.number_input(
            "Popularity",
            min_value=0.0,
            value=20.0,
            step=1.0
        )

    with col2:

        runtime = st.number_input(
            "Runtime (Minutes)",
            min_value=30,
            max_value=300,
            value=120
        )

        vote_average = st.slider(
            "Vote Average",
            min_value=0.0,
            max_value=10.0,
            value=7.0,
            step=0.1
        )

    st.markdown("---")

    if st.button("🎯 Predict Movie Success"):

        input_data = pd.DataFrame({
            "budget": [budget],
            "popularity": [popularity],
            "runtime": [runtime],
            "vote_average": [vote_average]
        })

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        confidence = max(probability) * 100

        st.markdown("---")

        if prediction == 1:

            st.success(
                f"🎉 Prediction: SUCCESSFUL MOVIE\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        else:

            st.error(
                f"❌ Prediction: NOT SUCCESSFUL\n\n"
                f"Confidence: {confidence:.2f}%"
            )

        st.markdown("---")

        st.subheader("📋 Prediction Summary")

        summary = pd.DataFrame({
            "Feature": [
                "Budget",
                "Popularity",
                "Runtime",
                "Vote Average"
            ],
            "Value": [
                budget,
                popularity,
                runtime,
                vote_average
            ]
        })

        st.table(summary)

# ==========================================================
# ⭐ FEATURE IMPORTANCE
# ==========================================================

elif page == "⭐ Feature Importance":

    st.title("⭐ Feature Importance")
    st.markdown(
        "Understand which features contribute the most to the Random Forest model."
    )

    st.markdown("---")

    

    # Display Table
    st.subheader("📋 Feature Importance Table")

    st.dataframe(
        feature_importance,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Plotly Bar Chart
    fig = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title="Feature Importance"
    )

    fig.update_layout(
        yaxis=dict(categoryorder="total ascending")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Business Interpretation
    st.subheader("💼 Business Interpretation")

    top_feature = feature_importance.iloc[0]["Feature"]

    st.success(
        f"""
**{top_feature}** is the most influential feature in predicting movie success.

This means the Random Forest model relies more heavily on this variable than the others when making predictions.
"""
    )

    st.info("""
### How to interpret Feature Importance

- Higher importance means the feature contributes more to prediction.
- Lower importance does not mean the feature is useless—it simply has less influence relative to the others.
- Feature importance helps stakeholders understand what drives the model's decisions, improving transparency and interpretability.
""")

# ==========================================================
# 💼 BUSINESS INSIGHTS
# ==========================================================

elif page == "💼 Business Insights":

    st.title("💼 Business Insights & Recommendations")
    st.markdown("Executive summary of the MovieIQ project.")

    st.markdown("---")

    # =============================
    # KPI Cards
    # =============================

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "🎥 Movies Analysed",
        f"{len(df):,}"
    )

    c2.metric(
        "🏆 Success Rate",
        f"{df['Success'].mean()*100:.1f}%"
    )

    c3.metric(
        "⭐ Top Predictor",
        feature_importance.iloc[0]["Feature"].title()
    )

    st.markdown("---")

    # =============================
    # Executive Summary
    # =============================

    st.header("📌 Executive Summary")

    st.info("""
MovieIQ uses historical movie data and a Random Forest Machine Learning model
to predict whether a movie is likely to become commercially successful.

The dashboard combines Exploratory Data Analysis, Statistical Testing,
Machine Learning and Interactive Visual Analytics to support
data-driven decision making.
""")

    st.markdown("---")

    # =============================
    # Key Insights
    # =============================

    st.header("📊 Key Insights")

    st.success("""
✅ Higher popularity significantly increases the likelihood of movie success.

✅ Movies with larger production budgets generally achieve higher revenue.

✅ Audience ratings positively influence commercial performance.

✅ Popularity was identified as the strongest predictor by the Random Forest model.
""")

    st.warning("""
⚠ Genre alone was not statistically associated with movie success.

Business decisions should consider multiple factors rather than relying only on genre.
""")

    st.markdown("---")

    # =============================
    # Business Recommendations
    # =============================

    st.header("💡 Recommendations")

    recommendations = [
        "Increase marketing investment to improve audience popularity before release.",
        "Allocate production budgets strategically based on expected returns.",
        "Monitor audience ratings to improve future productions.",
        "Use predictive analytics during green-light decisions.",
        "Combine financial, production and audience metrics for better forecasting."
    ]

    for rec in recommendations:
        st.markdown(f"✅ {rec}")

    st.markdown("---")

    # =============================
    # Final Conclusion
    # =============================

    st.header("🎯 Conclusion")

    st.success("""
MovieIQ demonstrates how Machine Learning can support strategic decisions in the film industry.

Rather than relying on intuition, studios can leverage predictive analytics to estimate
the probability of commercial success before a movie is released.
""")

    st.markdown("---")

    # =============================
    # Feature Importance
    # =============================

    st.header("⭐ Model Insights")

    top_feature = feature_importance.iloc[0]["Feature"]

    st.metric(
        "Most Important Feature",
        top_feature.title()
    )

    st.write(f"""
The Random Forest model identified **{top_feature.title()}**
as the most influential feature when predicting movie success.
""")

    st.markdown("---")

    # =============================
    # Business Recommendations
    # =============================

    st.header("💡 Business Recommendations")

    recommendations = [
        "Increase marketing efforts to improve movie popularity before release.",
        "Allocate production budgets strategically for high-potential projects.",
        "Use audience ratings and feedback to improve movie quality.",
        "Leverage predictive analytics during green-light decisions.",
        "Combine financial, audience, and production metrics instead of relying only on genre."
    ]

    for rec in recommendations:
        st.markdown(f"✅ {rec}")

    st.markdown("---")

    # =============================
    # Final Conclusion
    # =============================

    st.header("🎯 Conclusion")

    st.success("""
MovieIQ demonstrates how Machine Learning can support decision-making
in the film industry.

By combining Exploratory Data Analysis, Statistical Testing,
and Random Forest Classification, the application provides
an interpretable and data-driven approach to predicting
movie success.
""")