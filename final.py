import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Regression Analysis Functions
def simple_linear_regression(df, x_col, y_col):
    X = df[[x_col]]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

def multiple_linear_regression(df, x_cols, y_col):
    X = df[x_cols]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.summary()

def logistic_regression(df, x_cols, y_col):
    X = df[x_cols]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, cm

# Hypothesis Testing Functions
def one_sample_ttest(df, column, value):
    stat, p = stats.ttest_1samp(df[column].dropna(), value)
    return stat, p

def two_sample_ttest(df, column1, column2):
    stat, p = stats.ttest_ind(df[column1].dropna(), df[column2].dropna())
    return stat, p

def paired_ttest(df, column1, column2):
    stat, p = stats.ttest_rel(df[column1].dropna(), df[column2].dropna())
    return stat, p

def chi_square_test(df, column1, column2):
    contingency_table = pd.crosstab(df[column1], df[column2])
    stat, p, _, _ = stats.chi2_contingency(contingency_table)
    return stat, p

def mann_whitney_u_test(df, column1, column2):
    stat, p = mannwhitneyu(df[column1].dropna(), df[column2].dropna())
    return stat, p

def wilcoxon_test(df, column1, column2):
    stat, p = wilcoxon(df[column1].dropna(), df[column2].dropna())
    return stat, p
# Set the title of the app
st.title("Automated Statistical Report Generator")

# Create a file uploader for users to upload a CSV file
uploaded_file = st.file_uploader("Upload your data file (CSV)", type="csv")
# Check if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

     # Show DataFrame
    st.write("### Preview of Data:")
    st.dataframe(df)

    # Show column names
    st.write("### Column Names:")
    st.write(df.columns.tolist())


    # Add a sidebar for navigation
    st.sidebar.title("Navigate")
    app_mode = st.sidebar.radio("Select a page", ["Descriptive Statistics", "Correlation Graphs", "Regression Analysis", "Hypothesis Testing"])

    # Descriptive Statistics Page
    if app_mode == "Descriptive Statistics":
        st.header("Descriptive Statistics")
        st.write("Here is the summary of the data including mean, median, standard deviation, and more:")
        st.write(df.describe())  # Show summary statistics

        st.write("Interpretation:")
        st.write("""
            - *Count*: Number of non-null values for each feature.
            - *Mean*: The average of the values.
            - *Standard Deviation*: Measures the dispersion of the data from the mean.
            - *Min/Max*: The minimum and maximum values in the dataset.
            - *25%, 50%, 75%*: These are the percentiles of the data, showing the spread of the data.
        """)

    # Correlation Graphs Page
    elif app_mode == "Correlation Graphs":
        st.header("Correlation Graphs")
        st.write("Below is the correlation heatmap of the numeric features in your dataset:")

     # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

    # Check if there are numeric columns in the DataFrame
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()

            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot()

            st.write("Interpretation:")
            st.write("""
                - Correlation coefficients range from -1 to 1. A positive value indicates a direct relationship, while a negative value indicates an inverse relationship.
                - The closer the value is to 1 or -1, the stronger the relationship.
                - The heatmap visually helps to identify highly correlated features.
            """)
        else:
            st.write("No numeric columns found in the dataset for correlation.")

    if app_mode == "Regression Analysis":
        st.write("### Regression Analysis")
        regression_type = st.radio("Choose Regression Type", 
                               ("Simple Linear Regression", "Multiple Linear Regression", "Logistic Regression"))

    # Simple Linear Regression
        if regression_type == "Simple Linear Regression":
            st.write("### Simple Linear Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            x_col = st.selectbox("Select Predictor Variable", numeric_columns)
            y_col = st.selectbox("Select Response Variable", numeric_columns)
        if x_col and y_col:
            st.write(f"### Results for Simple Linear Regression (Predicting {y_col} from {x_col})")
            X = sm.add_constant(df[x_col])  # Add intercept
            Y = df[y_col]

            model = sm.OLS(Y, X).fit()
            st.write(model.summary())

            st.write("#### Interpretation:")
            st.write("- The coefficient represents the change in the response variable for a one-unit increase in the predictor.")
            st.write("- A significant p-value (<0.05) suggests a meaningful relationship.")
            st.write("- The R-squared value indicates the percentage of variability explained by the predictor.")

    # Multiple Linear Regression
        elif regression_type == "Multiple Linear Regression":
            st.write("### Multiple Linear Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            x_cols = st.multiselect("Select Predictor Variables", numeric_columns)
            y_col = st.selectbox("Select Response Variable", numeric_columns)

        if x_cols and y_col:
            st.write(f"### Results for Multiple Linear Regression (Predicting {y_col} from {', '.join(x_cols)})")
            X = sm.add_constant(df[x_cols])  # Add intercept
            Y = df[y_col]

            model = sm.OLS(Y, X).fit()
            st.write(model.summary())

            st.write("#### Interpretation:")
            st.write("- Each coefficient shows the effect of the predictor while holding others constant.")
            st.write("- A high R-squared suggests a good fit, but very high may indicate overfitting.")
            st.write("- Check Variance Inflation Factor (VIF) to avoid multicollinearity.")

    # Logistic Regression
        elif regression_type == "Logistic Regression":
            st.write("### Logistic Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=[object, 'category']).columns.tolist()  # Ensure categorical target
            x_cols = st.multiselect("Select Predictor Variables", numeric_columns)
            y_col = st.selectbox("Select Response Variable (Binary)", categorical_columns)

        if x_cols and y_col:
            st.write(f"### Results for Logistic Regression (Predicting {y_col} from {', '.join(x_cols)})")
            X = df[x_cols]
            Y = df[y_col].astype('category').cat.codes  # Convert categorical target to numerical 0/1

            model = LogisticRegression()
            model.fit(X, Y)
            predictions = model.predict(X)

            accuracy = accuracy_score(Y, predictions)
            cm = confusion_matrix(Y, predictions)

            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Confusion Matrix:\n{cm}")

            st.write("#### Interpretation:")
            st.write("- Accuracy represents the proportion of correct predictions.")
            st.write("- Confusion matrix shows TP, TN, FP, FN.")
            st.write("- High accuracy with balanced classes suggests a good model.")

    if app_mode == "Hypothesis Testing":
        st.write("### Hypothesis Testing")
        test_type = st.radio("Choose Test Type", ("One-sample t-test", "Two-sample t-test", "Paired t-test", "Chi-square Test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"))

        if test_type == "One-sample t-test":
            column = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns.tolist())
            value = st.number_input("Enter the value for comparison", value=0)
        if st.button("Perform One-sample t-test"):
            stat, p = one_sample_ttest(df, column, value)
            st.write(f"t-statistic: {stat}, p-value: {p}")
            st.write("#### Interpretation:")
            st.write("- A p-value < 0.05 suggests that the sample mean is significantly different from the given value.")

        elif test_type == "Two-sample t-test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
        if st.button("Perform Two-sample t-test"):
            stat, p = two_sample_ttest(df, column1, column2)
            st.write(f"t-statistic: {stat}, p-value: {p}")
            st.write("#### Interpretation:")
            st.write("- A significant p-value indicates a difference in means between the two groups.")

        elif test_type == "Paired t-test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
        if st.button("Perform Paired t-test"):
            stat, p = wilcoxon(df[column1].dropna(), df[column2].dropna())  # Paired test uses Wilcoxon if non-parametric
            st.write(f"Paired t-statistic: {stat}, p-value: {p}")
            st.write("#### Interpretation:")
            st.write("- A p-value < 0.05 suggests a significant difference in the paired samples.")

        elif test_type == "Chi-square Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[object]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[object]).columns.tolist())
        if st.button("Perform Chi-square Test"):
            stat, p = chi_square_test(df, column1, column2)
            st.write(f"Chi-square statistic: {stat}, p-value: {p}")
            st.write("#### Interpretation:")
            st.write("- A p-value < 0.05 suggests a significant association between the two categorical variables.")

        elif test_type == "Mann-Whitney U Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
        if st.button("Perform Mann-Whitney U Test"):
            stat, p = mann_whitney_u_test(df, column1, column2)
            st.write(f"U-statistic: {stat}, p-value: {p}")
            st.write("#### Interpretation:")
            st.write("- A p-value < 0.05 suggests a significant difference in distributions between the two groups.")

        elif test_type == "Wilcoxon Signed-Rank Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
        if st.button("Perform Wilcoxon Signed-Rank Test"):
            stat, p = wilcoxon_test(df, column1, column2)
            st.write(f"Wilcoxon statistic: {stat}, p-value: {p}")
            st.write("#### Interpretation:")
            st.write("- A p-value < 0.05 suggests a significant difference in paired samples.")
