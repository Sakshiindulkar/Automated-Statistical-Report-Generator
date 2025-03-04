import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_1samp, ttest_ind, wilcoxon, chi2_contingency, mannwhitneyu
import statsmodels.formula.api as smf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

# anova analysis functions
def clean_columns(df):
    df.columns = [re.sub(r'\W+', '_', col) for col in df.columns]
    return df
def check_normality(data, group_col, value_col):
    groups = data[group_col].unique()
    p_values = []
    for group in groups:
        stat, p = stats.shapiro(data[data[group_col] == group][value_col])
        p_values.append(p)
    return all(p > 0.05 for p in p_values)

def check_homogeneity(data, group_col, value_col):
    groups = [data[data[group_col] == g][value_col] for g in data[group_col].unique()]
    stat, p = stats.levene(*groups)
    return p > 0.05

def one_way_anova(data, group_col, value_col):
    model = smf.ols(f'{value_col} ~ C({group_col})', data=data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return model, anova_table

def two_way_anova(df, factor1, factor2, value_col):
    """Performs Two-Way ANOVA after handling missing and infinite values."""

    # Select only relevant columns
    df_cleaned = df[[factor1, factor2, value_col]].copy()

    # Drop rows with NaN values
    df_cleaned.dropna(inplace=True)

    # Replace infinite values with NaN, then drop them
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(inplace=True)

    # Ensure categorical variables are properly formatted
    df_cleaned[factor1] = df_cleaned[factor1].astype(str)
    df_cleaned[factor2] = df_cleaned[factor2].astype(str)

    # Debugging: Print cleaned data before running ANOVA
    if df_cleaned.isna().sum().sum() > 0:
        raise ValueError("🔴 ERROR: Data still contains NaN values after cleaning!")

    # Ensure at least 2 unique categories exist per factor
    if df_cleaned[factor1].nunique() < 2 or df_cleaned[factor2].nunique() < 2:
        raise ValueError("🔴 ERROR: Factors must have at least 2 unique categories each!")

    # Define ANOVA formula
    formula = f'Q("{value_col}") ~ C(Q("{factor1}")) + C(Q("{factor2}")) + C(Q("{factor1}")):C(Q("{factor2}"))'

    # Fit the model
    model = smf.ols(formula, data=df_cleaned).fit()

    # Generate ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    return model, anova_table

def check_linear_assumptions(df, x_col, y_col):
    # 1. Linearity: Check scatter plot
    sns.scatterplot(x=df[x_col], y=df[y_col])
    plt.title('Linearity Check: Scatter plot')
    plt.show()

    # 2. Independence: Durbin-Watson test
    X = df[[x_col]]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    dw_stat = durbin_watson(model.resid)
    print(f"Durbin-Watson stat: {dw_stat}")
    if dw_stat < 1.5 or dw_stat > 2.5:
        print("Warning: Durbin-Watson test indicates possible autocorrelation in residuals.")
        return False
    
    # 3. Homoscedasticity: Breusch-Pagan test
    _, p_value_bp, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan p-value: {p_value_bp}")
    if p_value_bp < 0.05:
        print("Warning: Homoscedasticity assumption violated (Breusch-Pagan test).")
        return False

    # 4. Normality of residuals: Shapiro-Wilk test
    stat, p_value_shapiro = shapiro(model.resid)
    print(f"Shapiro-Wilk p-value: {p_value_shapiro}")
    if p_value_shapiro < 0.05:
        print("Warning: Residuals are not normally distributed (Shapiro-Wilk test).")
        return False

    return True

# Multiple Linear Regression Assumptions Check
def check_multiple_linear_assumptions(df, x_cols, y_col):
    # 1. Linearity: Check pairplot of independent variables vs dependent variable
    sns.pairplot(df, vars=x_cols + [y_col])
    plt.title('Linearity Check: Pairplot')
    plt.show()

    # 2. Independence: Durbin-Watson test
    X = df[x_cols]
    y = df[y_col]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    dw_stat = durbin_watson(model.resid)
    print(f"Durbin-Watson stat: {dw_stat}")
    if dw_stat < 1.5 or dw_stat > 2.5:
        print("Warning: Durbin-Watson test indicates possible autocorrelation in residuals.")
        return False
    
    # 3. Homoscedasticity: Breusch-Pagan test
    _, p_value_bp, _, _ = het_breuschpagan(model.resid, model.model.exog)
    print(f"Breusch-Pagan p-value: {p_value_bp}")
    if p_value_bp < 0.05:
        print("Warning: Homoscedasticity assumption violated (Breusch-Pagan test).")
        return False

    # 4. Normality of residuals: Shapiro-Wilk test
    stat, p_value_shapiro = shapiro(model.resid)
    print(f"Shapiro-Wilk p-value: {p_value_shapiro}")
    if p_value_shapiro < 0.05:
        print("Warning: Residuals are not normally distributed (Shapiro-Wilk test).")
        return False

    return True

# Logistic Regression Assumptions Check
def check_logistic_assumptions(df, x_cols, y_col):
    # 1. Linearity of the Logit: We need to check if the log-odds of the dependent variable is linear
    for col in x_cols:
        sns.scatterplot(x=df[col], y=np.log(df[y_col] / (1 - df[y_col])))
        plt.title(f'Logit Linearity Check: {col}')
        plt.show()

    # 2. No multicollinearity: Check correlation matrix between predictors
    corr_matrix = df[x_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Multicollinearity Check: Correlation Matrix')
    plt.show()

    if any(abs(corr_matrix) > 0.8).sum() > len(x_cols):
        print("Warning: High multicollinearity detected between predictors.")
        return False

    return True

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
#####
st.set_page_config(
    page_title="App Overview", 
    page_icon="📊", 
    layout="wide"
)

st.title("📊 Automated Statistical Report Generator")
st.markdown(
    """
    ## Overview
    The Automated Statistical Report Generator is a powerful web application designed 
    to streamline statistical analysis and reporting. It provides users with a comprehensive 
    statistical summary, including descriptive statistics, correlation analysis, regression modeling, 
    hypothesis testing, and ANOVA. With an intuitive interface and automated data processing, this tool 
    is ideal for researchers, analysts, and students looking for accurate and insightful statistical reports.
    """)
    
st.title(" Key Features")    
st.markdown(
    """
    ## 🔹 Descriptive Statistics
    - Summarize data with mean, median, standard deviation, variance, skewness, kurtosis, and more.
    - Visualize distributions using histograms and boxplots.
    """)
st.markdown(
    """
    ## 🔹 Correlation Analysis
    - Generate a correlation heatmap to visualize relationships between variables.
    - Provide detailed interpretations of correlation strength and direction.
    """)
st.markdown(
    """
    ## 🔹 Regression Analysis
    - Perform linear, multiple, and logistic regression with detailed model summaries.
    - Check for assumptions like multicollinearity, normality, and homoscedasticity before running regression.
    - Provide interpretations of regression coefficients for better decision-making.
    - For logistic regression, analyze relationships between categorical outcomes and predictor variables.
    """)
st.markdown(
    """
    ## 🔹 Hypothesis Testing
    - Conduct various hypothesis tests, including t-tests, chi-square tests, and proportion tests.
    - Provide detailed interpretations, including whether to accept or reject the null hypothesis.
    """)
st.markdown(
    """
    ## 🔹 ANOVA (Analysis of Variance)
    - Perform One-Way and Two-Way ANOVA to compare means across multiple groups.
    - Generate ANOVA tables with F-statistics, p-values, and post-hoc tests for deeper insights.
    """)
st.markdown(
    """
    ## 🔹 Automated Report Generation
    - The app automatically cleans and preprocesses data before performing statistical analysis.
    - Generate comprehensive reports with tables, figures, and statistical interpretations based on user-provided data.
    """)

st.title(" Why Use This Tool?")

st.markdown(
    """✅ Saves Time - No need for manual calculations; get results instantly.
    """)
st.markdown(
    """✅ User-Friendly - No coding required, just upload your data and analyze.
    """)
st.markdown(
    """
    ✅ Accurate & Reliable - Uses validated statistical methods for high precision.
    """)
st.markdown(
    """✅ Customizable Reports - Get detailed reports with insights tailored to your dataset.
    """)

st.markdown(
    """🚀 Upload your dataset and let the Automated Statistical Report Generator handle the analysis for you!    
    """)

st.image('https://i.pinimg.com/1200x/95/fe/7a/95fe7a6efd8ed2054a5552dda2d78731.jpg', 
         caption="Automated Data Analysis Made Simple", 
         use_container_width=True)

st.sidebar.title("Navigate")
st.sidebar.write("Use the navigation panel to explore different features of the app.")


st.title("""Created by Group D:""")
st.markdown("Sakshi Indulkar (909)")
st.markdown("Tripti Yadav (939)")
st.markdown("Shruthi Thootey (935)")
st.markdown("Mangesh Patel (942)")
st.markdown("Divya Jain (913)")
st.markdown("Himanshu Pandey (921)")
st.markdown("Maheshwari Yadav (937)")
 

# Apply Light Lavender and Light Mint Green Gradient background with black text
st.markdown(
    """
    <style>
        body {
            background: linear-gradient(to bottom, #D1C4E9, #A5D6A7);  /* Light Lavender to Light Mint Green Gradient */
            color: black;  /* Black text for better readability on a light background */
        }
        .block-container {
            background: linear-gradient(to bottom, #D1C4E9, #A5D6A7);  /* Keep the content container background same */
            color: black;  /* Black text in content area */
        }
        .css-18e3t6p {
            color: black;  /* Ensure any other texts are also black */
        }
    </style>
    """, unsafe_allow_html=True
)

app_mode = st.sidebar.radio("  Select from Below ✅  ", ["Descriptive Statistics", "Correlation Graphs", "Regression Analysis", "Hypothesis Testing","ANOVA"])

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
    # Descriptive Statistics Page
    if app_mode == "Descriptive Statistics":
        st.header("Descriptive Statistics")
        st.write("Here is the summary of the data including mean, median, standard deviation, and more:")
        st.write(df.describe())  # Show summary statistics

        st.write("Interpretation:")
        st.write("""
            - Count: Number of non-null values for each feature.
            - Mean: The average of the values.
            - Standard Deviation: Measures the dispersion of the data from the mean.
            - Min/Max: The minimum and maximum values in the dataset.
            - 25%, 50%, 75%: These are the percentiles of the data, showing the spread of the data.
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

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig)

                # Finding the strongest positive and negative correlations
            strongest_positive = None
            strongest_negative = None
            max_pos_corr = 0  # Track highest positive correlation
            max_neg_corr = 0  # Track highest negative correlation

            for col1 in correlation_matrix.columns:
                for col2 in correlation_matrix.columns:
                    if col1 != col2:
                        corr_value = correlation_matrix.loc[col1, col2]

                        if corr_value > max_pos_corr:
                            max_pos_corr = corr_value
                            strongest_positive = (col1, col2, corr_value)

                        if corr_value < max_neg_corr:
                            max_neg_corr = abs(corr_value)
                            strongest_negative = (col1, col2, corr_value)

                 # Interpretation based on dataset
            st.write("### Interpretation of the Correlation Heatmap:")

            if strongest_positive:
                st.write(f"✅ *Strongest Positive Correlation:* {strongest_positive[0]} and {strongest_positive[1]} with a correlation of *{strongest_positive[2]:.2f}*.")
                if strongest_positive[2] > 0.8:
                    st.write("📈 These features are highly correlated, indicating they may carry similar information.")
                elif strongest_positive[2] > 0.5:
                    st.write("📊 These features show a moderate correlation, which could be useful for predictive analysis.")

            if strongest_negative:
                st.write(f"❌ *Strongest Negative Correlation:* {strongest_negative[0]} and {strongest_negative[1]} with a correlation of *{strongest_negative[2]:.2f}*.")
                if strongest_negative[2] < -0.8:
                    st.write("🔻 These features are strongly inversely related, i.e. when one increases, the other decreases significantly.")
                elif strongest_negative[2] < -0.5:
                    st.write("🔄 These features have a moderate inverse relationship, which may be useful for understanding dependencies.")
                else:
                    st.write("⚠️ The strongest negative correlation is weak, i.e. there is little to no inverse relationship.")

                # Identify potential multicollinearity issues
            multicollinear_features = [
                (col1, col2, correlation_matrix.loc[col1, col2])
                for col1 in correlation_matrix.columns
                for col2 in correlation_matrix.columns
                if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > 0.85
            ]

            if multicollinear_features:
                st.write("*Potential Multicollinearity Detected:* The following feature pairs have very high correlation (> 0.85), which might affect regression models:")
                for col1, col2, corr in multicollinear_features:
                    st.write(f"- {col1} and {col2}: Correlation = *{corr:.2f}*")
                st.write("🔹 Consider removing one of the highly correlated features to avoid redundancy in regression models.")

        else:
            st.write("No numeric columns found in the dataset for correlation.")

    elif app_mode == "Regression Analysis":
        st.write("### Regression Analysis")
        regression_type = st.radio("Choose Regression Type", 
                                ("Simple Linear Regression", "Multiple Linear Regression", "Logistic Regression"))
    
        # Simple Linear Regression
        if regression_type == "Simple Linear Regression":
            st.write("### Simple Linear Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            x_col = st.selectbox("Select Predictor Variable", numeric_columns)
            y_col = st.selectbox("Select Response Variable", numeric_columns)
            if st.button("Check Assumptions & Run Regression"):
                sns.scatterplot(x=df[x_col], y=df[y_col])
                plt.title('Linearity Check: Scatter plot')
                st.pyplot(plt)

                if x_col and y_col:
                    st.write(f"### Results for Simple Linear Regression (Predicting {y_col} from {x_col})")
                    X = sm.add_constant(df[x_col])  # Add intercept
                    Y = df[y_col]
                    
                    model = sm.OLS(Y, X).fit()
                    sns.scatterplot(x=model.fittedvalues, y=model.resid)
                    plt.title('Homoscedasticity Check: Residuals vs Fitted values')
                    st.pyplot(plt)

                    st.write(model.summary())

                    # Interpretation Based on Output
                    st.write("#### Interpretation:")

                    # Ensure that the model has at least one predictor before accessing index 1
                    if len(model.pvalues) > 1:  
                        p_value = model.pvalues[1]  # p-value of the predictor
                        coefficient = model.params[1]  # Coefficient of the predictor

                        # Interpretation for p-value
                        if p_value < 0.05:
                            st.write(f"- *p-value*: The p-value of {p_value:.4f} is less than 0.05, indicating that the predictor variable {x_col} is statistically significant in predicting {y_col}.")
                        else:
                            st.write(f"- *p-value*: The p-value of {p_value:.4f} is greater than 0.05, suggesting that {x_col} is not statistically significant in predicting {y_col}.")
                        
                        # Interpretation for Coefficient
                        st.write(f"- *Coefficient*: The coefficient of {x_col} is {coefficient:.4f}, meaning that for each unit increase in {x_col}, {y_col} is expected to change by {coefficient:.4f} units, holding all other factors constant.")
                    else:
                        st.write("⚠️ No valid predictor variables found in the model. Only the intercept is present.")

                    # Interpretation for R-squared
                    r_squared = model.rsquared  # R-squared value
                    st.write(f"- *R-squared*: The R-squared value is {r_squared:.4f}, which means that approximately {r_squared*100:.2f}% of the variance in {y_col} is explained by {x_col}.")
                    if r_squared < 0.3:
                        st.write("- This indicates a weak fit, suggesting other factors may be influencing the response variable.")
                    elif r_squared > 0.7:
                        st.write("- This suggests a good fit, where the model explains a significant portion of the variance.")
                    else:
                        st.write("- This suggests a moderate fit.")

        # Multiple Linear Regression
        if regression_type == "Multiple Linear Regression":
            st.write("### Multiple Linear Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            x_cols = st.multiselect("Select Predictor Variables", numeric_columns)
            y_col = st.selectbox("Select Response Variable", numeric_columns)
            if st.button("Check Assumptions & Run Regression"):
                sns.pairplot(df, vars=x_cols+[y_col])
                plt.title('Linearity Check: Pairplot')
                st.pyplot(plt)

                if x_cols and y_col:
                    st.write(f"### Results for Multiple Linear Regression (Predicting {y_col} from {', '.join(x_cols)})")
                    X = sm.add_constant(df[x_cols])  # Add intercept
                    Y = df[y_col]

                    model = sm.OLS(Y, X).fit()
                    sns.scatterplot(x=model.fittedvalues, y=model.resid)
                    plt.title('Homoscedasticity Check: Residuals vs Fitted values')
                    st.pyplot(plt)

                    st.write(model.summary())

                    # Interpretation Based on Output
                    st.write("#### Interpretation:")
                    num_predictors = len(model.pvalues) - 1  # Exclude intercept
                    for i, predictor in enumerate(x_cols):
                        if i < num_predictors:  # Avoid out-of-bounds error
                            p_value = model.pvalues[i + 1]  # p-value of the predictor
                            coefficient = model.params[i + 1]  # Coefficient of the predictor

                            if p_value < 0.05:
                                st.write(f"- **p-value** of {predictor}: The p-value of {p_value:.4f} is less than 0.05, indicating that {predictor} is statistically significant in predicting {y_col}.")
                            else:
                                st.write(f"- **p-value** of {predictor}: The p-value of {p_value:.4f} is greater than 0.05, suggesting that {predictor} is not statistically significant in predicting {y_col}.")
            
            # Interpretation for Coefficient
                            st.write(f"- **Coefficient** of {predictor}: The coefficient is {coefficient:.4f}, meaning that for each unit increase in {predictor}, {y_col} is expected to change by {coefficient:.4f} units, holding other variables constant.")
                        else:
                            st.write(f"⚠️ Warning: {predictor} is not present in model.pvalues.")
                        
                        
                    # VIF (Variance Inflation Factor) for multicollinearity check
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    vif_data = pd.DataFrame()
                    vif_data["Variable"] = X.columns
                    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
                    st.write("#### Variance Inflation Factor (VIF):")
                    st.write(vif_data)
                    for i, vif in enumerate(vif_data["VIF"]):
                        if vif > 10:
                            st.write(f"- *VIF* of {vif_data['Variable'][i]} is {vif:.2f}, indicating potential multicollinearity.")

        # Logistic Regression
        if regression_type == "Logistic Regression":
            st.write("### Logistic Regression")
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=[object, 'category']).columns.tolist()  # Ensure categorical target
            x_cols = st.multiselect("Select Predictor Variables", numeric_columns)
            y_col = st.selectbox("Select Response Variable (Binary)", categorical_columns)
            if st.button("Check Assumptions & Run Regression"):
                if x_cols and y_col:
                    st.write(f"### Results for Logistic Regression (Predicting {y_col} from {', '.join(x_cols)})")
                    X = df[x_cols]
                    Y = df[y_col].astype('category').cat.codes  # Convert categorical target to numerical 0/1

                    # Logistic Regression Model
                    model = LogisticRegression()
                    model.fit(X, Y)
                    predictions = model.predict(X)

                    # Calculate accuracy
                    accuracy = accuracy_score(Y, predictions)

                    # Confusion Matrix
                    cm = confusion_matrix(Y, predictions)

                    # Display results
                    st.write(f"*Accuracy*: {accuracy:.2f}")
                    st.write("#### Confusion Matrix:")
                    st.write(cm)

                    # Detailed performance metrics
                    precision = precision_score(Y, predictions, average='weighted')  # or 'macro', 'micro'
                    recall = recall_score(Y, predictions, average='weighted')  # or 'macro', 'micro'
                    f1 = f1_score(Y, predictions, average='weighted')  # or 'macro', 'micro'

                    st.write(f"*Precision*: {precision:.2f}")
                    st.write(f"*Recall*: {recall:.2f}")
                    st.write(f"*F1-Score*: {f1:.2f}")

                    # Classification Report
                    st.write("#### Classification Report:")
                    st.text(classification_report(Y, predictions))

                    # Confusion Matrix Heatmap
                    st.write("#### Confusion Matrix Heatmap:")
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title('Confusion Matrix Heatmap')
                    st.pyplot(fig)

                    # Interpretation Based on Metrics
                    st.write("#### Interpretation:")
                    st.write(f"- *Accuracy*: The accuracy of {accuracy:.2f} means that {accuracy * 100:.2f}% of the predictions are correct.")
                    if accuracy < 0.5:
                        st.write("- This suggests that the model is performing poorly, and further tuning or additional features might be needed.")
                    elif accuracy > 0.8:
                        st.write("- This suggests that the model is performing well.")
                    else:
                        st.write("- The accuracy is moderate, indicating a need for further improvements.")
                    
                    st.write(f"- *Precision*: The precision of {precision:.2f} indicates that {precision * 100:.2f}% of the predicted positive cases are actually positive.")
                    st.write(f"- *Recall*: The recall of {recall:.2f} indicates that {recall * 100:.2f}% of the actual positive cases are correctly identified.")
                    st.write(f"- *F1-Score*: The F1-score of {f1:.2f} indicates a balance between precision and recall. Higher values are preferable.")

        
        # Hypothesis Testing
    elif app_mode == "Hypothesis Testing":
        st.write("### Hypothesis Testing Page")
                # Hypothesis Testing Section
        test_type = st.radio("Choose Test Type", 
                            ("One-sample t-test", "Two-sample t-test", "Paired t-test", 
                            "Chi-square Test", "Mann-Whitney U Test", "Wilcoxon Signed-Rank Test"))


                    # One-sample t-test
        if test_type == "One-sample t-test":
            column = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns.tolist())
            value = st.number_input("Enter the value for comparison", value=0)
            if st.button("Perform Analysis"):
                stat, p = one_sample_ttest(df, column, value)
                st.write(f"*One-sample t-test Result*")
                st.write(f"t-statistic: {stat}, p-value: {p}")
                
                            # Interpretation based on p-value
                st.write("#### Interpretation:")
                st.write("- The one-sample t-test is used to compare the sample mean to a known value.")

                if p < 0.05:
                    st.write("p value =",p,"< 0.05"," Therefore, *Reject the null hypothesis*: The sample mean is significantly different from the given value.")
                else:
                    st.write("p value =",p,"> 0.05"," Therefore, *Fail to reject the null hypothesis*: There is no significant difference between the sample mean and the given value.")

                    # Two-sample t-test
        elif test_type == "Two-sample t-test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Analysis"):
                stat, p = two_sample_ttest(df, column1, column2)
                st.write(f"*Two-sample t-test Result*")
                st.write(f"t-statistic: {stat}, p-value: {p}")
                
                            # Interpretation based on p-value
                st.write("#### Interpretation:")
                st.write("- The two-sample t-test compares the means of two independent groups.")
                if p < 0.05:
                    st.write("p value =",p,"< 0.05"," Therefore,*Reject the null hypothesis*: There is a significant difference in means between the two groups.")
                else:
                    st.write("p value =",p,"> 0.05"," Therefore, *Fail to reject the null hypothesis*: There is no significant difference in means between the two groups.")

                    # Paired t-test (Wilcoxon Signed-Rank Test for non-parametric)
        elif test_type == "Paired t-test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Analysis"):
                stat, p = wilcoxon_test(df, column1, column2)
                st.write(f"*Paired t-test Result*")
                st.write(f"t-statistic: {stat}, p-value: {p}")
                
                            # Interpretation based on p-value
                st.write("#### Interpretation:")
                if p < 0.05:
                    st.write("p value =",p,"< 0.05"," Therefore, *Reject the null hypothesis*: There is a significant difference between the paired samples.")
                else:
                    st.write("p value =",p,"> 0.05"," Therefore, *Fail to reject the null hypothesis*: There is no significant difference between the paired samples.")

                    # Chi-square Test
        elif test_type == "Chi-square Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[object]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[object]).columns.tolist())
            if st.button("Perform Analysis"):
                stat, p = chi_square_test(df, column1, column2)
                st.write(f"*Chi-square Test Result*")
                st.write(f"Chi-square statistic: {stat}, p-value: {p}")
                
                            # Interpretation based on p-value
                st.write("#### Interpretation:")
                st.write("- The Chi-square test assesses whether there is an association between two categorical variables.")
                if p < 0.05:
                    st.write("p value =",p,"< 0.05"," Therefore, *Reject the null hypothesis*: There is a significant association between the two categorical variables.")
                else:
                    st.write("p value =",p,"> 0.05"," Therefore, *Fail to reject the null hypothesis*: There is no significant association between the two categorical variables.")

                    # Mann-Whitney U Test
        elif test_type == "Mann-Whitney U Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Analysis"):
                stat, p = mann_whitney_u_test(df, column1, column2)
                st.write(f"*Mann-Whitney U Test Result*")
                st.write(f"U-statistic: {stat}, p-value: {p}")
                
                            # Interpretation based on p-value
                st.write("#### Interpretation:")
                st.write("- The Mann-Whitney U test is a non-parametric test used to compare distributions between two independent groups.")
                if p < 0.05:
                    st.write("p value =",p,"< 0.05"," Therefore, *Reject the null hypothesis*: There is a significant difference between the distributions of the two groups.")
                else:
                    st.write("p value =",p,"> 0.05"," Therefore, *Fail to reject the null hypothesis*: There is no significant difference between the distributions of the two groups.")

                    # Wilcoxon Signed-Rank Test
        elif test_type == "Wilcoxon Signed-Rank Test":
            column1 = st.selectbox("Select First Column", df.select_dtypes(include=[np.number]).columns.tolist())
            column2 = st.selectbox("Select Second Column", df.select_dtypes(include=[np.number]).columns.tolist())
            if st.button("Perform Analysis"):
                stat, p = wilcoxon_test(df, column1, column2)
                st.write(f"*Wilcoxon Signed-Rank Test Result*")
                st.write(f"Wilcoxon statistic: {stat}, p-value: {p}")
                
                            # Interpretation based on p-value
                st.write("#### Interpretation:")
                st.write("- The Wilcoxon Signed-Rank test is used to test differences between paired samples.")
                if p < 0.05:
                    st.write("p value =",p,"< 0.05"," Therefore, *Reject the null hypothesis*: There is a significant difference between the paired samples.")
                else:
                    st.write("p value =",p,"> 0.05"," Therefore, *Fail to reject the null hypothesis*: There is no significant difference between the paired samples.")
## Anova analysis
    elif app_mode == "ANOVA":
        analysis_type = st.selectbox("Select ANOVA type", ["One-Way ANOVA", "Two-Way ANOVA"])
    
        if analysis_type == "One-Way ANOVA":
            group_col = st.selectbox("Select Grouping Variable", df.select_dtypes(include=['object', 'category']).columns)
            value_col = st.selectbox("Select Dependent Variable", df.select_dtypes(include=['number']).columns)
            
            if st.button("Run One-Way ANOVA"):
                if check_normality(df, group_col, value_col) and check_homogeneity(df, group_col, value_col):
                    st.success("Assumptions met: Normality and Homogeneity hold.")
                    model, anova_table = one_way_anova(df, group_col, value_col)
                    st.write("### One-Way ANOVA Table")
                    st.write(anova_table)
                    
                    if anova_table["PR(>F)"][0] < 0.05:
                        st.write("*Interpretation:* There is a significant difference between groups.")
                    else:
                        st.write("*Interpretation:* No significant difference between groups.")
                else:
                    st.warning("Assumption Check Failed: Data does not meet ANOVA assumptions.")

        elif analysis_type == "Two-Way ANOVA":
            factor1 = st.selectbox("Select First Factor", df.select_dtypes(include=['object', 'category']).columns)
            factor2 = st.selectbox("Select Second Factor", [col for col in df.select_dtypes(include=['object', 'category']).columns if col != factor1])
            value_col = st.selectbox("Select Dependent Variable", df.select_dtypes(include=['number']).columns)
            
            if st.button("Run Two-Way ANOVA"):
                if check_normality(df, factor1, value_col) and check_homogeneity(df, factor1, value_col) and check_normality(df, factor2, value_col) and check_homogeneity(df, factor2, value_col):
                    st.success("Assumptions met: Normality and Homogeneity hold.")
                    model, anova_table = two_way_anova(df, factor1, factor2, value_col)
                    st.write("### Two-Way ANOVA Table")
                    st.write(anova_table)
                    
                    if anova_table["PR(>F)"][0] < 0.05:
                        st.write(f"*Interpretation:* {factor1} has a significant effect on {value_col}.")
                    else:
                        st.write(f"*Interpretation:* {factor1} does not have a significant effect on {value_col}.")
                    
                    if anova_table["PR(>F)"][1] < 0.05:
                        st.write(f"*Interpretation:* {factor2} has a significant effect on {value_col}.")
                    else:
                        st.write(f"*Interpretation:* {factor2} does not have a significant effect on {value_col}.")
                    
                    if anova_table["PR(>F)"][2] < 0.05:
                        st.write("*Interpretation:* There is a significant interaction effect between the factors.")
                    else:
                        st.write("*Interpretation:* No significant interaction effect between the factors.")
                else:
                    st.warning("Assumption Check Failed: Data does not meet ANOVA assumptions.")
