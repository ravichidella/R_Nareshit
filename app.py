import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import get_model_results
#this is a test file

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Telecom Churn Analytics page",
    page_icon="📡",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv("telecom_churn_data.csv")

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📡 Telecom Churn Analytics Dashboard")

st.markdown(
    "### Customer behavior, usage and churn analysis"
)

st.divider()

# --------------------------------------------------
# KPI CALCULATIONS
# --------------------------------------------------

total_customers = len(df)

churned_customers = df["churn"].eq(1).sum()

non_churned_customers = df["churn"].eq(0).sum()

churn_rate = (
    churned_customers / df["churn"].notna().sum()
) * 100

# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👥 Total Customers",
        value=f"{total_customers:,}"
    )

with col2:
    st.metric(
        label="📉 Churned Customers",
        value=f"{churned_customers:,}"
    )

with col3:
    st.metric(
        label="✅ Active Customers",
        value=f"{non_churned_customers:,}"
    )

with col4:
    st.metric(
        label="📊 Churn Rate",
        value=f"{churn_rate:.2f}%"
    )

st.divider()

# --------------------------------------------------
# CHURN DISTRIBUTION
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("📊 Churn Distribution")

    churn_data = df["churn"].value_counts(dropna=False).reset_index()

    churn_data.columns = ["Churn", "Customers"]

    churn_data["Churn"] = churn_data["Churn"].map({
        0.0: "Not Churned",
        1.0: "Churned"
    }).fillna("Missing")

    fig = px.bar(
        churn_data,
        x="Churn",
        y="Customers",
        text="Customers",
        title="Customer Churn Count"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# GENDER DISTRIBUTION
# --------------------------------------------------

with col2:

    st.subheader("👨‍👩‍👧 Gender Distribution")

    gender_data = (
        df["gender"]
        .value_counts(dropna=False)
        .reset_index()
    )

    gender_data.columns = ["Gender", "Customers"]

    gender_data["Gender"] = gender_data["Gender"].fillna(
        "Missing"
    )

    fig = px.pie(
        gender_data,
        names="Gender",
        values="Customers",
        hole=0.4,
        title="Customer Gender Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------

st.divider()

st.subheader("📋 Data Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)

# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("ℹ️ Dataset Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with info_col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with info_col3:
    st.metric(
        "Total Cells",
        df.size
    )

# --------------------------------------------------
# STEP 3 - COLUMN TYPE ANALYSIS
# --------------------------------------------------

st.divider()

st.header("🔤 Step 3: Categorical & Numerical Columns")

# Identify categorical columns
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

# Identify numerical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

col1, col2 = st.columns(2)

with col1:

    st.subheader("🔤 Categorical Columns")

    st.metric(
        "Number of Categorical Columns",
        len(categorical_columns)
    )

    for column in categorical_columns:
        st.write(f"• `{column}`")

with col2:

    st.subheader("🔢 Numerical Columns")

    st.metric(
        "Number of Numerical Columns",
        len(numerical_columns)
    )

    for column in numerical_columns:
        st.write(f"• `{column}`")

# --------------------------------------------------
# STEP 4 - MISSING VALUE ANALYSIS
# --------------------------------------------------

st.divider()

st.header("🧹 Step 4: Missing Value Analysis")

# Calculate missing values
missing_count = df.isnull().sum()

missing_percentage = (
    df.isnull().sum() / len(df) * 100
)

missing_data = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": missing_count.values,
    "Missing %": missing_percentage.values
})

# Keep only columns having missing values
missing_data = missing_data[
    missing_data["Missing Values"] > 0
].sort_values(
    by="Missing Values",
    ascending=False
)

if missing_data.empty:

    st.success("🎉 No missing values found!")

else:

    st.subheader("📋 Missing Value Summary")

    st.dataframe(
        missing_data,
        use_container_width=True,
        hide_index=True
    )

    # Missing value chart
    fig = px.bar(
        missing_data,
        x="Column",
        y="Missing Values",
        text="Missing Values",
        title="Missing Values by Column"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

df = pd.read_csv("telecom_churn_data.csv")

# --------------------------------------------------
# STEP 4B - HANDLE MISSING VALUES
# --------------------------------------------------

# Fill missing gender values with the mode
df["gender"] = df["gender"].fillna(
    df["gender"].mode()[0]
)

# Fill missing numerical values with median
df["maximum_days_inactive"] = df[
    "maximum_days_inactive"
].fillna(
    df["maximum_days_inactive"].median()
)

# --------------------------------------------------
# VERIFY MISSING VALUES
# --------------------------------------------------

st.subheader("✅ Missing Value Check After Treatment")

remaining_missing = df.isnull().sum()

remaining_missing = remaining_missing[
    remaining_missing > 0
]

if remaining_missing.empty:

    st.success(
        "🎉 All feature missing values have been handled!"
    )

else:

    remaining_missing_df = pd.DataFrame({
        "Column": remaining_missing.index,
        "Remaining Missing Values": remaining_missing.values
    })

    st.dataframe(
        remaining_missing_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "The remaining missing churn values are intentionally "
        "kept because churn is the target variable."
    )


# --------------------------------------------------
# STEP 5 - CATEGORICAL DATA ANALYSIS
# --------------------------------------------------

st.divider()

st.header("🔤 Step 5: Categorical Data Analysis")

# Meaningful categorical columns
categorical_analysis_columns = [
    "gender",
    "multi_screen",
    "mail_subscribed"
]

# Select a categorical column
selected_cat = st.selectbox(
    "Select a categorical column:",
    categorical_analysis_columns
)

st.subheader(f"📊 Analysis of: {selected_cat}")

# --------------------------------------------------
# UNIQUE VALUES
# --------------------------------------------------

unique_values = df[selected_cat].unique()

st.write("### 1️⃣ Unique Values")

st.write(unique_values.tolist())

# --------------------------------------------------
# NUMBER OF UNIQUE VALUES
# --------------------------------------------------

number_unique = df[selected_cat].nunique()

st.write("### 2️⃣ Number of Unique Values")

st.metric(
    "Unique Categories",
    number_unique
)

# --------------------------------------------------
# VALUE COUNTS
# --------------------------------------------------

st.write("### 3️⃣ Value Counts")

value_counts = (
    df[selected_cat]
    .value_counts(dropna=False)
    .reset_index()
)

value_counts.columns = [
    selected_cat,
    "Count"
]

st.dataframe(
    value_counts,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# CHARTS
# --------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

# Bar Chart
with chart_col1:

    st.write("### 4️⃣ Bar Chart")

    fig_bar = px.bar(
        value_counts,
        x=selected_cat,
        y="Count",
        text="Count",
        title=f"{selected_cat} - Bar Chart"
    )

    fig_bar.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )

# Pie Chart
with chart_col2:

    st.write("### 5️⃣ Pie Chart")

    fig_pie = px.pie(
        value_counts,
        names=selected_cat,
        values="Count",
        hole=0.4,
        title=f"{selected_cat} - Pie Chart"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

# --------------------------------------------------
# COUNT PLOT
# --------------------------------------------------

st.write("### 6️⃣ Count Plot")

fig_count = px.histogram(
    df,
    x=selected_cat,
    color=selected_cat,
    title=f"{selected_cat} - Count Plot",
    text_auto=True
)

st.plotly_chart(
    fig_count,
    use_container_width=True
)


# --------------------------------------------------
# STEP 6 - NUMERICAL DATA ANALYSIS
# --------------------------------------------------

st.divider()

st.header("🔢 Step 6: Numerical Data Analysis")

# Get numerical columns
numerical_analysis_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

# Remove identifiers and target from general numerical analysis
analysis_columns = [
    col for col in numerical_analysis_columns
    if col not in ["customer_id", "churn"]
]

# Select numerical column
selected_num = st.selectbox(
    "Select a numerical column:",
    analysis_columns
)

st.subheader(f"📊 Analysis of: {selected_num}")

# --------------------------------------------------
# 1. DESCRIBE FUNCTION
# --------------------------------------------------

st.write("### 1️⃣ Descriptive Statistics")

describe_data = df[selected_num].describe()

describe_df = pd.DataFrame({
    "Statistic": describe_data.index,
    "Value": describe_data.values
})

st.dataframe(
    describe_df,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# 2. HISTOGRAM
# --------------------------------------------------

st.write("### 2️⃣ Histogram")

fig_hist = px.histogram(
    df,
    x=selected_num,
    nbins=30,
    title=f"{selected_num} Distribution",
    marginal="box"
)

st.plotly_chart(
    fig_hist,
    use_container_width=True
)

# --------------------------------------------------
# 3. SKEWNESS
# --------------------------------------------------

st.write("### 3️⃣ Data Skewness")

skewness_value = df[selected_num].skew()

if skewness_value > 1:

    skew_interpretation = "Highly Right Skewed"

elif skewness_value > 0.5:

    skew_interpretation = "Moderately Right Skewed"

elif skewness_value < -1:

    skew_interpretation = "Highly Left Skewed"

elif skewness_value < -0.5:

    skew_interpretation = "Moderately Left Skewed"

else:

    skew_interpretation = "Approximately Symmetric"

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Skewness",
        f"{skewness_value:.3f}"
    )

with col2:

    st.info(
        f"Interpretation: **{skew_interpretation}**"
    )

# --------------------------------------------------
# STEP 7 - BOX PLOT & OUTLIER ANALYSIS
# --------------------------------------------------

st.divider()

st.header("📦 Step 7: Box Plot & Outlier Analysis")

# Numerical columns for outlier analysis
outlier_columns = [
    col for col in df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    if col not in ["customer_id", "churn"]
]

# Select column
selected_outlier_col = st.selectbox(
    "Select a numerical column for outlier analysis:",
    outlier_columns,
    key="outlier_column"
)

# --------------------------------------------------
# BOX PLOT
# --------------------------------------------------

st.subheader(f"📦 Box Plot: {selected_outlier_col}")

fig_box = px.box(
    df,
    y=selected_outlier_col,
    points="outliers",
    title=f"{selected_outlier_col} - Box Plot"
)

st.plotly_chart(
    fig_box,
    use_container_width=True
)

# --------------------------------------------------
# IQR CALCULATION
# --------------------------------------------------

Q1 = df[selected_outlier_col].quantile(0.25)

Q3 = df[selected_outlier_col].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - (1.5 * IQR)

upper_limit = Q3 + (1.5 * IQR)

# --------------------------------------------------
# FIND OUTLIERS
# --------------------------------------------------

outliers = df[
    (df[selected_outlier_col] < lower_limit) |
    (df[selected_outlier_col] > upper_limit)
]

outlier_count = len(outliers)

total_values = df[selected_outlier_col].notna().sum()

outlier_percentage = (
    outlier_count / total_values * 100
)

# --------------------------------------------------
# DISPLAY IQR INFORMATION
# --------------------------------------------------

st.subheader("📐 IQR Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Q1",
        f"{Q1:.2f}"
    )

with col2:
    st.metric(
        "Q3",
        f"{Q3:.2f}"
    )

with col3:
    st.metric(
        "IQR",
        f"{IQR:.2f}"
    )

with col4:
    st.metric(
        "Outliers",
        f"{outlier_count}"
    )

# --------------------------------------------------
# OUTLIER LIMITS
# --------------------------------------------------

st.write("### 🚧 Outlier Boundaries")

boundary_col1, boundary_col2 = st.columns(2)

with boundary_col1:

    st.info(
        f"Lower Limit: **{lower_limit:.2f}**"
    )

with boundary_col2:

    st.warning(
        f"Upper Limit: **{upper_limit:.2f}**"
    )

# --------------------------------------------------
# OUTLIER PERCENTAGE
# --------------------------------------------------

st.metric(
    "Outlier Percentage",
    f"{outlier_percentage:.2f}%"
)

# --------------------------------------------------
# OUTLIER DATA
# --------------------------------------------------

if outlier_count > 0:

    st.subheader("🚨 Outlier Records")

    st.dataframe(
        outliers[[selected_outlier_col]],
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "✅ No outliers detected using the 1.5 × IQR rule."
    )

# --------------------------------------------------
# STEP 8 - TRANSFORMATION METHODS
# --------------------------------------------------

st.divider()

st.header("🔄 Step 8: Transformation Methods")

st.write(
    "Transformations can help reduce skewness and make "
    "numerical variables more suitable for machine learning."
)

# Numerical columns
transformation_columns = [
    col for col in df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    if col not in ["customer_id", "churn"]
]

# Select column
selected_transform_col = st.selectbox(
    "Select a numerical column:",
    transformation_columns,
    key="transformation_column"
)

# Select transformation
transformation_method = st.selectbox(
    "Select transformation method:",
    [
        "Log Transformation",
        "Square Root Transformation",
        "Power Transformation"
    ]
)

# Original data
original_data = df[selected_transform_col].dropna()

# Original skewness
original_skewness = original_data.skew()

# --------------------------------------------------
# APPLY TRANSFORMATION
# --------------------------------------------------

try:

    if transformation_method == "Log Transformation":

        # Log1p works with zero values
        transformed_data = np.log1p(original_data)

    elif transformation_method == "Square Root Transformation":

        # Square root requires non-negative values
        if (original_data < 0).any():

            st.error(
                "Square root transformation cannot be applied "
                "because this column contains negative values."
            )

            transformed_data = None

        else:

            transformed_data = np.sqrt(original_data)

    else:

        # Yeo-Johnson Power Transformation
        from sklearn.preprocessing import PowerTransformer

        power_transformer = PowerTransformer(
            method="yeo-johnson"
        )

        transformed_data = power_transformer.fit_transform(
            original_data.to_numpy().reshape(-1, 1)
        ).flatten()

    # --------------------------------------------------
    # DISPLAY SKEWNESS
    # --------------------------------------------------

    if transformed_data is not None:

        transformed_skewness = pd.Series(
            transformed_data
        ).skew()

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Original Skewness",
                f"{original_skewness:.3f}"
            )

        with col2:

            st.metric(
                "Transformed Skewness",
                f"{transformed_skewness:.3f}"
            )

        # --------------------------------------------------
        # DISTRIBUTION COMPARISON
        # --------------------------------------------------

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            st.subheader("📊 Before Transformation")

            fig_before = px.histogram(
                x=original_data,
                nbins=30,
                title="Original Distribution"
            )

            st.plotly_chart(
                fig_before,
                use_container_width=True
            )

        with chart_col2:

            st.subheader("📊 After Transformation")

            fig_after = px.histogram(
                x=transformed_data,
                nbins=30,
                title=f"After {transformation_method}"
            )

            st.plotly_chart(
                fig_after,
                use_container_width=True
            )

        # --------------------------------------------------
        # INTERPRETATION
        # --------------------------------------------------

        if abs(transformed_skewness) < abs(original_skewness):

            st.success(
                "✅ This transformation reduced the absolute skewness."
            )

        else:

            st.warning(
                "⚠️ This transformation did not reduce the "
                "absolute skewness for this column."
            )

except Exception as e:

    st.error(
        f"Transformation could not be applied: {e}"
    )

# --------------------------------------------------
# STEP 9 - ENCODING METHODS
# --------------------------------------------------

st.divider()

st.header("🔢 Step 9: Encoding Methods")

st.write(
    "Encoding converts categorical values into numerical "
    "values that machine-learning algorithms can understand."
)

encoding_method = st.selectbox(
    "Select Encoding Method:",
    [
        "Map",
        "np.where",
        "LabelEncoder",
        "OneHotEncoder"
    ]
)

# --------------------------------------------------
# METHOD 1 - MAP
# --------------------------------------------------

if encoding_method == "Map":

    st.subheader("1️⃣ Map Encoding")

    st.write(
        "Map is useful when you have a binary categorical "
        "column and want to manually assign values."
    )

    temp_df = df[["gender"]].copy()

    mapping = {
        "Male": 1,
        "Female": 0
    }

    temp_df["gender_encoded"] = temp_df["gender"].map(
        mapping
    )

    st.write("Mapping used:")

    st.code(
        """
{
    'Male': 1,
    'Female': 0
}
"""
    )

    st.dataframe(
        temp_df.head(20),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# METHOD 2 - NP.WHERE
# --------------------------------------------------

elif encoding_method == "np.where":

    st.subheader("2️⃣ np.where Encoding")

    st.write(
        "np.where can be used to create binary values "
        "based on a condition."
    )

    temp_df = df[["gender"]].copy()

    temp_df["gender_encoded"] = np.where(
        temp_df["gender"] == "Male",
        1,
        0
    )

    st.code(
        """
np.where(
    gender == 'Male',
    1,
    0
)
"""
    )

    st.dataframe(
        temp_df.head(20),
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# METHOD 3 - LABEL ENCODER
# --------------------------------------------------

elif encoding_method == "LabelEncoder":

    st.subheader("3️⃣ LabelEncoder")

    from sklearn.preprocessing import LabelEncoder

    temp_df = df[["multi_screen"]].copy()

    label_encoder = LabelEncoder()

    temp_df["multi_screen_encoded"] = (
        label_encoder.fit_transform(
            temp_df["multi_screen"].astype(str)
        )
    )

    # Show mapping
    mapping_df = pd.DataFrame({
        "Original Value": label_encoder.classes_,
        "Encoded Value": range(
            len(label_encoder.classes_)
        )
    })

    st.write("Encoding Mapping:")

    st.dataframe(
        mapping_df,
        use_container_width=True,
        hide_index=True
    )

    st.write("Encoded Data:")

    st.dataframe(
        temp_df.head(20),
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "⚠️ LabelEncoder creates numerical labels. "
        "For nominal categories, OneHotEncoder is often preferable."
    )

# --------------------------------------------------
# METHOD 4 - ONE HOT ENCODER
# --------------------------------------------------

else:

    st.subheader("4️⃣ OneHotEncoder")

    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    encoded_array = encoder.fit_transform(
        df[["multi_screen"]]
    )

    encoded_columns = encoder.get_feature_names_out(
        ["multi_screen"]
    )

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoded_columns,
        index=df.index
    )

    st.write("Original Data:")

    st.dataframe(
        df[["multi_screen"]].head(10),
        use_container_width=True,
        hide_index=True
    )

    st.write("One-Hot Encoded Data:")

    st.dataframe(
        encoded_df.head(10),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
# STEP 10 - SCALING METHODS
# --------------------------------------------------

st.divider()

st.header("⚖️ Step 10: Scaling Methods")

st.write(
    "Scaling puts numerical features onto comparable scales. "
    "This is especially important for algorithms such as PCA, "
    "KNN, K-Means and SVM."
)

# Numerical columns
scaling_columns = [
    col for col in df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    if col not in ["customer_id", "churn"]
]

# Select numerical column
selected_scaling_col = st.selectbox(
    "Select a numerical column:",
    scaling_columns,
    key="scaling_column"
)

# Select scaling method
scaling_method = st.selectbox(
    "Select Scaling Method:",
    [
        "Z-Score Standardization",
        "Min-Max Scaling"
    ]
)

# Original data
original_values = df[
    [selected_scaling_col]
].dropna()

# --------------------------------------------------
# APPLY SCALING
# --------------------------------------------------

if scaling_method == "Z-Score Standardization":

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(
        original_values
    )

    method_name = "Z-Score Standardization"

else:

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()

    scaled_values = scaler.fit_transform(
        original_values
    )

    method_name = "Min-Max Scaling"

# Create comparison dataframe

scaled_df = pd.DataFrame(
    scaled_values,
    columns=["Scaled Value"],
    index=original_values.index
)

comparison_df = pd.DataFrame({
    "Original Value": original_values[
        selected_scaling_col
    ],
    "Scaled Value": scaled_df["Scaled Value"]
})

# --------------------------------------------------
# DISPLAY FORMULA
# --------------------------------------------------

st.subheader("📐 Scaling Method")

if scaling_method == "Z-Score Standardization":

    st.latex(
        r"Z = \frac{X - \mu}{\sigma}"
    )

    st.info(
        "Z-score scaling generally produces data with "
        "mean ≈ 0 and standard deviation ≈ 1."
    )

else:

    st.latex(
        r"X_{scaled} = \frac{X-X_{min}}{X_{max}-X_{min}}"
    )

    st.info(
        "Min-Max scaling generally converts values "
        "to a range between 0 and 1."
    )

# --------------------------------------------------
# BEFORE / AFTER METRICS
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Original Data")

    st.metric(
        "Minimum",
        f"{original_values[selected_scaling_col].min():.2f}"
    )

    st.metric(
        "Maximum",
        f"{original_values[selected_scaling_col].max():.2f}"
    )

with col2:

    st.subheader("Scaled Data")

    st.metric(
        "Minimum",
        f"{scaled_df['Scaled Value'].min():.3f}"
    )

    st.metric(
        "Maximum",
        f"{scaled_df['Scaled Value'].max():.3f}"
    )

# --------------------------------------------------
# DATA COMPARISON
# --------------------------------------------------

st.subheader("📋 Original vs Scaled Values")

st.dataframe(
    comparison_df.head(20),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DISTRIBUTION COMPARISON
# --------------------------------------------------

chart_col1, chart_col2 = st.columns(2)

with chart_col1:

    st.subheader("📊 Original Distribution")

    fig_original = px.histogram(
        original_values,
        x=selected_scaling_col,
        nbins=30,
        title="Before Scaling"
    )

    st.plotly_chart(
        fig_original,
        use_container_width=True
    )

with chart_col2:

    st.subheader(f"📊 {method_name}")

    fig_scaled = px.histogram(
        scaled_df,
        x="Scaled Value",
        nbins=30,
        title="After Scaling"
    )

    st.plotly_chart(
        fig_scaled,
        use_container_width=True
    )

# --------------------------------------------------
# STEP 11 - PCA
# --------------------------------------------------

st.divider()

st.header("🧩 Step 11: Principal Component Analysis (PCA)")

st.write(
    "PCA reduces the number of numerical features while "
    "preserving as much variance as possible."
)

# --------------------------------------------------
# SELECT NUMERICAL FEATURES
# --------------------------------------------------

pca_columns = [
    col for col in df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    if col not in ["customer_id", "churn"]
]

st.subheader("🔢 PCA Features")

selected_pca_columns = st.multiselect(
    "Select numerical features for PCA:",
    pca_columns,
    default=pca_columns
)

# --------------------------------------------------
# NUMBER OF COMPONENTS
# --------------------------------------------------

if len(selected_pca_columns) >= 2:

    max_components = len(selected_pca_columns)

    n_components = st.slider(
        "Number of Principal Components:",
        min_value=2,
        max_value=max_components,
        value=min(2, max_components)
    )

    # --------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------

    pca_data = df[selected_pca_columns].copy()

    # Fill missing values using median
    pca_data = pca_data.fillna(
        pca_data.median()
    )

    # --------------------------------------------------
    # SCALE DATA
    # --------------------------------------------------

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scaled_pca_data = scaler.fit_transform(
        pca_data
    )

    # --------------------------------------------------
    # APPLY PCA
    # --------------------------------------------------

    from sklearn.decomposition import PCA

    pca = PCA(
        n_components=n_components
    )

    principal_components = pca.fit_transform(
        scaled_pca_data
    )

    # --------------------------------------------------
    # EXPLAINED VARIANCE
    # --------------------------------------------------

    explained_variance = (
        pca.explained_variance_ratio_ * 100
    )

    cumulative_variance = (
        explained_variance.cumsum()
    )

    variance_df = pd.DataFrame({
        "Component": [
            f"PC{i + 1}"
            for i in range(n_components)
        ],
        "Explained Variance (%)": explained_variance,
        "Cumulative Variance (%)": cumulative_variance
    })

    # --------------------------------------------------
    # PCA SUMMARY
    # --------------------------------------------------

    st.subheader("📊 PCA Explained Variance")

    st.dataframe(
        variance_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # VARIANCE CHART
    # --------------------------------------------------

    fig_variance = px.bar(
        variance_df,
        x="Component",
        y="Explained Variance (%)",
        text="Explained Variance (%)",
        title="Variance Explained by Each Principal Component"
    )

    fig_variance.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )

    st.plotly_chart(
        fig_variance,
        use_container_width=True
    )

    # --------------------------------------------------
    # CUMULATIVE VARIANCE
    # --------------------------------------------------

    fig_cumulative = px.line(
        variance_df,
        x="Component",
        y="Cumulative Variance (%)",
        markers=True,
        title="Cumulative Explained Variance"
    )

    fig_cumulative.update_yaxes(
        range=[0, 100]
    )

    st.plotly_chart(
        fig_cumulative,
        use_container_width=True
    )

    # --------------------------------------------------
    # PCA COMPONENT DATA
    # --------------------------------------------------

    component_columns = [
        f"PC{i + 1}"
        for i in range(n_components)
    ]

    pca_result = pd.DataFrame(
        principal_components,
        columns=component_columns
    )

    st.subheader("🧩 Principal Component Data")

    st.dataframe(
        pca_result.head(20),
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # 2D PCA VISUALIZATION
    # --------------------------------------------------

    if n_components >= 2:

        st.subheader("📍 PCA 2D Visualization")

        fig_pca = px.scatter(
            pca_result,
            x="PC1",
            y="PC2",
            title="Customers in PCA Space"
        )

        st.plotly_chart(
            fig_pca,
            use_container_width=True
        )

    # --------------------------------------------------
    # PCA LOADINGS
    # --------------------------------------------------

    st.subheader("🔍 PCA Feature Loadings")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=component_columns,
        index=selected_pca_columns
    )

    st.dataframe(
        loadings,
        use_container_width=True
    )

else:

    st.warning(
        "⚠️ Please select at least 2 numerical features for PCA."
    )

# --------------------------------------------------
# STEP 12 - MODEL COMPARISON
# --------------------------------------------------

st.divider()

st.header("🤖 Machine Learning Model Comparison")

st.write(
    "Three classification models were trained and evaluated "
    "using the Telecom Churn dataset."
)

# Get model results
model_results = get_model_results()

# --------------------------------------------------
# DISPLAY RESULTS TABLE
# --------------------------------------------------

st.subheader("📊 Model Performance")

display_results = model_results.copy()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

display_results[metric_columns] = (
    display_results[metric_columns] * 100
).round(2)

st.dataframe(
    display_results,
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# FIND BEST MODEL
# --------------------------------------------------

best_model_row = model_results.loc[
    model_results["ROC AUC"].idxmax()
]

best_model_name = best_model_row["Model"]

best_roc_auc = best_model_row["ROC AUC"] * 100

st.success(
    f"🏆 Best Model based on ROC-AUC: "
    f"**{best_model_name}** ({best_roc_auc:.2f}%)"
)

# --------------------------------------------------
# MODEL METRICS CHART
# --------------------------------------------------

chart_data = model_results.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC AUC"
    ],
    var_name="Metric",
    value_name="Score"
)

chart_data["Score"] = chart_data["Score"] * 100

fig_model = px.bar(
    chart_data,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text_auto=".1f",
    title="Model Performance Comparison"
)

fig_model.update_yaxes(
    range=[0, 100],
    title="Score (%)"
)

st.plotly_chart(
    fig_model,
    use_container_width=True
)

# --------------------------------------------------
# STEP 14 - CUSTOMER CHURN PREDICTION
# --------------------------------------------------

st.divider()

st.header("🎯 Customer Churn Prediction")

st.write(
    "Enter customer information below to predict "
    "the probability of churn."
)

# --------------------------------------------------
# IMPORT PREDICTION FUNCTION
# --------------------------------------------------

from model import predict_churn

# --------------------------------------------------
# CUSTOMER INPUT FORM
# --------------------------------------------------

st.subheader("👤 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=100,
        value=35
    )

    no_of_days_subscribed = st.number_input(
        "Days Subscribed",
        min_value=0,
        value=365
    )

    multi_screen = st.selectbox(
        "Multi Screen",
        ["no", "yes"]
    )

with col2:

    mail_subscribed = st.selectbox(
        "Mail Subscribed",
        ["no", "yes"]
    )

    weekly_mins_watched = st.number_input(
        "Weekly Minutes Watched",
        min_value=0.0,
        value=400.0
    )

    minimum_daily_mins = st.number_input(
        "Minimum Daily Minutes",
        min_value=0.0,
        value=20.0
    )

    maximum_daily_mins = st.number_input(
        "Maximum Daily Minutes",
        min_value=0.0,
        value=60.0
    )

with col3:

    weekly_max_night_mins = st.number_input(
        "Weekly Max Night Minutes",
        min_value=0,
        value=200
    )

    videos_watched = st.number_input(
        "Videos Watched",
        min_value=0,
        value=10
    )

    maximum_days_inactive = st.number_input(
        "Maximum Days Inactive",
        min_value=0.0,
        value=3.0
    )

    customer_support_calls = st.number_input(
        "Customer Support Calls",
        min_value=0,
        value=1
    )


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🔮 Predict Churn",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# MAKE PREDICTION
# --------------------------------------------------

if predict_button:

    customer_input = pd.DataFrame({
        "year": [2015],
        "gender": [gender],
        "age": [age],
        "no_of_days_subscribed": [
            no_of_days_subscribed
        ],
        "multi_screen": [multi_screen],
        "mail_subscribed": [mail_subscribed],
        "weekly_mins_watched": [
            weekly_mins_watched
        ],
        "minimum_daily_mins": [
            minimum_daily_mins
        ],
        "maximum_daily_mins": [
            maximum_daily_mins
        ],
        "weekly_max_night_mins": [
            weekly_max_night_mins
        ],
        "videos_watched": [
            videos_watched
        ],
        "maximum_days_inactive": [
            maximum_days_inactive
        ],
        "customer_support_calls": [
            customer_support_calls
        ]
    })

    prediction, probability = predict_churn(
        customer_input
    )

    probability_percentage = probability * 100

    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    st.divider()

    st.subheader("🎯 Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        if prediction == 1:

            st.error(
                "⚠️ CUSTOMER LIKELY TO CHURN"
            )

        else:

            st.success(
                "✅ CUSTOMER LIKELY TO STAY"
            )

    with result_col2:

        st.metric(
            "Churn Probability",
            f"{probability_percentage:.2f}%"
        )

    # --------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------

    st.progress(
        float(probability)
    )

    if probability >= 0.70:

        st.warning(
            "🚨 High churn risk. Consider taking "
            "customer retention action."
        )

    elif probability >= 0.40:

        st.info(
            "⚠️ Medium churn risk. Customer may "
            "require additional attention."
        )

    else:

        st.success(
            "🟢 Low churn risk. Customer appears "
            "relatively stable."
        )