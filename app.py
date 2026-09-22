import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    StratifiedKFold,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from sklearn.dummy import (
    DummyRegressor,
    DummyClassifier,
)

from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

from sklearn.tree import (
    DecisionTreeRegressor,
    DecisionTreeClassifier,
)

from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)

from sklearn.neighbors import (
    KNeighborsRegressor,
    KNeighborsClassifier,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Auto ML",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center;">
        Auto ML
    </h1>

    <p style="
        text-align:center;
        font-size:18px;
        margin-bottom:40px;
    ">
        Upload a CSV dataset, define a prediction problem,
        and compare machine-learning models.
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

# This is VERY important.
# It prevents our results from disappearing every time
# Streamlit reruns the page.

if "results" not in st.session_state:
    st.session_state.results = None

if "model_details" not in st.session_state:
    st.session_state.model_details = {}

if "problem_type" not in st.session_state:
    st.session_state.problem_type = None

if "target_name" not in st.session_state:
    st.session_state.target_name = None

if "baseline" not in st.session_state:
    st.session_state.baseline = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def looks_like_identifier(series, column_name):
    """
    Try to detect columns that look like identifiers.

    Examples:
        ID
        customer_id
        player_id
        transaction_id

    A nearly unique column may also be an identifier.
    """

    name = column_name.lower().strip()

    id_names = [
        "id",
        "index",
        "identifier",
        "record_id",
    ]

    if name in id_names:
        return True

    if name.endswith("_id"):
        return True

    if name.endswith("id") and len(name) > 2:
        return True

    non_null = series.dropna()

    if len(non_null) == 0:
        return False

    uniqueness_ratio = (
        non_null.nunique() / len(non_null)
    )

    # Don't automatically flag every continuous numeric
    # variable just because values happen to be unique.
    if (
        uniqueness_ratio > 0.98
        and not pd.api.types.is_float_dtype(series)
    ):
        return True

    return False


def detect_problem_type(series):
    """
    Determine whether the selected target appears to represent
    a regression or classification problem.
    """

    clean_series = series.dropna()

    if len(clean_series) == 0:
        return None

    # Text / boolean targets are classification.
    if (
        pd.api.types.is_object_dtype(clean_series)
        or pd.api.types.is_bool_dtype(clean_series)
        or isinstance(clean_series.dtype, pd.CategoricalDtype)
    ):
        return "Classification"

    unique_count = clean_series.nunique()

    # Numeric variables with only a few unique values
    # are probably class labels.
    if unique_count <= 10:
        return "Classification"

    return "Regression"


def build_preprocessor(X):
    """
    Automatically preprocess numeric and categorical columns.
    """

    numeric_columns = (
        X.select_dtypes(include=np.number)
        .columns
        .tolist()
    )

    categorical_columns = (
        X.select_dtypes(exclude=np.number)
        .columns
        .tolist()
    )

    transformers = []

    # Numeric processing
    if numeric_columns:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_columns,
            )
        )

    # Categorical processing
    if categorical_columns:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent"
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


# ============================================================
# 1. UPLOAD DATASET
# ============================================================

st.header("1. Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is None:
    st.info(
        "Upload a CSV file to begin."
    )

    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:

    df = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        "The file could not be read as a CSV."
    )

    with st.expander(
        "Technical details"
    ):
        st.code(str(error))

    st.stop()


if df.empty:

    st.error(
        "The uploaded dataset is empty."
    )

    st.stop()


if df.shape[1] < 2:

    st.error(
        "The dataset needs at least two columns."
    )

    st.stop()


st.success(
    f"Dataset uploaded successfully — "
    f"{df.shape[0]:,} rows × "
    f"{df.shape[1]:,} columns."
)


# ============================================================
# DATA PREVIEW
# ============================================================

with st.expander(
    "Preview Dataset"
):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

    col2.metric(
        "Columns",
        f"{df.shape[1]:,}"
    )

    col3.metric(
        "Missing Values",
        f"{df.isna().sum().sum():,}"
    )


# ============================================================
# CHECK COLUMN QUALITY
# ============================================================

constant_columns = []

identifier_columns = []


for column in df.columns:

    if df[column].nunique(
        dropna=True
    ) <= 1:

        constant_columns.append(
            column
        )

    elif looks_like_identifier(
        df[column],
        column
    ):

        identifier_columns.append(
            column
        )


# ============================================================
# 2. TARGET SELECTION
# ============================================================

st.divider()

st.header(
    "2. Select Prediction Target"
)

st.write(
    """
    Choose the variable you want the models to predict.
    """
)


target_options = [
    column
    for column in df.columns
    if column not in constant_columns
]


target = st.selectbox(
    "Target variable",
    target_options
)


# ============================================================
# TARGET WARNINGS
# ============================================================

if target in identifier_columns:

    st.warning(
        f"'{target}' appears to be an identifier. "
        "Identifier columns usually do not represent "
        "meaningful prediction targets."
    )


problem_type = detect_problem_type(
    df[target]
)


if problem_type is None:

    st.error(
        "The selected target does not contain usable data."
    )

    st.stop()


st.info(
    f"Detected machine-learning problem: "
    f"**{problem_type}**"
)


# ============================================================
# 3. FEATURE SELECTION
# ============================================================

st.divider()

st.header(
    "3. Select Prediction Features"
)

st.write(
    """
    Choose the variables the models are allowed to use
    when predicting the target.
    """
)


feature_options = [
    column
    for column in df.columns
    if (
        column != target
        and column not in constant_columns
    )
]


# Automatically exclude identifier-like variables
# from the default selection.

default_features = [
    column
    for column in feature_options
    if column not in identifier_columns
]


selected_features = st.multiselect(
    "Prediction features",
    feature_options,
    default=default_features
)


excluded_identifiers = [
    column
    for column in identifier_columns
    if column != target
]


if excluded_identifiers:

    st.caption(
        "Excluded by default because they appear "
        "to be identifiers: "
        + ", ".join(excluded_identifiers)
    )


if len(selected_features) == 0:

    st.warning(
        "Select at least one prediction feature."
    )

    st.stop()


# ============================================================
# 4. MODEL SELECTION
# ============================================================

st.divider()

st.header(
    "4. Select Three Models"
)


if problem_type == "Regression":

    available_models = [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "K-Nearest Neighbors",
        "Gradient Boosting",
    ]

else:

    available_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "K-Nearest Neighbors",
        "Gradient Boosting",
    ]


# Default to three models.

default_models = [
    available_models[0],
    available_models[2],
    available_models[4],
]


selected_models = st.multiselect(
    "Choose exactly three models to compare",
    available_models,
    default=default_models,
    max_selections=3
)


if len(selected_models) < 3:

    st.info(
        f"Select {3 - len(selected_models)} "
        f"more model(s)."
    )


# ============================================================
# RUN MODELS
# ============================================================

run_models = st.button(
    "Run Three Models",
    type="primary",
    disabled=(
        len(selected_models) != 3
    )
)


if run_models:

    # Clear previous results.
    st.session_state.results = None
    st.session_state.model_details = {}


    # ========================================================
    # PREPARE DATA
    # ========================================================

    working_df = df[
        selected_features + [target]
    ].copy()


    # Target cannot be missing.
    working_df = working_df.dropna(
        subset=[target]
    )


    if len(working_df) < 20:

        st.error(
            "There are too few usable rows to "
            "evaluate these models reliably."
        )

        st.stop()


    X = working_df[
        selected_features
    ]

    y = working_df[
        target
    ]


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    try:

        if problem_type == "Classification":

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                    stratify=y,
                )
            )

        else:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.20,
                    random_state=42,
                )
            )

    except ValueError:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.20,
                random_state=42,
            )
        )


    # ========================================================
    # PREPROCESSING
    # ========================================================

    preprocessor = build_preprocessor(
        X
    )


    # ========================================================
    # MODEL DEFINITIONS
    # ========================================================

    if problem_type == "Regression":

        model_library = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(
                    random_state=42
                ),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                ),

            "K-Nearest Neighbors":
                KNeighborsRegressor(),

            "Gradient Boosting":
                GradientBoostingRegressor(
                    random_state=42
                ),
        }

    else:

        model_library = {

            "Logistic Regression":
                LogisticRegression(
                    max_iter=2000
                ),

            "Decision Tree":
                DecisionTreeClassifier(
                    random_state=42
                ),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                ),

            "K-Nearest Neighbors":
                KNeighborsClassifier(),

            "Gradient Boosting":
                GradientBoostingClassifier(
                    random_state=42
                ),
        }


    # ========================================================
    # CROSS VALIDATION
    # ========================================================

    if problem_type == "Regression":

        cv = KFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

    else:

        minimum_class_size = (
            y.value_counts().min()
        )

        if minimum_class_size >= 5:

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

        else:

            cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )


    # ========================================================
    # BASELINE MODEL
    # ========================================================

    if problem_type == "Regression":

        baseline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    DummyRegressor(
                        strategy="mean"
                    )
                ),
            ]
        )

    else:

        baseline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor
                ),
                (
                    "model",
                    DummyClassifier(
                        strategy="most_frequent"
                    )
                ),
            ]
        )


    baseline.fit(
        X_train,
        y_train
    )


    baseline_predictions = (
        baseline.predict(
            X_test
        )
    )


    if problem_type == "Regression":

        baseline_score = (
            mean_absolute_error(
                y_test,
                baseline_predictions
            )
        )

    else:

        baseline_score = (
            accuracy_score(
                y_test,
                baseline_predictions
            )
        )


    # ========================================================
    # TRAIN THREE MODELS
    # ========================================================

    results = []

    details = {}

    progress = st.progress(0)


    for i, model_name in enumerate(
        selected_models
    ):

        try:

            model = model_library[
                model_name
            ]


            pipeline = Pipeline(
                steps=[
                    (
                        "preprocessor",
                        preprocessor
                    ),
                    (
                        "model",
                        model
                    ),
                ]
            )


            # Train model
            pipeline.fit(
                X_train,
                y_train
            )


            # Test predictions
            predictions = (
                pipeline.predict(
                    X_test
                )
            )


            # =================================================
            # REGRESSION RESULTS
            # =================================================

            if problem_type == "Regression":

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                )

                r2 = r2_score(
                    y_test,
                    predictions
                )


                cv_scores = (
                    cross_val_score(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring="r2"
                    )
                )


                results.append(
                    {
                        "Model": model_name,
                        "MAE": mae,
                        "RMSE": rmse,
                        "R²": r2,
                        "CV Score":
                            cv_scores.mean(),
                        "CV Std":
                            cv_scores.std(),
                    }
                )


                prediction_table = (
                    pd.DataFrame(
                        {
                            "Actual":
                                y_test.values,
                            "Predicted":
                                predictions,
                        },
                        index=y_test.index
                    )
                )


                prediction_table[
                    "Error"
                ] = (
                    prediction_table[
                        "Actual"
                    ]
                    - prediction_table[
                        "Predicted"
                    ]
                )


                prediction_table[
                    "Absolute Error"
                ] = (
                    prediction_table[
                        "Error"
                    ].abs()
                )


                details[
                    model_name
                ] = {

                    "pipeline":
                        pipeline,

                    "predictions":
                        predictions,

                    "table":
                        prediction_table,

                    "mae":
                        mae,

                    "rmse":
                        rmse,

                    "r2":
                        r2,

                    "cv_mean":
                        cv_scores.mean(),

                    "cv_std":
                        cv_scores.std(),
                }


            # =================================================
            # CLASSIFICATION RESULTS
            # =================================================

            else:

                accuracy = (
                    accuracy_score(
                        y_test,
                        predictions
                    )
                )

                precision = (
                    precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                )

                recall = (
                    recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                )

                f1 = (
                    f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                )


                cv_scores = (
                    cross_val_score(
                        pipeline,
                        X,
                        y,
                        cv=cv,
                        scoring="accuracy"
                    )
                )


                results.append(
                    {
                        "Model":
                            model_name,

                        "Accuracy":
                            accuracy,

                        "Precision":
                            precision,

                        "Recall":
                            recall,

                        "F1":
                            f1,

                        "CV Score":
                            cv_scores.mean(),

                        "CV Std":
                            cv_scores.std(),
                    }
                )


                details[
                    model_name
                ] = {

                    "pipeline":
                        pipeline,

                    "predictions":
                        predictions,

                    "accuracy":
                        accuracy,

                    "precision":
                        precision,

                    "recall":
                        recall,

                    "f1":
                        f1,

                    "cv_mean":
                        cv_scores.mean(),

                    "cv_std":
                        cv_scores.std(),

                    "y_test":
                        y_test,
                }


        except Exception as error:

            st.error(
                f"{model_name} could not be trained."
            )

            with st.expander(
                f"Technical details — {model_name}"
            ):

                st.code(
                    str(error)
                )


        progress.progress(
            (i + 1) / 3
        )


    # ========================================================
    # SAVE EVERYTHING
    # ========================================================

    st.session_state.results = (
        pd.DataFrame(results)
    )

    st.session_state.model_details = (
        details
    )

    st.session_state.problem_type = (
        problem_type
    )

    st.session_state.target_name = (
        target
    )

    st.session_state.baseline = (
        baseline_score
    )

    st.session_state.test_size = (
        len(y_test)
    )

    st.session_state.total_size = (
        len(y)
    )


# ============================================================
# RESULTS SECTION
# ============================================================

if st.session_state.results is not None:

    results_df = (
        st.session_state.results
    )

    details = (
        st.session_state.model_details
    )

    saved_problem_type = (
        st.session_state.problem_type
    )

    saved_target = (
        st.session_state.target_name
    )


    st.divider()

    st.header(
        "Model Results"
    )


    st.write(
        f"""
        **Prediction target:** {saved_target}

        **Usable observations:** {st.session_state.total_size:,}

        **Held-out test observations:** {st.session_state.test_size:,}

        The three models were trained using the same data
        and evaluated using the same test set so their
        performance can be compared directly.
        """
    )


    # ========================================================
    # METRIC EXPLANATION
    # ========================================================

    with st.expander(
        "What do these metrics mean?"
    ):

        if saved_problem_type == "Regression":

            st.markdown(
                """
                ### MAE — Mean Absolute Error

                The average size of the model's prediction
                errors.

                **Lower is better.**

                If MAE = 5, predictions differ from the
                actual target by about 5 units on average.

                ---

                ### RMSE — Root Mean Squared Error

                Another measure of prediction error.

                RMSE penalizes large mistakes more strongly
                than MAE.

                **Lower is better.**

                ---

                ### R² — Coefficient of Determination

                Measures how closely predictions correspond
                to variation in the actual target.

                **Higher is generally better.**

                - 1.0 = perfect prediction
                - 0.0 = similar to predicting the mean
                - Below 0 = worse than that simple baseline

                ---

                ### CV Score — Cross-Validation Score

                The model is tested repeatedly on different
                subsets of the dataset.

                A strong CV score that is similar to the
                test-set R² provides more evidence that the
                model's performance is consistent.
                """
            )

        else:

            st.markdown(
                """
                ### Accuracy

                Percentage of observations classified
                correctly.

                **Higher is better.**

                ---

                ### Precision

                Measures how often predicted classes are
                correct.

                **Higher is better.**

                ---

                ### Recall

                Measures how successfully actual cases are
                identified.

                **Higher is better.**

                ---

                ### F1 Score

                Combines precision and recall into one
                measure.

                **Higher is better.**

                ---

                ### CV Score

                Average performance across multiple
                train/test partitions of the dataset.

                It helps determine whether model performance
                is consistent rather than dependent on one
                favorable test split.
                """
            )


    # ========================================================
    # THREE MODELS SIDE BY SIDE
    # ========================================================

    st.subheader(
        "Side-by-Side Comparison"
    )


    model_columns = st.columns(3)


    for column, (_, row) in zip(
        model_columns,
        results_df.iterrows()
    ):

        model_name = row[
            "Model"
        ]


        with column:

            st.markdown(
                f"### {model_name}"
            )


            if saved_problem_type == "Regression":

                st.metric(
                    "MAE",
                    f"{row['MAE']:.3f}"
                )

                st.metric(
                    "RMSE",
                    f"{row['RMSE']:.3f}"
                )

                st.metric(
                    "R²",
                    f"{row['R²']:.3f}"
                )

                st.metric(
                    "Cross-Validated R²",
                    f"{row['CV Score']:.3f}"
                )

                st.caption(
                    f"CV variation: "
                    f"± {row['CV Std']:.3f}"
                )

            else:

                st.metric(
                    "Accuracy",
                    f"{row['Accuracy']:.3f}"
                )

                st.metric(
                    "Precision",
                    f"{row['Precision']:.3f}"
                )

                st.metric(
                    "Recall",
                    f"{row['Recall']:.3f}"
                )

                st.metric(
                    "F1",
                    f"{row['F1']:.3f}"
                )

                st.metric(
                    "Cross-Validated Accuracy",
                    f"{row['CV Score']:.3f}"
                )


    # ========================================================
    # BASELINE
    # ========================================================

    st.subheader(
        "Baseline Check"
    )


    if saved_problem_type == "Regression":

        st.write(
            f"""
            A simple model that always predicts the average
            target value has an MAE of
            **{st.session_state.baseline:.3f}**.

            The trained models should generally produce a
            lower MAE than this baseline to demonstrate
            useful predictive improvement.
            """
        )

    else:

        st.write(
            f"""
            A simple model that always predicts the most
            common class has an accuracy of
            **{st.session_state.baseline:.3f}**.

            The trained models should generally outperform
            this baseline.
            """
        )


    # ========================================================
    # EXPLORE MODEL
    # ========================================================

    st.divider()

    st.header(
        "Explore a Model"
    )


    explore_model = st.selectbox(
        "Choose one of the three models",
        results_df["Model"].tolist(),
        key="explore_model"
    )


    model_data = details[
        explore_model
    ]


    st.subheader(
        explore_model
    )


    # ========================================================
    # REGRESSION EXPLORATION
    # ========================================================

    if saved_problem_type == "Regression":

        st.write(
            """
            The graph below compares the model's predictions
            with the actual values in the held-out test data.
            """
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        actual = (
            model_data["table"][
                "Actual"
            ]
        )

        predicted = (
            model_data["table"][
                "Predicted"
            ]
        )


        ax.scatter(
            actual,
            predicted,
            alpha=0.7
        )


        minimum = min(
            actual.min(),
            predicted.min()
        )

        maximum = max(
            actual.max(),
            predicted.max()
        )


        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--"
        )


        ax.set_xlabel(
            f"Actual {saved_target}"
        )

        ax.set_ylabel(
            f"Predicted {saved_target}"
        )

        ax.set_title(
            "Actual vs. Predicted"
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        st.caption(
            """
            Points close to the diagonal line represent
            accurate predictions. Points farther from the
            line represent larger errors.
            """
        )


        # ----------------------------------------------------
        # ERROR DISTRIBUTION
        # ----------------------------------------------------

        st.subheader(
            "Prediction Error Distribution"
        )


        fig, ax = plt.subplots(
            figsize=(7, 4)
        )


        ax.hist(
            model_data["table"]["Error"],
            bins=15
        )


        ax.axvline(
            0,
            linestyle="--"
        )


        ax.set_xlabel(
            "Prediction Error"
        )

        ax.set_ylabel(
            "Frequency"
        )

        ax.set_title(
            "Distribution of Prediction Errors"
        )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        st.caption(
            """
            Errors clustered near zero indicate more accurate
            predictions. A wide distribution indicates that
            prediction errors vary substantially.
            """
        )


        # ----------------------------------------------------
        # LARGEST ERRORS
        # ----------------------------------------------------

        st.subheader(
            "Largest Prediction Errors"
        )


        largest_errors = (
            model_data["table"]
            .sort_values(
                "Absolute Error",
                ascending=False
            )
            .head(10)
        )


        st.dataframe(
            largest_errors.style.format(
                precision=3
            ),
            use_container_width=True
        )


    # ========================================================
    # CLASSIFICATION EXPLORATION
    # ========================================================

    else:

        y_test = (
            model_data["y_test"]
        )

        predictions = (
            model_data["predictions"]
        )


        labels = np.unique(
            np.concatenate(
                [
                    np.asarray(y_test),
                    np.asarray(predictions),
                ]
            )
        )


        cm = confusion_matrix(
            y_test,
            predictions,
            labels=labels
        )


        st.subheader(
            "Confusion Matrix"
        )


        fig, ax = plt.subplots(
            figsize=(7, 5)
        )


        ax.imshow(cm)


        ax.set_xticks(
            np.arange(len(labels))
        )

        ax.set_yticks(
            np.arange(len(labels))
        )


        ax.set_xticklabels(
            labels,
            rotation=45,
            ha="right"
        )

        ax.set_yticklabels(
            labels
        )


        ax.set_xlabel(
            "Predicted Class"
        )

        ax.set_ylabel(
            "Actual Class"
        )


        for i in range(
            len(labels)
        ):

            for j in range(
                len(labels)
            ):

                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center"
                )


        plt.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


        st.caption(
            """
            Values along the main diagonal are correct
            predictions. Values outside the diagonal show
            where the model confused one class with another.
            """
        )


    # ========================================================
    # FINAL WARNING
    # ========================================================

    st.divider()

    st.warning(
        """
        **Model performance should be interpreted carefully.**

        High scores do not automatically mean a model will
        perform equally well on future data. Identifiers,
        duplicated information, highly related variables,
        small datasets, and data leakage can make performance
        appear stronger than it really is.

        Consider the meaning and origin of your variables when
        interpreting these results.
        """
    )