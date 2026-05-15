# ============================================================
# INSTALL REQUIRED LIBRARIES
# ============================================================
# Run this in VS Code terminal first:
#
# py -m pip install streamlit pandas numpy matplotlib statsmodels
#
# Then run app:
#
# py -m streamlit run app.py
#
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX

import warnings
import os

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

DATA_FOLDER = "data"

plt.switch_backend("Agg")

st.set_page_config(
    page_title="MedTech Forecast Dashboard",
    layout="wide"
)

# ============================================================
# BUILD FOLDER STRUCTURE
# ============================================================

folder_map = {}

for company in os.listdir(DATA_FOLDER):

    company_path = os.path.join(DATA_FOLDER, company)

    if os.path.isdir(company_path):

        if company.startswith("."):
            continue

        folder_map[company] = {}

        for region in os.listdir(company_path):

            region_path = os.path.join(company_path, region)

            if os.path.isdir(region_path):

                therapy_dict = {}

                for item in os.listdir(region_path):

                    item_path = os.path.join(region_path, item)

                    # ============================================
                    # THERAPY AREA FOLDER
                    # ============================================

                    if os.path.isdir(item_path):

                        csv_files = [

                            f.replace(".csv", "")

                            for f in os.listdir(item_path)

                            if f.endswith(".csv")

                        ]

                        if csv_files:

                            therapy_dict[item] = csv_files

                # ============================================
                # REGION WITH THERAPY
                # ============================================

                if therapy_dict:

                    folder_map[company][region] = therapy_dict

                # ============================================
                # REGION WITH DIRECT CSV
                # ============================================

                else:

                    csv_files = [

                        f.replace(".csv", "")

                        for f in os.listdir(region_path)

                        if f.endswith(".csv")

                    ]

                    if csv_files:

                        folder_map[company][region] = {

                            "_no_therapy": csv_files

                        }

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_companies():

    return sorted(list(folder_map.keys()))


def get_regions(company):

    return sorted(list(folder_map.get(company, {}).keys()))


def get_therapy_areas(company, region):

    if company not in folder_map:
        return []

    if region not in folder_map[company]:
        return []

    therapy_dict = folder_map[company][region]

    if "_no_therapy" in therapy_dict:
        return []

    return sorted(list(therapy_dict.keys()))


def get_devices(company, region, therapy=None):

    if company not in folder_map:
        return []

    if region not in folder_map[company]:
        return []

    therapy_dict = folder_map[company][region]

    if "_no_therapy" in therapy_dict:

        return therapy_dict["_no_therapy"]

    if therapy:

        return therapy_dict.get(therapy, [])

    return []


def get_file_path(company, region, device, therapy=None):

    if therapy:

        return os.path.join(
            DATA_FOLDER,
            company,
            region,
            therapy,
            device + ".csv"
        )

    return os.path.join(
        DATA_FOLDER,
        company,
        region,
        device + ".csv"
    )

# ============================================================
# LOAD SERIES
# ============================================================

def load_series(file_path):

    df = pd.read_csv(file_path)

    df["Date"] = pd.to_datetime(
        df["Date"],
        format="%d-%m-%Y",
        errors="coerce"
    )

    df = df.groupby("Date", as_index=False).agg({

    "Sales": "sum",

    "GrowthRate": "mean"

    })


    df = df.dropna(
        subset=["Date", "Sales", "GrowthRate"]
    )

    df = df.sort_values("Date")

    df = df.set_index("Date")

    ts_q = df["Sales"].asfreq("QE").astype(float).dropna()

    exog = df["GrowthRate"].asfreq("QE").astype(float).dropna()

    return df, ts_q, exog

# ============================================================
# MODEL TRAINING
# ============================================================

def get_best_model(train_data, train_exog, model_type):

    best_aic = np.inf

    best_order = None

    best_seasonal = (0, 0, 0, 0)

    best_model = None

    # ========================================================
    # SARIMAX
    # ========================================================

    if model_type == "SARIMAX":

        s = 4

        p = d = q = range(0, 2)

        P = D = Q = range(0, 2)

        for (pi, di, qi) in product(p, d, q):

            for (Pi, Di, Qi) in product(P, D, Q):

                try:

                    model = SARIMAX(

                        train_data,

                        exog=train_exog,

                        order=(pi, di, qi),

                        seasonal_order=(Pi, Di, Qi, s),

                        trend="c",

                        enforce_stationarity=False,

                        enforce_invertibility=False

                    )

                    res = model.fit(disp=False)

                    if res.aic < best_aic:

                        best_aic = res.aic

                        best_order = (pi, di, qi)

                        best_seasonal = (Pi, Di, Qi, s)

                        best_model = res

                except:
                    pass

    # ========================================================
    # ARIMAX
    # ========================================================

    else:

        p = d = q = range(0, 2)

        for (pi, di, qi) in product(p, d, q):

            try:

                model = SARIMAX(

                    train_data,

                    exog=train_exog,

                    order=(pi, di, qi),

                    seasonal_order=(0, 0, 0, 0),

                    trend="c",

                    enforce_stationarity=False,

                    enforce_invertibility=False

                )

                res = model.fit(disp=False)

                if res.aic < best_aic:

                    best_aic = res.aic

                    best_order = (pi, di, qi)

                    best_seasonal = (0, 0, 0, 0)

                    best_model = res

            except:
                pass

    return best_model, best_order, best_seasonal

# ============================================================
# SINGLE DEVICE FORECAST
# ============================================================

def forecast_single_device(

    company,
    region,
    therapy,
    device,
    model_type,
    horizon

):

    try:

        file_path = get_file_path(

            company,
            region,
            device,
            therapy

        )

        df, ts_q, exog = load_series(file_path)

        TEST_Q = 4

        if len(ts_q) <= TEST_Q + 2:

            return None, None, None, None

        # ====================================================
        # TRAIN TEST SPLIT
        # ====================================================

        train_data = ts_q.iloc[:-TEST_Q]

        test_data = ts_q.iloc[-TEST_Q:]

        train_exog = exog.iloc[:-TEST_Q]

        test_exog = exog.iloc[-TEST_Q:]

        # ====================================================
        # MODEL SEARCH
        # ====================================================

        best_model, best_order, best_seasonal = get_best_model(

            train_data,
            train_exog,
            model_type

        )

        # ====================================================
        # TEST FORECAST
        # ====================================================

        test_forecast = best_model.get_forecast(

            steps=len(test_data),

            exog=test_exog

        )

        test_pred = pd.Series(

            test_forecast.predicted_mean,

            index=test_data.index

        )

        accuracy = 100 - (

            abs(test_data - test_pred)

            / test_data * 100

        )

        # ====================================================
        # FUTURE FORECAST
        # ====================================================

        h = int(horizon)

        start = ts_q.index[-1] + pd.offsets.QuarterEnd()

        future_index = pd.date_range(

            start,

            periods=h,

            freq="QE"

        )

        exog_model = SARIMAX(

            exog,

            order=(1, 0, 0)

        )

        exog_fit = exog_model.fit(disp=False)

        future_exog = pd.Series(

            exog_fit.get_forecast(

                steps=h

            ).predicted_mean,

            index=future_index

        )

        final_model = SARIMAX(

            ts_q,

            exog=exog,

            order=best_order,

            seasonal_order=best_seasonal,

            trend="c",

            enforce_stationarity=False,

            enforce_invertibility=False

        ).fit(disp=False)

        future_fc = final_model.get_forecast(

            steps=h,

            exog=future_exog

        )

        future_mean = future_fc.predicted_mean

        # ====================================================
        # CREATE FINAL TABLE
        # ====================================================

        results = []

        # Historical
        for dt, val in ts_q.items():

            results.append([

                company,
                region,
                therapy,
                device,
                dt.date(),
                round(val, 2),
                "Historical"

            ])

        # Test Predictions
        for dt, val in test_pred.items():

            results.append([

                company,
                region,
                therapy,
                device,
                dt.date(),
                round(val, 2),
                "Test Prediction"

            ])

        # Forecast
        for dt, val in future_mean.items():

            results.append([

                company,
                region,
                therapy,
                device,
                dt.date(),
                round(val, 2),
                "Forecast"

            ])

        final_df = pd.DataFrame(

            results,

            columns=[

                "Company",
                "Region",
                "Therapy",
                "Device",
                "Date",
                "Value",
                "Type"

            ]

        )

        # ====================================================
        # PLOT
        # ====================================================

        fig, ax = plt.subplots(figsize=(12, 6))

        train_data.plot(ax=ax, label="Training")

        test_data.plot(ax=ax, label="Testing")

        test_pred.plot(ax=ax, label="Test Prediction")

        future_mean.plot(ax=ax, label="Future Forecast")

        ax.legend()

        ax.set_title(f"{device} Forecast")

        return final_df, fig, accuracy.mean(), future_mean.iloc[-1]

    except Exception as e:

        st.error(str(e))

        return None, None, None, None

# ============================================================
# FULL COMPANY FORECAST
# ============================================================

def forecast_full_company(

    company,
    model_type,
    horizon

):

    all_results = []

    summary_rows = []

    company_regions = get_regions(company)

    for region in company_regions:

        therapy_list = get_therapy_areas(company, region)

        # ====================================================
        # NO THERAPY
        # ====================================================

        if len(therapy_list) == 0:

            devices = get_devices(company, region)

            for device in devices:

                try:

                    df, _, acc, latest_fc = forecast_single_device(

                        company,
                        region,
                        None,
                        device,
                        model_type,
                        horizon

                    )

                    if df is not None:

                        all_results.append(df)

                        summary_rows.append([

                            company,
                            region,
                            "",
                            device,
                            round(acc, 2),
                            round(latest_fc, 2)

                        ])

                except:
                    pass

        # ====================================================
        # WITH THERAPY
        # ====================================================

        else:

            for therapy in therapy_list:

                devices = get_devices(

                    company,
                    region,
                    therapy

                )

                for device in devices:

                    try:

                        df, _, acc, latest_fc = forecast_single_device(

                            company,
                            region,
                            therapy,
                            device,
                            model_type,
                            horizon

                        )

                        if df is not None:

                            all_results.append(df)

                            summary_rows.append([

                                company,
                                region,
                                therapy,
                                device,
                                round(acc, 2),
                                round(latest_fc, 2)

                            ])

                    except:
                        pass

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    if len(all_results) == 0:

        return None, None, None

    final_output = pd.concat(all_results)

    summary_df = pd.DataFrame(

        summary_rows,

        columns=[

            "Company",
            "Region",
            "Therapy",
            "Device",
            "Mean Accuracy %",
            "Latest Forecast"

        ]

    )

    output_file = "full_company_forecast.csv"

    final_output.to_csv(

        output_file,

        index=False

    )

    return final_output, summary_df, output_file

# ============================================================
# TITLE
# ============================================================

st.title(
    "📈 MedTech Enterprise Forecasting Dashboard"
)

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Forecast Settings")

companies = get_companies()

if len(companies) == 0:

    st.error("No company folders found.")

    st.stop()

# ============================================================
# MODE
# ============================================================

mode = st.sidebar.radio(

    "Select Forecast Mode",

    [

        "Single Device Forecast",

        "Full Company Forecast"

    ]

)

# ============================================================
# MODEL
# ============================================================

model = st.sidebar.selectbox(

    "Select Forecast Model",

    ["SARIMAX", "ARIMAX"]

)

# ============================================================
# HORIZON
# ============================================================

horizon = st.sidebar.slider(

    "Forecast Horizon (Quarters)",

    1,
    48,
    4

)

# ============================================================
# SINGLE DEVICE MODE
# ============================================================

if mode == "Single Device Forecast":

    company = st.sidebar.selectbox(

        "Select Company",

        companies

    )

    regions = get_regions(company)

    region = st.sidebar.selectbox(

        "Select Region",

        regions

    )

    therapy_list = get_therapy_areas(

        company,
        region

    )

    therapy = None

    if len(therapy_list) > 0:

        therapy = st.sidebar.selectbox(

            "Select Therapy Area",

            therapy_list

        )

    devices = get_devices(

        company,
        region,
        therapy

    )

    device = st.sidebar.selectbox(

        "Select Device",

        devices

    )

    run_btn = st.sidebar.button(

        "Run Device Forecast"

    )

    if run_btn:

        with st.spinner(

            "Running Forecast..."

        ):

            result_df, fig, acc, latest_fc = forecast_single_device(

                company,
                region,
                therapy,
                device,
                model,
                horizon

            )

        if result_df is not None:

            st.success(

                "Forecast Completed"

            )

            col1, col2 = st.columns(2)

            col1.metric(

                "Mean Accuracy %",

                round(acc, 2)

            )

            col2.metric(

                "Latest Forecast",

                round(latest_fc, 2)

            )

            st.subheader(

                "Forecast Output"

            )

            st.dataframe(

                result_df,

                use_container_width=True

            )

            st.subheader(

                "Visualization"

            )

            st.pyplot(fig)

            csv = result_df.to_csv(index=False)

            st.download_button(

                "📥 Download CSV",

                csv,

                file_name="device_forecast.csv",

                mime="text/csv"

            )

# ============================================================
# FULL COMPANY MODE
# ============================================================

else:

    company = st.sidebar.selectbox(

        "Select Company",

        companies

    )

    run_btn = st.sidebar.button(

        "Run Full Company Forecast"

    )

    if run_btn:

        with st.spinner(

            "Running Full Company Forecast..."

        ):

            final_output, summary_df, output_file = forecast_full_company(

                company,
                model,
                horizon

            )

        if final_output is not None:

            st.success(

                "Full Company Forecast Completed"

            )

            st.subheader(

                "Forecast Summary"

            )

            st.dataframe(

                summary_df,

                use_container_width=True

            )

            st.subheader(

                "Full Historical + Test + Forecast Output"

            )

            st.dataframe(

                final_output,

                use_container_width=True,

                height=600

            )

            with open(output_file, "rb") as f:

                st.download_button(

                    label="📥 Download Full Company Forecast CSV",

                    data=f,

                    file_name="full_company_forecast.csv",

                    mime="text/csv"

                )
