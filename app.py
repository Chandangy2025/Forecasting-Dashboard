import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import os

warnings.filterwarnings("ignore")

DATA_FOLDER = "data"

plt.switch_backend("Agg")

# ============================
# Build folder structure
# ============================

folder_map = {}

for company in os.listdir(DATA_FOLDER):

    cp = os.path.join(DATA_FOLDER, company)

    if os.path.isdir(cp):

        folder_map[company] = {}

        for region in os.listdir(cp):

            rp = os.path.join(cp, region)

            if os.path.isdir(rp):

                therapy_dict = {}

                for item in os.listdir(rp):

                    ip = os.path.join(rp, item)

                    if os.path.isdir(ip):

                        files = [

                            f.replace(".csv", "")

                            for f in os.listdir(ip)

                            if f.endswith(".csv")

                        ]

                        if files:

                            therapy_dict[item] = files

                if therapy_dict:

                    folder_map[company][region] = therapy_dict

                else:

                    files = [

                        f.replace(".csv", "")

                        for f in os.listdir(rp)

                        if f.endswith(".csv")

                    ]

                    folder_map[company][region] = {

                        "_no_therapy": files

                    }

# ============================
# Helpers
# ============================

def get_companies():

    return list(folder_map.keys())


def get_regions(c):

    return list(folder_map.get(c, {}).keys())


def get_therapy_areas(c, r):

    d = folder_map[c][r]

    return [] if "_no_therapy" in d else list(d.keys())


def get_devices(c, r, t=None):

    d = folder_map[c][r]

    if "_no_therapy" in d:

        return d["_no_therapy"]

    return d.get(t, [])


def get_file_path(c, r, d, t=None):

    if t:

        return os.path.join(

            DATA_FOLDER,
            c,
            r,
            t,
            d + ".csv"

        )

    return os.path.join(

        DATA_FOLDER,
        c,
        r,
        d + ".csv"

    )

# ============================
# COMMON DATA CLEANING FIX
# ============================

def load_and_clean(fp):

    df = pd.read_csv(fp)

    df["Date"] = pd.to_datetime(

        df["Date"],
        errors="coerce"

    )

    # ==================================
    # HANDLE DUPLICATE DATES
    # ==================================

    df = df.groupby("Date").agg({

        "Sales": "sum",

        "GrowthRate": "mean"

    }).reset_index()

    df = df.sort_values("Date")

    df = df.set_index("Date")

    ts = df["Sales"].asfreq("QE").astype(float).dropna()

    exog = df["GrowthRate"].asfreq("QE").astype(float).dropna()

    return ts, exog

# ============================
# Optimized dropdown
# ============================

def optimized_region_update(c, r):

    if not c or not r:

        return [], [], False

    t_list = get_therapy_areas(c, r)

    if t_list:

        d_list = get_devices(c, r, t_list[0])

        return t_list, d_list, True

    else:

        d_list = get_devices(c, r)

        return [], d_list, False

# ============================
# EVALUATION
# ============================

def evaluate_dashboard(c, r, t, d, model_type):

    try:

        fp = get_file_path(c, r, d, t)

        ts, exog = load_and_clean(fp)

        TEST_Q = 4

        if len(ts) <= TEST_Q + 2:

            return pd.DataFrame({

                "Error": ["Dataset too small"]

            }), None

        train = ts.iloc[:-TEST_Q]

        test = ts.iloc[-TEST_Q:]

        train_exog = exog.iloc[:-TEST_Q]

        test_exog = exog.iloc[-TEST_Q:]

        best_aic = np.inf

        best_model = None

        p = d = q = range(0, 2)

        # ==================================
        # SARIMAX
        # ==================================

        if model_type == "SARIMAX":

            P = D = Q = range(0, 2)

            s = 4

            for (pi, di, qi) in product(p, d, q):

                for (Pi, Di, Qi) in product(P, D, Q):

                    try:

                        m = SARIMAX(

                            train,

                            exog=train_exog,

                            order=(pi, di, qi),

                            seasonal_order=(Pi, Di, Qi, s),

                            trend="c",

                            enforce_stationarity=False,

                            enforce_invertibility=False

                        ).fit(disp=False)

                        if m.aic < best_aic:

                            best_aic = m.aic

                            best_model = m

                    except:

                        pass

        # ==================================
        # ARIMAX
        # ==================================

        else:

            for (pi, di, qi) in product(p, d, q):

                try:

                    m = SARIMAX(

                        train,

                        exog=train_exog,

                        order=(pi, di, qi),

                        seasonal_order=(0, 0, 0, 0),

                        trend="c",

                        enforce_stationarity=False,

                        enforce_invertibility=False

                    ).fit(disp=False)

                    if m.aic < best_aic:

                        best_aic = m.aic

                        best_model = m

                except:

                    pass

        pred = best_model.get_forecast(

            steps=len(test),

            exog=test_exog

        ).predicted_mean

        acc = 100 - (

            abs(test - pred)

            / test * 100

        )

        accuracy_df = pd.DataFrame({

            "Actual": test,

            "Predicted": pred,

            "Accuracy %": acc

        })

        # ==================================
        # PLOT
        # ==================================

        fig, ax = plt.subplots(

            figsize=(12, 6)

        )

        train.plot(ax=ax, label="Training")

        test.plot(ax=ax, label="Testing")

        pred.plot(ax=ax, label="Predicted")

        ax.legend()

        ax.set_title(

            f"{d} Evaluation"

        )

        return accuracy_df, fig

    except Exception as e:

        return pd.DataFrame({

            "Error": [str(e)]

        }), None

# ============================
# FORECAST MODEL
# ============================

def run_forecast_model(fp, model_type, horizon):

    ts, exog = load_and_clean(fp)

    if len(ts) < 6:

        return None

    best_aic = np.inf

    best_order = None

    best_seasonal = (0, 0, 0, 0)

    p = d = q = range(0, 2)

    # ==================================
    # SARIMAX
    # ==================================

    if model_type == "SARIMAX":

        P = D = Q = range(0, 2)

        s = 4

        for (pi, di, qi) in product(p, d, q):

            for (Pi, Di, Qi) in product(P, D, Q):

                try:

                    m = SARIMAX(

                        ts,

                        exog=exog,

                        order=(pi, di, qi),

                        seasonal_order=(Pi, Di, Qi, s)

                    ).fit(disp=False)

                    if m.aic < best_aic:

                        best_aic = m.aic

                        best_order = (pi, di, qi)

                        best_seasonal = (Pi, Di, Qi, s)

                except:

                    pass

    # ==================================
    # ARIMAX
    # ==================================

    else:

        for (pi, di, qi) in product(p, d, q):

            try:

                m = SARIMAX(

                    ts,

                    exog=exog,

                    order=(pi, di, qi),

                    seasonal_order=(0, 0, 0, 0)

                ).fit(disp=False)

                if m.aic < best_aic:

                    best_aic = m.aic

                    best_order = (pi, di, qi)

            except:

                pass

    final_model = SARIMAX(

        ts,

        exog=exog,

        order=best_order,

        seasonal_order=best_seasonal

    ).fit(disp=False)

    h = int(horizon)

    start = ts.index[-1] + pd.offsets.QuarterEnd()

    future_idx = pd.date_range(

        start,

        periods=h,

        freq="QE"

    )

    exog_model = SARIMAX(

        exog,

        order=(1, 0, 0)

    ).fit(disp=False)

    future_exog = exog_model.get_forecast(

        steps=h

    ).predicted_mean

    fc = final_model.get_forecast(

        steps=h,

        exog=future_exog

    )

    return pd.Series(

        fc.predicted_mean,

        index=future_idx

    )

# ============================
# FULL COMPANY FORECAST
# ============================

def forecast_full_company(c, model, horizon):

    results = []

    for r in get_regions(c):

        t_list = get_therapy_areas(c, r)

        # ==================================
        # WITH THERAPY
        # ==================================

        if t_list:

            for t in t_list:

                for d in get_devices(c, r, t):

                    fp = get_file_path(c, r, d, t)

                    try:

                        # ======================
                        # HISTORICAL
                        # ======================

                        raw_df = pd.read_csv(fp)

                        raw_df["Date"] = pd.to_datetime(

                            raw_df["Date"],
                            errors="coerce"

                        )

                        for _, row in raw_df.iterrows():

                            results.append([

                                c,
                                r,
                                t,
                                d,
                                row["Date"].date(),
                                row["Sales"],
                                "Historical"

                            ])

                        # ======================
                        # FORECAST
                        # ======================

                        fc = run_forecast_model(

                            fp,
                            model,
                            horizon

                        )

                        if fc is None:

                            continue

                        for dt, val in fc.items():

                            results.append([

                                c,
                                r,
                                t,
                                d,
                                dt.date(),
                                val,
                                "Forecast"

                            ])

                    except Exception as e:

                        print(e)

        # ==================================
        # WITHOUT THERAPY
        # ==================================

        else:

            for d in get_devices(c, r):

                fp = get_file_path(c, r, d)

                try:

                    raw_df = pd.read_csv(fp)

                    raw_df["Date"] = pd.to_datetime(

                        raw_df["Date"],
                        errors="coerce"

                    )

                    for _, row in raw_df.iterrows():

                        results.append([

                            c,
                            r,
                            "NA",
                            d,
                            row["Date"].date(),
                            row["Sales"],
                            "Historical"

                        ])

                    fc = run_forecast_model(

                        fp,
                        model,
                        horizon

                    )

                    if fc is None:

                        continue

                    for dt, val in fc.items():

                        results.append([

                            c,
                            r,
                            "NA",
                            d,
                            dt.date(),
                            val,
                            "Forecast"

                        ])

                except Exception as e:

                    print(e)

    df = pd.DataFrame(results, columns=[

        "Company",
        "Region",
        "Therapy",
        "Device",
        "Date",
        "Value",
        "Type"

    ])

    file = "full_company_forecast.csv"

    df.to_csv(file, index=False)

    return df, file

# ============================
# STREAMLIT PAGE
# ============================

st.set_page_config(

    page_title="MedTech Forecast Dashboard",

    layout="wide"

)

# ============================
# TITLE
# ============================

st.title(

    "📈 MedTech Forecasting Dashboard"

)

st.markdown("---")

# ============================
# TABS
# ============================

tab1, tab2 = st.tabs([

    "Model Evaluation",
    "Company Forecast"

])

# ============================
# TAB 1
# ============================

with tab1:

    st.sidebar.header("Evaluation Settings")

    company = st.sidebar.selectbox(

        "Company",

        get_companies()

    )

    region = st.sidebar.selectbox(

        "Region",

        get_regions(company)

    )

    therapy_list = get_therapy_areas(

        company,
        region

    )

    therapy = None

    if therapy_list:

        therapy = st.sidebar.selectbox(

            "Therapy",

            therapy_list

        )

    devices = get_devices(

        company,
        region,
        therapy

    )

    device = st.sidebar.selectbox(

        "Device",

        devices

    )

    model = st.sidebar.selectbox(

        "Model",

        ["SARIMAX", "ARIMAX"]

    )

    if st.sidebar.button("Evaluate Model"):

        with st.spinner(

            "Running Evaluation..."

        ):

            out, fig = evaluate_dashboard(

                company,
                region,
                therapy,
                device,
                model

            )

        st.subheader("Accuracy Output")

        st.dataframe(

            out,

            use_container_width=True

        )

        if fig:

            st.pyplot(fig)

# ============================
# TAB 2
# ============================

with tab2:

    st.sidebar.header("Company Forecast Settings")

    comp2 = st.sidebar.selectbox(

        "Forecast Company",

        get_companies(),

        key="fc"

    )

    model2 = st.sidebar.selectbox(

        "Forecast Model",

        ["SARIMAX", "ARIMAX"],

        key="m2"

    )

    horizon = st.sidebar.slider(

        "Forecast Horizon",

        1,
        48,
        4,

        key="h2"

    )

    if st.sidebar.button(

        "Run Full Company Forecast"

    ):

        with st.spinner(

            "Running Full Forecast..."

        ):

            table, file = forecast_full_company(

                comp2,
                model2,
                horizon

            )

        st.success(

            "Forecast Completed"

        )

        st.dataframe(

            table,

            use_container_width=True

        )

        with open(file, "rb") as f:

            st.download_button(

                "Download Forecast CSV",

                f,

                file_name=file,

                mime="text/csv"

            )
