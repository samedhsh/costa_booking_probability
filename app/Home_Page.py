import streamlit as st
from streamlit_extras.app_logo import add_logo
from streamlit_extras.colored_header import colored_header
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
load_dotenv()

# -------------------------------------------------------------
# Set page configuration
# -------------------------------------------------------------
st.set_page_config(page_title="Batch prediction of booking probability",
                    # page_icon = "",
                    # layout = "wide",
                    )

# -------------------------------------------------------------
# Header of the page
# -------------------------------------------------------------
colored_header(
    label="Batch prediction of booking probability",
    description=" ",
    color_name="green-70",
)

# -------------------------------------------------------------
# Context within page
# -------------------------------------------------------------
# Add images
image1, image2, image3 = st.columns(3)
image2.image("./logos/costa_logo.jpg", width = 200)
# image2.image("./logos/costa_logo2.jpg", width = 200)

# Add text
st.write(""" 
         
This page is for batch prediction of booking probability of cruises. CSV is the only acceptable input dataset format.
         
The columns should be the same as provided word file by Costa for booking probability

""")

def perform_predictions(df):

    ids_to_predict = df["ID"]

    df.drop(columns=[
                  "occupation_id_desc", "privacy_channel_email_date", "privacy_channel_phone_date", "privacy_channel_post_date",
                  "privacy_send_material_date", "loyalty_subscribe_date", "last_nps_segment", "flag_bkd_cxl_last3y",
                  "flag_bkd_cxl_last_month", "flag_opt_cxl_last3y", "flag_opt_cxl_last_month", "num_cruises_last3y",
                  "total_gross_cru_rev_eur_mean", "total_gross_obr_rev_eur_mean", "source_desc", "opener_segment", "last_pax_type"], 
                   axis=1, inplace=True) 
    
    ##################################################################
    ### fillna for prediction

    mode_pcc_flag = df["flag_app"].mode().values[0]
    df["flag_app"].fillna(mode_pcc_flag, inplace=True)

    mode_last_cruise_duration = df["last_cruise_duration"].mode().values[0]
    df["last_cruise_duration"].fillna(mode_last_cruise_duration, inplace=True)

    median_last_anticipation = df["last_cruise_bkg_anticipation"].median()
    df["last_cruise_bkg_anticipation"].fillna(median_last_anticipation, inplace=True)

    ##############################################################################
    ### changing to categorical column from numerical

    bins = [-np.inf, 90, 730, 1460, np.inf]       # based on their distribution
    names = ['less_than_3month', 'between_3month_and_2year','between_2year_and_4year', 'more_than_4year']

    df['period_after_cruise_end'] = pd.cut(df['days_after_cruise_end'], bins, labels=names)
    df.drop(columns=["days_after_cruise_end"], axis=1, inplace=True)

    bins2 = [-np.inf, 0, 4, np.inf] # based on their distribution
    names2 = ['0', 'between_0_and_4','more_than_4']

    df['category_click_last_month'] = pd.cut(df['click_last_month'], bins2, labels=names2)
    df['category_open_last_month'] = pd.cut(df['open_last_month'], bins2, labels=names2)
    df['category_sent_last_month'] = pd.cut(df['sent_last_month'], bins2, labels=names2)
    df.drop(columns=["click_last_month", "open_last_month", "sent_last_month"], axis=1, inplace=True)

    ##############################################################################
    ### encoding and scaling of categorical and numerical features

    categorical_columns = ["had_a_flight_in_last_cruise", "BKG_Channel", "rfm_segment",  "period_after_cruise_end",
                            "last_paid_cabin_meta_cat_code", "category_open_last_month", 
                            "category_sent_last_month"
                            ] #,"last_cruise_program_desc", ,"category_click_last_month"

    numerical_columns = ["AGE", "LOYALTY_TOTAL_SCORE", "is_loyalty", "NUMBER_OF_CRUISES", "pcc_flag", "flag_app", 
                        "last_cruise_duration", "last_cruise_bkg_anticipation"
                        ]

    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder.fit(df[categorical_columns])
    encoded_df = encoder.transform(df[categorical_columns]).toarray()
    encoded_df = pd.DataFrame(encoded_df, columns=encoder.get_feature_names(categorical_columns))

    scaler = MinMaxScaler(feature_range=(0,1))

    # fit and transform the numerical columns of the dataframe
    scaled_df = scaler.fit_transform(df[numerical_columns])

    # create a new dataframe with the scaled values
    scaled_df = pd.DataFrame(scaled_df, columns=numerical_columns)

    X_test = pd.concat([encoded_df, scaled_df], axis=1) ## merging the encoded categorical and numerical features

    with open("model_rfc.pkl",'rb') as model_path:
        model = pickle.load(model_path)

    y_pred = model.predict(X_test)
    y_pred_serie = pd.Series(y_pred)

    df_output = pd.concat([ids_to_predict, y_pred_serie], axis=1)
    df_output = df_output.rename(columns={ 0: 'Predicted_booking'}) 
    
    return df_output

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';')
        st.success("File successfully uploaded and loaded into DataFrame!")

        if st.checkbox("Show DataFrame"):
            st.write(df)
        
        if st.button('Perform Prediction'):
            # Perform operations or analysis on the DataFrame here
            st.write("Summary Statistics:")
            st.write(df.drop(columns=["booking", "ID"], axis=1).describe())

            df_output = perform_predictions(df=df)
            st.write("Output Dataset:")
            st.write(df_output)

            # You can add more operations or analysis as needed
    except Exception as e:
        st.error(f"An error occurred: {e}")
