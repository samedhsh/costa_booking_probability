# costa_booking_probability

Data Preprocessing and Model Training:

dataset.csv is input file which is furnished by Costa. 10% of rows which contain null in booking column are dropped and the other rows are used for train and test.

model_rfc.ipynb is the notebook which processes the data and trains a RandomForestClassifier in order to predict the booking probability. The final model with class_weight is saved as model_rfc.pkl.

Prediction:
Home_Page.py is for testing the 10% of rows with null booking using a streamlit app to upload the input dataset and get the predictions.
