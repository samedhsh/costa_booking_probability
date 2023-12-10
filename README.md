# costa_booking_probability

Data Preprocessing and Model Training:

dataset.csv is the input file furnished by Costa. 10% of rows that contain null in the booking column are dropped and the other rows are used for train and test.

model_rfc.ipynb is the notebook that processes the data and trains a RandomForestClassifier to predict the booking probability. The final model with class_weight is saved as model_rfc.pkl.

Prediction:
Home_Page.py is for testing the 10% of rows with null booking using a Streamlit app to upload the input dataset and get the predictions.
![st_1](https://github.com/samedhsh/costa_booking_probability/assets/80158302/ee4e94ff-a9d2-4f75-81e3-a0d88fb5eb57)
