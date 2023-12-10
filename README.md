# costa_booking_probability

**Data Preprocessing and Model Training:**

**dataset.csv** is the input file furnished by Costa. 10% of rows that contain null in the booking column are dropped and the other rows are used for train and test.

**model_rfc.ipynb** is the notebook that processes the data and trains a **RandomForestClassifier** to predict the booking probability. The final model with class_weight is saved as model_rfc.pkl. The goal of training and optimization was to detect the customers who wanted to book as much as possible. It causes the algorithm to predict more class zero as class one wrongly.

**prediction.csv** is the predicted output with two columns: ID and predicted_booking. The total number of predicted class one should be more than the reality as it is explained in the model training goal.

**Prediction:**

**Home_Page.py** is for testing the 10% of rows with null booking using a **Streamlit** app to upload the input dataset and get the predictions.

![st_1](https://github.com/samedhsh/costa_booking_probability/assets/80158302/ee4e94ff-a9d2-4f75-81e3-a0d88fb5eb57)

after uploading data, by clicking on the Show DataFrame box, the input dataset will be shown. 

![st_2](https://github.com/samedhsh/costa_booking_probability/assets/80158302/23311706-41ec-4c42-827c-5329b7cc02eb)

Finally, by clicking on the Perform Prediction button, the statistics of the input dataset with the prediction will appear.

![st_3](https://github.com/samedhsh/costa_booking_probability/assets/80158302/1fdaf7ce-68a3-4690-b054-5ac7deebbe31)


There are two ways to run the prediction after cloning the git repo:

1- navigating in the **CMD** to the app/ directory and run the Home_Page.py with the following command:
            **>>>  python -m streamlit run Home_Page.py**

2- Running the **Dockerfile** locally to build the Docker image and run it.




