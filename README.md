# CUSTOM_GENAI_CHATBOT


import time
starttime=time.time()

#Importing Libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from functools import reduce
from pyspark.sql.types import ArrayType, StringType
import re
from pyspark.sql import functions as F
from pyspark.sql.functions import count,isnan,col,when, lit
from pyspark.sql.types import ArrayType, StringType, StructType, StructField

# Define the custom tokenizer
def custom_tokenizer(text):
    pattern = re.compile(r'[a-zA-Z]+\d+')
    return pattern.findall(text)

# Load the saved models and vectorizers
clf_cat1 = pickle.load(open("clf_cat1.pkl", "rb"))
clf_cat2 = pickle.load(open("clf_cat2.pkl", "rb"))
tfidf_payer_name = pickle.load(open("tfidf_payer_name.pkl", "rb"))
tfidf_payee_name = pickle.load(open("tfidf_payee_name.pkl", "rb"))
tfidf_payee_account_type = pickle.load(open("tfidf_payee_account_type.pkl", "rb"))
tfidf_payer_account_type = pickle.load(open("tfidf_payer_account_type.pkl", "rb"))
tfidf_payer_vpa = pickle.load(open("tfidf_payer_vpa.pkl", "rb"))
tfidf_payee_vpa = pickle.load(open("tfidf_payee_vpa.pkl", "rb"))

# Define the prediction function with handling for None values
def predict_categories(payer_name, payee_name, payee_account_type,
                       payer_account_type, payer_vpa, payee_vpa):
    
    # Handle potential None values
    payer_name = '' if payer_name is None else payer_name
    payee_name = '' if payee_name is None else payee_name
    payee_account_type = '' if payee_account_type is None else payee_account_type
    payer_account_type = '' if payer_account_type is None else payer_account_type
    payer_vpa = '' if payer_vpa is None else payer_vpa
    payee_vpa = '' if payee_vpa is None else payee_vpa
    
    # Transform input data using the TFIDF vectorizers
    payer_name_vec = tfidf_payer_name.transform([payer_name])
    payee_name_vec = tfidf_payee_name.transform([payee_name])
    payee_account_type_vec = tfidf_payee_account_type.transform([payee_account_type])
    payer_account_type_vec = tfidf_payer_account_type.transform([payer_account_type])
    payer_vpa_vec = tfidf_payer_vpa.transform([payer_vpa])
    payee_vpa_vec = tfidf_payee_vpa.transform([payee_vpa])

    tfidf_matrix = pd.concat([pd.DataFrame(payer_name_vec.toarray()),
                              pd.DataFrame(payee_name_vec.toarray()),
                              pd.DataFrame(payee_account_type_vec.toarray()),
                              pd.DataFrame(payer_account_type_vec.toarray()),
                              pd.DataFrame(payer_vpa_vec.toarray()),
                              pd.DataFrame(payee_vpa_vec.toarray())], axis=1)

    # Predict
    prediction_cat1 = clf_cat1.predict(tfidf_matrix)
    prediction_cat2 = clf_cat2.predict(tfidf_matrix)

    return [prediction_cat1[0], prediction_cat2[0]]

def compute_aggregates(df, category_col, amount_col):
    # Pivot on the category column to compute counts and sums for each category
    agg_df = df.groupBy("payer_account_number", "payer_account_type").pivot(category_col).agg(
        F.count(amount_col).alias("count"),
        F.sum(amount_col).alias("sum")
    )
    
    # Rename columns and add value type column based on the sum
    for category in df.select(category_col).distinct().rdd.flatMap(lambda x: x).collect():
        agg_df = agg_df.withColumnRenamed(category + "_count", "count_" + category) \
                      .withColumnRenamed(category + "_sum", "sum_" + category) \
                      .withColumn("type_" + category, 
                                  F.when((F.col("count_" + category) == 0) | (F.col("sum_" + category).isNull()) | (F.col("sum_" + category) == 0), "No transactions")
                                  .when(F.col("payer_account_type") == "SAVINGS", 
                                        F.when(F.col("sum_" + category) < 5000000, "10")
                                        .when((F.col("sum_" + category) >= 5000000) & (F.col("sum_" + category) < 10000000), "9")
                                        .when((F.col("sum_" + category) >= 10000000) & (F.col("sum_" + category) < 15000000), "8")
                                        .when((F.col("sum_" + category) >= 15000000) & (F.col("sum_" + category) < 20000000), "7")
                                        .when((F.col("sum_" + category) >= 20000000) & (F.col("sum_" + category) < 25000000), "6")
                                        .when((F.col("sum_" + category) >= 25000000) & (F.col("sum_" + category) < 30000000), "5")
                                        .when((F.col("sum_" + category) >= 30000000) & (F.col("sum_" + category) < 35000000), "4")
                                        .when((F.col("sum_" + category) >= 35000000) & (F.col("sum_" + category) < 40000000), "3")
                                        .when((F.col("sum_" + category) >= 40000000) & (F.col("sum_" + category) < 45000000), "2")
                                        .otherwise("1"))
                                  .when(F.col("payer_account_type") == "CURRENT", 
                                        F.when(F.col("sum_" + category) < 15000000, "10")
                                        .when((F.col("sum_" + category) >= 15000000) & (F.col("sum_" + category) < 30000000), "9")
                                        .when((F.col("sum_" + category) >= 30000000) & (F.col("sum_" + category) < 45000000), "8")
                                        .when((F.col("sum_" + category) >= 45000000) & (F.col("sum_" + category) < 60000000), "7")
                                        .when((F.col("sum_" + category) >= 60000000) & (F.col("sum_" + category) < 75000000), "6")
                                        .when((F.col("sum_" + category) >= 75000000) & (F.col("sum_" + category) < 90000000), "5")
                                        .when((F.col("sum_" + category) >= 90000000) & (F.col("sum_" + category) < 105000000), "4")
                                        .when((F.col("sum_" + category) >= 105000000) & (F.col("sum_" + category) < 120000000), "3")
                                        .when((F.col("sum_" + category) >= 120000000) & (F.col("sum_" + category) < 135000000), "2")
                                        .otherwise("1"))
                                  .otherwise("Unknown Type"))
    
    return agg_df

df2 = sqlContext.table('loyalty_out_test.prospect_phonepe_leads').filter((F.col('txn_date')=='2022-01-28')
                                                                         &(F.col('txn_hour')==13)
                                                                        )

# Convert the Spark DataFrame to Pandas DataFrame for processing
df = df2.toPandas()

# Handle NaN values by replacing them with empty strings
df.fillna("", inplace=True)

# Transform the features
features = pd.concat([
    pd.DataFrame(tfidf_vectorizer.transform(df[column]).toarray())
    for tfidf_vectorizer, column in zip([tfidf_payer_name, tfidf_payee_name, tfidf_payee_account_type,
                                         tfidf_payer_account_type, tfidf_payer_vpa, tfidf_payee_vpa],
                                        ['payer_name', 'payee_name', 'payee_account_type',
                                         'payer_account_type', 'payer_vpa', 'payee_vpa'])
], axis=1) 

# Predict categories
df['category_level1'] = clf_cat1.predict(features)
df['category_level2'] = clf_cat2.predict(features)

# Convert the Pandas DataFrame back to a Spark DataFrame
result_df = sqlContext.createDataFrame(df)

result_df.show()
endtime=time.time()
print endtime - starttime


# Save the trained models to the current directory
model_cat1_path = "model_cat1"
model_cat2_path = "model_cat2"

model_cat1.write().overwrite().save(model_cat1_path)
model_cat2.write().overwrite().save(model_cat2_path)
