# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # mapInPandasによる分散推論

# COMMAND ----------

# MAGIC %md
# MAGIC sklearnモデルをトレーニングして、MLflowで記録します。

# COMMAND ----------

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

with mlflow.start_run(run_name="sklearn-rf-model") as run:
  # データのインポート
  spark_df = spark.read.csv("dbfs:/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-numeric.csv", header=True, inferSchema=True).drop("zipcode")
  df = spark_df.toPandas()
  X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

  # モデルの作成、トレーニング、予測の生成
  rf = RandomForestRegressor(n_estimators=100, max_depth=10)
  rf.fit(X_train, y_train)
  predictions = rf.predict(X_test)

  # モデルの記録
  mlflow.sklearn.log_model(rf, "random-forest-model")

  # パラメーターの記録
  mlflow.log_param("n_estimators", 100)
  mlflow.log_param("max_depth", 10)

  # メトリクスの記録
  mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
  mlflow.log_metric("mae", mean_absolute_error(y_test, predictions))  
  mlflow.log_metric("r2", r2_score(y_test, predictions))  

# COMMAND ----------

# MAGIC %md
# MAGIC Sparkデータフレームを作成し、Sparkデータフレームにモデルを適用します。

# COMMAND ----------

sparkDF = spark.createDataFrame(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC mapInPandasを用いて並列にモデルを適用します。

# COMMAND ----------

def predict(iterator):
  model_path = f"runs:/{run.info.run_uuid}/random-forest-model" # モデルのロード
  model = mlflow.sklearn.load_model(model_path)
  for features in iterator:
    yield pd.concat([features, pd.Series(model.predict(features), name="prediction")], axis=1)
    
display(sparkDF.mapInPandas(predict, """`host_total_listings_count` DOUBLE,`neighbourhood_cleansed` BIGINT,`latitude` DOUBLE,`longitude` DOUBLE,`property_type` BIGINT,`room_type` BIGINT,`accommodates` DOUBLE,`bathrooms` DOUBLE,`bedrooms` DOUBLE,`beds` DOUBLE,`bed_type` BIGINT,`minimum_nights` DOUBLE,`number_of_reviews` DOUBLE,`review_scores_rating` DOUBLE,`review_scores_accuracy` DOUBLE,`review_scores_cleanliness` DOUBLE,`review_scores_checkin` DOUBLE,`review_scores_communication` DOUBLE,`review_scores_location` DOUBLE,`review_scores_value` DOUBLE, prediction double"""))
