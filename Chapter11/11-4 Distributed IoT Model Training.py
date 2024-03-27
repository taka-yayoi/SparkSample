# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC ## Pandas Function APIによるIoTモデルの分散トレーニング
# MAGIC
# MAGIC このノートブックでは、pandas function APIでどのようにシングルノードの機械学習ソリューションをスケールさせるのかをデモンストレーションします。

# COMMAND ----------

# MAGIC %md
# MAGIC 以下のダミーデータを作成します:
# MAGIC
# MAGIC - `device_id`: 10台の異なるデバイス
# MAGIC - `record_id`: 10kのユニークなレコード
# MAGIC - `feature_1`: モデルトレーニングの特徴量
# MAGIC - `feature_2`: モデルトレーニングの特徴量
# MAGIC - `feature_3`: モデルトレーニングの特徴量
# MAGIC - `label`: 予測しようとする変数

# COMMAND ----------

import pyspark.sql.functions as f

df = (spark.range(1000*1000)
  .select(f.col("id").alias("record_id"), (f.col("id")%10).alias("device_id"))
  .withColumn("feature_1", f.rand() * 1)
  .withColumn("feature_2", f.rand() * 2)
  .withColumn("feature_3", f.rand() * 3)
  .withColumn("label", (f.col("feature_1") + f.col("feature_2") + f.col("feature_3")) + f.rand())
)

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC 戻り値のスキーマを定義します。

# COMMAND ----------

import pyspark.sql.types as t

trainReturnSchema = t.StructType([
  t.StructField("device_id", t.IntegerType()), # ユニークなデバイスID
  t.StructField("n_used", t.IntegerType()),    # トレーニングで使うレコード数
  t.StructField("model_path", t.StringType()), # 特定のデバイスのモデルへのパス
  t.StructField("mse", t.FloatType())          # モデルパフォーマンスのメトリック
])

# COMMAND ----------

# MAGIC %md
# MAGIC 特定のデバイスのすべてのデータを受け取り、モデルをトレーニングし、ネストされたランとして保存し、上記のスキーマを持つデータフレームを返却する関数を定義します。
# MAGIC
# MAGIC これらのすべてのモデルをトラッキングするためにMLflowを活用します。

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
  """
  グルーピングされたインスタンスでsklearnモデルをトレーニング
  """
  # メタデータの取得
  device_id = df_pandas["device_id"].iloc[0]
  n_used = df_pandas.shape[0]
  run_id = df_pandas["run_id"].iloc[0] # ネストされたランを行うためのランIDの取得
  
  # モデルのトレーニング
  X = df_pandas[["feature_1", "feature_2", "feature_3"]]
  y = df_pandas["label"]
  rf = RandomForestRegressor()
  rf.fit(X, y)

  # モデルの評価
  predictions = rf.predict(X)
  mse = mean_squared_error(y, predictions) # トレーニング/テストスプリットを追加できることに注意してください
 
  # トップレベルトレーニングの再開
  with mlflow.start_run(run_id=run_id):
    # 特定のデバイスに対するネストされたランの作成
    with mlflow.start_run(run_name=str(device_id), nested=True) as run:
      mlflow.sklearn.log_model(rf, str(device_id))
      mlflow.log_metric("mse", mse)
      
      artifact_uri = f"runs:/{run.info.run_id}/{device_id}"
      # 上記のスキーマにマッチする戻り値のpandasデータフレームを作成
      returnDF = pd.DataFrame([[device_id, n_used, artifact_uri, mse]], 
        columns=["device_id", "n_used", "model_path", "mse"])

  return returnDF 

# COMMAND ----------

# MAGIC %md
# MAGIC グルーピングされたデータにapplyInPandasを適用します。

# COMMAND ----------

with mlflow.start_run(run_name="Training session for all devices") as run:
  run_id = run.info.run_uuid
  
  modelDirectoriesDF = (df
    .withColumn("run_id", f.lit(run_id)) # run_idを追加
    .groupby("device_id")
    .applyInPandas(train_model, schema=trainReturnSchema)
  )
  
combinedDF = (df
  .join(modelDirectoriesDF, on="device_id", how="left")
)

display(combinedDF)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルを適用するための関数を定義します。*これは、デバイスごとにDBFSから一度の読み取りのみを必要とします。*

# COMMAND ----------

applyReturnSchema = t.StructType([
  t.StructField("record_id", t.IntegerType()),
  t.StructField("prediction", t.FloatType())
])

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
  """
  pandasデータフレームとして表現される特定のデバイスのデータにモデルを適用
  """
  model_path = df_pandas["model_path"].iloc[0]
  
  input_columns = ["feature_1", "feature_2", "feature_3"]
  X = df_pandas[input_columns]
  
  model = mlflow.sklearn.load_model(model_path)
  prediction = model.predict(X)
  
  returnDF = pd.DataFrame({
    "record_id": df_pandas["record_id"],
    "prediction": prediction
  })
  return returnDF

predictionDF = combinedDF.groupby("device_id").applyInPandas(apply_model, schema=applyReturnSchema)
display(predictionDF)
