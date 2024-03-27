# Databricks notebook source
# MAGIC %md
# MAGIC # MLflowによるモデルのトラッキング
# MAGIC
# MAGIC MLflowはDatabricks Runtime for MLにプレインストールされています。MLランタイムを使っていない場合には、MLflowをインストールする必要があります。

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

filePath = "dbfs:/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)

categoricalCols = [field for (field, dataType) in trainDF.dtypes 
                   if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]
stringIndexer = StringIndexer(inputCols=categoricalCols, 
                              outputCols=indexOutputCols, 
                              handleInvalid="skip")

numericCols = [field for (field, dataType) in trainDF.dtypes 
               if ((dataType == "double") & (field != "price"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, 
                               outputCol="features")

rf = RandomForestRegressor(labelCol="price", maxBins=40, maxDepth=5, 
                           numTrees=100, seed=42)

pipeline = Pipeline(stages=[stringIndexer, vecAssembler, rf])

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowによるモデルのトラッキング

# COMMAND ----------

import mlflow
import mlflow.spark
import pandas as pd

with mlflow.start_run(run_name="random-forest") as run:
  # パラメーターの記録: Num Trees と Max Depth
  mlflow.log_param("num_trees", rf.getNumTrees())
  mlflow.log_param("max_depth", rf.getMaxDepth())
 
  # モデルの記録
  pipelineModel = pipeline.fit(trainDF)
  mlflow.spark.log_model(pipelineModel, "model")

  # メトリクスの記録: RMSE と R2
  predDF = pipelineModel.transform(testDF)
  regressionEvaluator = RegressionEvaluator(predictionCol="prediction", 
                                            labelCol="price")
  rmse = regressionEvaluator.setMetricName("rmse").evaluate(predDF)
  r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
  mlflow.log_metrics({"rmse": rmse, "r2": r2})

  # アーティファクトの記録: 特徴量の重要度スコア
  rfModel = pipelineModel.stages[-1]
  pandasDF = (pd.DataFrame(list(zip(vecAssembler.getInputCols(), 
                                    rfModel.featureImportances)), 
                          columns=["feature", "importance"])
              .sort_values(by="importance", ascending=False))
  # 最初にローカルファイルシステムに書き出し、MLflowにファイルの場所を通知
  pandasDF.to_csv("/tmp/feature-importance.csv", index=False)
  mlflow.log_artifact("/tmp/feature-importance.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflowClient
# MAGIC
# MAGIC [MLflowClient](https://mlflow.org/docs/latest/python_api/mlflow.client.html)は、MLflowエクスペリメント、ラン、モデルバージョン、登録モデルに対するPythonのCRUDインタフェースを提供します。

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()
runs = client.search_runs(run.info.experiment_id,
                          order_by=["attributes.start_time desc"], 
                          max_results=1)
run_id = runs[0].info.run_id
runs[0].data.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC ## バッチ予測の生成
# MAGIC
# MAGIC バッチ予測を生成するためにモデルをロードしなおしましょう。

# COMMAND ----------

# MLflowで保存したモデルのロード
pipelineModel = mlflow.spark.load_model(f"runs:/{run_id}/model")

# 予測の生成
inputPath = "dbfs:/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
inputDF = spark.read.parquet(inputPath)
predDF = pipelineModel.transform(inputDF)

# COMMAND ----------

display(predDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## ストリーミング予測の生成
# MAGIC
# MAGIC ストリーミングの予測結果を生成するために同じことを行えます。

# COMMAND ----------

# MLflowで保存したモデルのロード
pipelineModel = mlflow.spark.load_model(f"runs:/{run_id}/model")

# シミュレーションされたストリーミングデータをセットアップ
repartitionedPath = "dbfs:/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean-100p.parquet"
schema = spark.read.parquet(repartitionedPath).schema

streamingData = (spark
                 .readStream
                 .schema(schema) # このようにスキーマを設定可能
                 .option("maxFilesPerTrigger", 1)
                 .parquet(repartitionedPath))

# 予測の生成
streamPred = pipelineModel.transform(streamingData)

# COMMAND ----------

# ストリーミング予測結果の表示
display(streamPred)

# COMMAND ----------


