# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # ハイパーパラメータチューニング
# MAGIC
# MAGIC ベストなハイパーパラメータを見つけ出すために、ランダムフォレストに対してハイパーパラメータチューニングを実施しましょう！

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
(trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

numericCols = [field for (field, dataType) in trainDF.dtypes if ((dataType == "double") & (field != "price"))]
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## ランダムフォレスト

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline

rf = RandomForestRegressor(labelCol="price", maxBins=40, seed=42)
pipeline = Pipeline(stages = [stringIndexer, vecAssembler, rf])

# COMMAND ----------

# MAGIC %md
# MAGIC ## グリッドサーチ
# MAGIC
# MAGIC チューニング可能なハイパーパラメータは多数存在し、手動で設定するには長い時間を要します。
# MAGIC
# MAGIC よりシステマティックなアプローチで最適なハイパーパラメータを見つけ出すために、Sparkの`ParamGridBuilder`を活用しましょう [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.ParamGridBuilder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.ParamGridBuilder)。
# MAGIC
# MAGIC テストするハイパーパラメータのグリッドを定義しましょう:
# MAGIC - maxDepth: 決定木の最大の深さ(`2, 4, 6`の値を使用)
# MAGIC - numTrees: 決定木の数(`10, 100`の値を使用)

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [2, 4, 6])
            .addGrid(rf.numTrees, [10, 100])
            .build())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 交差検証
# MAGIC
# MAGIC 最適なmaxDepthを特定するために、3フォールドの交差検証を活用します。
# MAGIC
# MAGIC ![crossValidation](https://files.training.databricks.com/images/301/CrossValidation.png)
# MAGIC
# MAGIC 3フォールドの交差検証によって、データの2/3でトレーニングを行い、(ホールドアウトされた)残りの1/3で評価を行います。このプロセスを3回繰り返すので、それぞれのフォールドは検証用セットとして動作する機会があります。そして、3ラウンドの結果を平均します。

# COMMAND ----------

# MAGIC %md
# MAGIC 以下を伝えるために、`CrossValidator`には`estimator`(パイプライン), `evaluator`, `estimatorParamMaps`を入力します:
# MAGIC - 使用するモデル
# MAGIC - モデルの評価方法
# MAGIC - モデルに設定するハイパーパラメータ
# MAGIC
# MAGIC また、データを分割するフォールドの数を(3)に設定し、データが同じように分割されるようにシードも設定します [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.tuning.CrossValidator)。

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol="price", 
                                predictionCol="prediction", 
                                metricName="rmse")

cv = CrossValidator(estimator=pipeline, 
                    evaluator=evaluator, 
                    estimatorParamMaps=paramGrid, 
                    numFolds=3, 
                    seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC **問題**: この時点でいくつのモデルをトレーニングしていますか？

# COMMAND ----------

cvModel = cv.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parallelismパラメーター
# MAGIC
# MAGIC うーん...実行に長い時間を要しています。これは、並列ではなく直列でモデルがトレーニングされているからです！
# MAGIC
# MAGIC Spark 2.3では、[parallelism](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.tuning.CrossValidator.parallelism)パラメータが導入されました。ドキュメントでは、`並列アルゴリズムを実行する際のスレッド数 (>= 1)`と述べられています。
# MAGIC
# MAGIC この値を4に設定し、トレーニングが早くなるかどうかを見てみましょう。

# COMMAND ----------

cvModel = cv.setParallelism(4).fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC **問題**: うーん...依然として時間がかかっています。交差検証器の中にパイプラインを埋め込むべきか、パイプラインに交差検証器を埋め込むべきでしょうか？
# MAGIC
# MAGIC パイプラインにエスティメーターやトランスフォーマーが含まれるかに依存します。StringIndexer(エスティメーター)のようなものがパイプラインにある場合、交差検証器にパイプライン全体を埋め込むと、毎回再フィットさせなくてはなりません。

# COMMAND ----------

cv = CrossValidator(estimator=rf, 
                    evaluator=evaluator, 
                    estimatorParamMaps=paramGrid, 
                    numFolds=3, 
                    parallelism=4, 
                    seed=42)

pipeline = Pipeline(stages=[stringIndexer, vecAssembler, cv])

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ベストなハイパーパラメータの設定を持つモデルを見てみましょう。

# COMMAND ----------

list(zip(cvModel.getEstimatorParamMaps(), cvModel.avgMetrics))

# COMMAND ----------

# MAGIC %md
# MAGIC テストデータセットでどうなるのかを見てみましょう。

# COMMAND ----------

predDF = pipelineModel.transform(testDF)

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = regressionEvaluator.evaluate(predDF)
r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

