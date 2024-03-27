# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # One-Hot Encoding
# MAGIC
# MAGIC このノートブックでは、モデルの追加の特徴量を追加し、カテゴリー型の特徴量をどのように取り扱うのかを議論します。

# COMMAND ----------

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train/Testの分割
# MAGIC
# MAGIC アップルツーアップルで比較できるように、前回のノートブックと同じシードを用いて同じように80/20の分割を行いましょう(クラスター構成を変更していない場合に限ります！)。

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## オプション1: StringIndexer, OneHotEncoder, VectorAssembler
# MAGIC
# MAGIC ここでは、カテゴリー型の変数に対してワンホットエンコーディング(OHE)を行います。使用する最初のアプローチでは、StringIndexer, OneHotEncoder, VectorAssemblerを組み合わせます。
# MAGIC
# MAGIC 最初に、文字列ラベルのカラムを、ラベルインデックスのMLカラムにマッピングするために`StringIndexer`を使用する必要があります　[Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.StringIndexer)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.StringIndexer)。
# MAGIC
# MAGIC そして、StringIndexerのアウトプットに`OneHotEncoder`を適用します [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.OneHotEncoder)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.OneHotEncoder)。

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

categoricalCols = [field for (field, dataType) in trainDF.dtypes 
                   if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]
oheOutputCols = [x + "OHE" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, 
                              outputCols=indexOutputCols, 
                              handleInvalid="skip")
oheEncoder = OneHotEncoder(inputCols=indexOutputCols, 
                           outputCols=oheOutputCols)

# COMMAND ----------

# MAGIC %md
# MAGIC これで、数値特徴量とOHEされたカテゴリー型の特徴量を組み合わせることができます。

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numericCols = [field for (field, dataType) in trainDF.dtypes 
               if ((dataType == "double") & (field != "price"))]
assemblerInputs = oheOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, 
                               outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## オプション2: RFormula
# MAGIC
# MAGIC StringIndexerとOneHotEncoderに手動でどのカラムがカテゴリー型かを指定するのではなく、RFormulaは自動でこれらのことを行います [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.RFormula)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.RFormula)。
# MAGIC
# MAGIC RFormulaを用いることで、文字列型のカラムがある場合、カテゴリー型の特徴量としてそれらを取り扱い、文字列のインデックスの作成とワンホットエンコーディングを行います。それ以外の場合は何もしません。そして、ワンホットエンコーディングされた特徴量と数値特徴量を、`features`と呼ばれる単一のベクトルにまとめます。

# COMMAND ----------

from pyspark.ml.feature import RFormula

rFormula = RFormula(formula="price ~ .", featuresCol="features", labelCol="price", handleInvalid="skip")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 線形回帰
# MAGIC
# MAGIC すべての特徴量に対応したので、線形回帰モデルを構築しましょう。

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="price", featuresCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## パイプライン
# MAGIC
# MAGIC これらすべてのステージをパイプラインにまとめます。`Pipeline`はすべてのトランスフォーマーとエスティメーターを整理するための手段です [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.Pipeline)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.Pipeline)。
# MAGIC
# MAGIC オプション1(StringIndexer, OneHotEncoderEstimator, VectorAssembler)とオプション2(RFormula)で同じ結果になることを検証します。

# COMMAND ----------

# オプション 1: StringIndexer + OHE + VectorAssembler
from pyspark.ml import Pipeline

stages = [stringIndexer, oheEncoder, vecAssembler, lr]
pipeline = Pipeline(stages=stages)

pipelineModel = pipeline.fit(trainDF)
predDF = pipelineModel.transform(testDF)
predDF.select("features", "price", "prediction").show(5)

# COMMAND ----------

# オプション 2: RFormula
from pyspark.ml import Pipeline

pipeline = Pipeline(stages = [rFormula, lr])

pipelineModel = pipeline.fit(trainDF)
predDF = pipelineModel.transform(testDF)
predDF.select("features", "price", "prediction").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの評価: RMSE

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")

rmse = round(regressionEvaluator.evaluate(predDF), 2)
print(f"RMSE is {rmse}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## R2
# MAGIC
# MAGIC ![](https://files.training.databricks.com/images/r2d2.jpg) R2はどうなっているでしょうか？

# COMMAND ----------

r2 = round(regressionEvaluator.setMetricName("r2").evaluate(predDF), 2)
print(f"R2 is {r2}")

# COMMAND ----------

pipelinePath = "/tmp/sf-airbnb/lr-pipeline-model"
pipelineModel.write().overwrite().save(pipelinePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルのロード
# MAGIC
# MAGIC モデルでロードする際、ロードし直すモデルのタイプを知っている必要があります(線形回帰、あるいはロジスティック回帰モデル？)。
# MAGIC
# MAGIC このため、常に汎用的なPipelineModelをロードできるように、お使いのトランスフォーマーとエスティメーターをパイプラインに組み込むことをお勧めします。

# COMMAND ----------

from pyspark.ml import PipelineModel

savedPipelineModel = PipelineModel.load(pipelinePath)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 分散環境
# MAGIC
# MAGIC 分散環境において線形回帰がどのように実装されているのか、ボトルネックが何かを学ぶことに興味があるのであれば、以下の講義スライドをチェックしてください:
# MAGIC * [distributed-linear-regression-1](https://files.training.databricks.com/static/docs/distributed-linear-regression-1.pdf)
# MAGIC * [distributed-linear-regression-2](https://files.training.databricks.com/static/docs/distributed-linear-regression-2.pdf)
# MAGIC
