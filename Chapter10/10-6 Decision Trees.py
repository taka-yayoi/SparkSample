# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # 決定木
# MAGIC
# MAGIC 以前のノートブックではパラメトリックモデルである線形回帰を取り扱いました。線形回帰モデルさらにハイパーパラメーターチューニングを行うこともできますが、ツリーベースの手法をトライし、パフォーマンスの改善を見ていきます。

# COMMAND ----------

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ## カテゴリ型の特徴量をどのように取り扱うべきか？
# MAGIC
# MAGIC 以前のノートブックでStringIndexer/OneHotEncoderEstimator/VectorAssemblerやRFormulaを活用できることを見てきました。
# MAGIC
# MAGIC **しかし、決定木、特にランダムフォレストでは、変数をOHEすべきではありません。**
# MAGIC
# MAGIC なぜでしょうか？それは、スプリットの作成方法が違い(ツリーを可視化するとわかります)、特徴量の重要度スコアが正しくありません。
# MAGIC
# MAGIC (すぐに説明する)ランダムフォレストでは、結果は劇的に変化します。このため、RFormulaを使うのではなく、シンプルにStringIndexer/VectorAssemblerを使います。

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

categoricalCols = [field for (field, dataType) in trainDF.dtypes if dataType == "string"]
indexOutputCols = [x + "Index" for x in categoricalCols]

stringIndexer = StringIndexer(inputCols=categoricalCols, outputCols=indexOutputCols, handleInvalid="skip")

# COMMAND ----------

# MAGIC %md
# MAGIC ## VectorAssembler
# MAGIC
# MAGIC すべてのカテゴリ型と数値型の入力のすべてを組み合わせるためにVectorAssemblerを使いましょう [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler)。

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# 数値列のみにフィルタリング(そして、ラベルのpriceを除外)
numericCols = [field for (field, dataType) in trainDF.dtypes 
               if ((dataType == "double") & (field != "price"))]
# 上で定義したStringIndexerの出力と数値列を結合
assemblerInputs = indexOutputCols + numericCols
vecAssembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 決定木
# MAGIC
# MAGIC デフォルトのハイパーパラメータで`DecisionTreeRegressor`を構築しましょう [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.DecisionTreeRegressor)。

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(labelCol="price")

# COMMAND ----------

# MAGIC %md
# MAGIC ## パイプラインのフィッティング

# COMMAND ----------

from pyspark.ml import Pipeline

# ステージをパイプラインに結合
stages = [stringIndexer, vecAssembler, dt]
pipeline = Pipeline(stages=stages)

# フィッティングを行うにはコメントを解除
pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## maxBins
# MAGIC
# MAGIC パラメーターの[maxBins](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.DecisionTreeRegressor.maxBins)は何でしょうか？(Sparkが使う)分散決定技のPLANET実装を見て、Matei Zahariaらによる[Yggdrasil](https://cs.stanford.edu/~matei/papers/2016/nips_yggdrasil.pdf)という論文と比較してみましょう。これは、`maxBins`パラメータを説明する助けとなります。

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/DistDecisionTrees.png" height=500px>

# COMMAND ----------

# MAGIC %md
# MAGIC Sparkでは、データは行ごとにパーティションが作成されます。このため、スプリットを行う必要がある際には、それぞれのワーカーでそれぞれの分割ポイントのすべての特徴量のサマリー統計情報を計算する必要があります。そして、これらのサマリー統計情報は作成されるそれぞれのスプリットに対して(ツリーのreduceを通じて)集約される必要があります。
# MAGIC
# MAGIC 考えてみましょう: ワーカー1では値`32`があり、他のワーカーではその値ではない場合どうなるでしょうか。あるスプリットがどのくらい良いものであるのかをどのように伝えるのでしょうか？このため、Sparkでは連続的な変数をバケットに離散化するためのmaxBinsパラメーターがありますが、バケットの数はカテゴリ型変数の数と同じくらい大きなものである必要があります。

# COMMAND ----------

# MAGIC %md
# MAGIC 先に進めて、maxBinsを`40`に増やします。

# COMMAND ----------

dt.setMaxBins(40)

# COMMAND ----------

# MAGIC %md
# MAGIC テイク２です。

# COMMAND ----------

pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 決定木の可視化

# COMMAND ----------

dtModel = pipelineModel.stages[-1]
print(dtModel.toDebugString)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 特徴量の重要度
# MAGIC
# MAGIC 先に進めて、フィッティングした決定木モデルを取得し、特徴量の重要度スコアを見てみましょう。

# COMMAND ----------

dtModel = pipelineModel.stages[-1]
dtModel.featureImportances

# COMMAND ----------

# MAGIC %md
# MAGIC ## 特徴量重要度の解釈
# MAGIC
# MAGIC うーん、特徴量4や11が何であるのかを理解するのは困難です。特徴量重要度のスコアが「小さなデータ」であれば、オリジナルのカラム名に復旧するためにPandasを使いましょう。

# COMMAND ----------

import pandas as pd
dtModel = pipelineModel.stages[-1]
featureImp = pd.DataFrame(
  list(zip(vecAssembler.getInputCols(), dtModel.featureImportances)),
  columns=["feature", "importance"])
featureImp.sort_values(by="importance", ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## テストセットにモデルを適用

# COMMAND ----------

predDF = pipelineModel.transform(testDF)

display(predDF.select("features", "price", "prediction").orderBy("price", ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 落とし穴
# MAGIC
# MAGIC 規模の大きいAirbnb賃貸情報の場合はどうしましょうか？例えば20のベッドルームと20のバスルームです。決定木はどのように予測を行うのでしょうか？
# MAGIC
# MAGIC 決定木はトレーニングしたものより大きな値を予測できないことがわかります。ここでのトレーニングセットの最大値は$10,000なので、それよりも大きな値を予測することはできません。

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regressionEvaluator = RegressionEvaluator(predictionCol="prediction", 
                                          labelCol="price", 
                                          metricName="rmse")

rmse = regressionEvaluator.evaluate(predDF)
r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## なんてこった！
# MAGIC
# MAGIC このモデルは線形回帰モデルよりも精度が悪いです。
# MAGIC
# MAGIC 次のいくつかのノートブックでは、単体の決定木のパフォーマンスよりも改善するために、ハイパーパラメーターチューニングやアンサンブルモデルを見ていきましょう。
