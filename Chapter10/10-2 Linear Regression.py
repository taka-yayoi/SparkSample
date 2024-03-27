# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # 回帰問題: レンタル価格の予測
# MAGIC
# MAGIC このノートブックでは、サンフランシスコのAirbnbのレンタル価格を予測するために以前のlabでクレンジングしたデータセットを使用します。

# COMMAND ----------

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet"
airbnbDF = spark.read.parquet(filePath)
display(airbnbDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## トレーニング/テストの分割
# MAGIC
# MAGIC MLモデルを構築する際、テストデータを参照するべきではありません(なぜでしょうか？)。
# MAGIC
# MAGIC トレーニングデータセットに80%をキープし、20%をテストデータセットとして取っておきます。`randomSplit`メソッド[Python](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame.randomSplit)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.sql.Dataset)を活用します。
# MAGIC
# MAGIC **問題**: なぜ、シードを設定する必要があるのでしょうか？

# COMMAND ----------

trainDF, testDF = airbnbDF.randomSplit([.8, .2], seed=42)
print(f"There are {trainDF.cache().count()} rows in the training set, and {testDF.cache().count()} in the test set")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **問題**: クラスター設定を変更するとどうなるのでしょうか？
# MAGIC
# MAGIC これを試すには、1台のみのワーカーを持つクラスターと2台のワーカーを持つ別のクラスターを起動します。
# MAGIC
# MAGIC **注意**
# MAGIC
# MAGIC このデータは非常に小さいもの(1パーティション)であり、違いを確認するには`databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean-100p.parquet`のように、大規模なデータセット(2+のパーティションなど)でテストする必要があるかもしれません。しかし、以下のコードでは、異なるクラスター設定で、どのように異なってパーティショニングされるのかをシミュレートするために、シンプルにrepartitionを行い、我々のトレーニングセットで同じ数のデータポイントを取得できるかどうかを確認しています。

# COMMAND ----------

(trainRepartitionDF, testRepartitionDF) = (airbnbDF
                                           .repartition(24)
                                           .randomSplit([.8, .2], seed=42))

print(trainRepartitionDF.count())

# COMMAND ----------

# MAGIC %md
# MAGIC 80/20でtrain/testを分割する際、これは80/20分割の「近似」となります。正確な80/20の分割ではなく、我々のデータのパーティショニングが変化すると、train/testで異なる数のデータポイントを取得するだけではなく、データポイント自体も異なるものになります。
# MAGIC
# MAGIC おすすめは、再現性の問題に遭遇しないように、一度データを分割したらそれぞれのtrain/testフォルダに書き出すというものです。

# COMMAND ----------

# MAGIC %md
# MAGIC `bedrooms`の数を指定したら`price`を予測する非常にシンプルな線形回帰モデルを構築します。
# MAGIC
# MAGIC **問題**: 線形回帰モデルにおける仮定にはどのようなものがありますか？

# COMMAND ----------

display(trainDF.select("price", "bedrooms").summary())

# COMMAND ----------

# MAGIC %md
# MAGIC 価格についてはデータセットでいくつかの外れ値があります(一晩$10,000??)。モデルを構築する際にはこのことを念頭に置いてください :)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Vector Assembler
# MAGIC
# MAGIC 線形回帰では、入力としてVector型のカラムを期待します。
# MAGIC
# MAGIC `VectorAssembler` [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.VectorAssembler)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.VectorAssembler)を用いて、簡単に`bedrooms`の値を単一のベクトルに変換できます。VectorAssemblerは**transformer**の一例です。トランスフォーマーはデータフレームを受け取り、1つ以上のカラムが追加された新規のデータフレームを返却します。これらはデータから学習は行いませんが、ルールベースの変換処理を適用します。

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["bedrooms"], outputCol="features")

vecTrainDF = vecAssembler.transform(trainDF)

vecTrainDF.select("bedrooms", "features", "price").show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 線形回帰
# MAGIC
# MAGIC データの準備ができたので、最初のモデルを構築するために`LinearRegression`エスティメーター [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.regression.LinearRegression)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.regression.LinearRegression) を活用できます。エスティメーターは、入力としてデータフレームを受け取ってモデルを返却し、モデルは`.fit()`メソッドを持ちます。

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="price")
lrModel = lr.fit(vecTrainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの調査

# COMMAND ----------

m = round(lrModel.coefficients[0], 2)
b = round(lrModel.intercept, 2)

print(f"The formula for the linear regression line is price = {m}*bedrooms + {b}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## パイプライン

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[vecAssembler, lr])
pipelineModel = pipeline.fit(trainDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## テストセットへの適用

# COMMAND ----------

predDF = pipelineModel.transform(testDF)

predDF.select("bedrooms", "features", "price", "prediction").show(10)

# COMMAND ----------


