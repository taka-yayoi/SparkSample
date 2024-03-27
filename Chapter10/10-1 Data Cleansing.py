# Databricks notebook source
# MAGIC
# MAGIC %md-sandbox
# MAGIC # Data Cleansing with Airbnb
# MAGIC
# MAGIC いくつかの探索的データ分析やクレンジングからスタートします。[Inside Airbnb](http://insideairbnb.com/get-the-data.html)のSF Airbnbレンタルデータセットを使用します。
# MAGIC
# MAGIC <img src="http://insideairbnb.com/images/insideairbnb_graphic_site_1200px.png" style="width:800px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC SF Airbnbデータセットをロードしましょう(オプションの動作を確認したい場合にはコメントアウトしてください)。

# COMMAND ----------

filePath = "/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb.csv"

rawDF = spark.read.csv(filePath, header="true", inferSchema="true", multiLine="true", escape='"')

display(rawDF)

# COMMAND ----------

rawDF.columns

# COMMAND ----------

# MAGIC %md
# MAGIC シンプルにするために、このデータセットの特定のカラムのみを保持します。あとで、特徴量選択について言及します。

# COMMAND ----------

columnsToKeep = [
  "host_is_superhost",
  "cancellation_policy",
  "instant_bookable",
  "host_total_listings_count",
  "neighbourhood_cleansed",
  "latitude",
  "longitude",
  "property_type",
  "room_type",
  "accommodates",
  "bathrooms",
  "bedrooms",
  "beds",
  "bed_type",
  "minimum_nights",
  "number_of_reviews",
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value",
  "price"]

baseDF = rawDF.select(columnsToKeep)
baseDF.cache().count()
display(baseDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## データ型の修正
# MAGIC
# MAGIC 上のスキーマを確認してみます。`price`フィールドが文字列として判定されていることがわかります。我々のタスクでは、これを数値型のフィールド(double型)にする必要があります。
# MAGIC
# MAGIC 修正しましょう。

# COMMAND ----------

from pyspark.sql.functions import col, translate

fixedPriceDF = baseDF.withColumn("price", translate(col("price"), "$,", "").cast("double"))

display(fixedPriceDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## サマリー統計情報
# MAGIC
# MAGIC 2つのオプションがあります:
# MAGIC * describe
# MAGIC * summary (describe + IQR)

# COMMAND ----------

display(fixedPriceDF.describe())

# COMMAND ----------

display(fixedPriceDF.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Null値
# MAGIC
# MAGIC Null値を取り扱うための数多くの手法があります。時には、nullは実際には予測しようとする事柄のキーのインジケータとなることがあります(例: フォームの特定の割合を記入しない場合、承認される確率が減少する)。
# MAGIC
# MAGIC nullを取り扱う方法として:
# MAGIC * nullを含むすべてのレコードを削除
# MAGIC * 数値型:
# MAGIC   * mean/median/zeroなどで補完
# MAGIC * カテゴリー型:
# MAGIC   * モードで置換
# MAGIC   * nullに対する特殊なカテゴリーを作成
# MAGIC * 欠損地を挿入するために設計されたALSのようなテクニックを使う
# MAGIC
# MAGIC **カテゴリー特徴量/数値特徴量に何かしらの補完テクニックを用いる場合、当該フィールドが保管されたことを示す追加フィールドを含めるべきです(なぜこれが必要なのかを考えてみてください)**

# COMMAND ----------

# MAGIC %md
# MAGIC カテゴリー特徴量`host_is_superhost`にいくつかnullが含まれています。これらのカラムのいずれかがnullである行を除外しましょう。
# MAGIC
# MAGIC SparkMLのImputer(この後カバーします)は、カテゴリー特徴量の補完をサポートしていませんので、この時点ではこれがもっともシンプルなアプローチとなります。

# COMMAND ----------

noNullsDF = fixedPriceDF.na.drop(subset=["host_is_superhost"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 補完: Doubleへのキャスト
# MAGIC
# MAGIC SparkMLの`Imputer`は、すべてのフィールドがdouble型である必要があります [Python](https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.feature.Imputer)/[Scala](https://spark.apache.org/docs/latest/api/scala/#org.apache.spark.ml.feature.Imputer)。すべてのintegerフィールドをdoubleにキャストしましょう。

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

integerColumns = [x.name for x in baseDF.schema.fields if x.dataType == IntegerType()]
doublesDF = noNullsDF

for c in integerColumns:
  doublesDF = doublesDF.withColumn(c, col(c).cast("double"))

columns = "\n - ".join(integerColumns)
print(f"Columns converted from Integer to Double:\n - {columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC 値を補完したかどうかを示すダミー変数を追加します。

# COMMAND ----------

from pyspark.sql.functions import when

imputeCols = [
  "bedrooms",
  "bathrooms",
  "beds", 
  "review_scores_rating",
  "review_scores_accuracy",
  "review_scores_cleanliness",
  "review_scores_checkin",
  "review_scores_communication",
  "review_scores_location",
  "review_scores_value"
]

for c in imputeCols:
  doublesDF = doublesDF.withColumn(c + "_na", when(col(c).isNull(), 1.0).otherwise(0.0))

# COMMAND ----------

display(doublesDF.describe())

# COMMAND ----------

from pyspark.ml.feature import Imputer

imputer = Imputer(strategy="median", inputCols=imputeCols, outputCols=imputeCols)

imputedDF = imputer.fit(doublesDF).transform(doublesDF)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 外れ値を排除
# MAGIC
# MAGIC `price`カラムの *min* と *max* の値を見てみましょう:

# COMMAND ----------

display(imputedDF.select("price").describe())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC いくつか非常に高価な物件があります。しかし、これらに対して何をすべきかを決めるのはデータサイエンティストの仕事です。しかし、「無料」のAirbnbはフィルタリングします。
# MAGIC
# MAGIC *price*がゼロである物件がいくつあるのかを見てみましょう。

# COMMAND ----------

imputedDF.filter(col("price") == 0).count()

# COMMAND ----------

# MAGIC %md
# MAGIC 厳密に正の*price*を持つ行のみを保持します。

# COMMAND ----------

posPricesDF = imputedDF.filter(col("price") > 0)

# COMMAND ----------

# MAGIC %md
# MAGIC *minimum_nights*カラムの *min* と *max* を見てみましょう:

# COMMAND ----------

display(posPricesDF.select("minimum_nights").describe())

# COMMAND ----------

display(posPricesDF
  .groupBy("minimum_nights").count()
  .orderBy(col("count").desc(), col("minimum_nights"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC 1年という最小滞在期間は合理的な制限のように見えます。*minimum_nights*が365を上回るレコードを除外しましょう:

# COMMAND ----------

cleanDF = posPricesDF.filter(col("minimum_nights") <= 365)

display(cleanDF)

# COMMAND ----------

# MAGIC %md
# MAGIC OK、データが綺麗になりました。これを用いてモデルの構築をスタートできるように、このデータフレームをファイルに保存しましょう。

# COMMAND ----------

outputPath = "/tmp/sf-airbnb/sf-airbnb-clean.parquet"

cleanDF.write.mode("overwrite").parquet(outputPath)

