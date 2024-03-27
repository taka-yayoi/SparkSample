# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # Example 4.1
# MAGIC
# MAGIC This notebook shows Example 4.1 from the book showing how to use SQL on a US Flights Dataset dataset.

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

# MAGIC %md
# MAGIC 日付フォーマットを読めるフォーマットに変換するUDFを定義します。
# MAGIC
# MAGIC *注意*: 日付は文字列で年が含まれていないため、SQLの`year()`関数を使うことが困難な場合があります。

# COMMAND ----------

def to_date_format_udf(d_str):
  l = [char for char in d_str]
  return "".join(l[0:2]) + "/" +  "".join(l[2:4]) + " " + " " +"".join(l[4:6]) + ":" + "".join(l[6:])

to_date_format_udf("02190925")

# COMMAND ----------

# MAGIC %md
# MAGIC UDFを登録します。

# COMMAND ----------

spark.udf.register("to_date_format_udf", to_date_format_udf, StringType())

# COMMAND ----------

# MAGIC %md
# MAGIC USを出発するフライトデータを読み込みます。

# COMMAND ----------

df = (spark.read.format("csv")
      .schema("date STRING, delay INT, distance INT, origin STRING, destination STRING")
      .option("header", "true")
      .option("path", "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv")
      .load())

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC UDFをテストします。

# COMMAND ----------

df.selectExpr("to_date_format_udf(date) as data_format").show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC SQLクエリーを実行する一次ビューを作成します。

# COMMAND ----------

df.createOrReplaceTempView("us_delay_flights_tbl")

# COMMAND ----------

# MAGIC %md
# MAGIC クエリーに都合が良いようにテーブルをキャッシュします。

# COMMAND ----------

# MAGIC %sql
# MAGIC CACHE TABLE us_delay_flights_tbl

# COMMAND ----------

# MAGIC %md
# MAGIC 読みやすくなるようにすべての`date`を`date_fm`に変換します。
# MAGIC
# MAGIC 注意: オンザフライで変換するためにUDFを使用しています。

# COMMAND ----------

spark.sql("SELECT *, date, to_date_format_udf(date) AS date_fm FROM us_delay_flights_tbl").show(10, truncate=False)

# COMMAND ----------

spark.sql("SELECT COUNT(*) FROM us_delay_flights_tbl").show() # Keep case consistent for all SQL??

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 1:
# MAGIC
# MAGIC 出発地と到着地との距離が1000より大きい全てのフライトを検索

# COMMAND ----------

spark.sql("SELECT distance, origin, destination FROM us_delay_flights_tbl WHERE distance > 1000 ORDER BY distance DESC").show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Sparkデータフレームで行う同じクエリー

# COMMAND ----------

df.select("distance", "origin", "destination").where(col("distance") > 1000).orderBy(desc("distance")).show(10, truncate=False)

# COMMAND ----------

df.select("distance", "origin", "destination").where("distance > 1000").orderBy("distance", ascending=False).show(10)

# COMMAND ----------

df.select("distance", "origin", "destination").where("distance > 1000").orderBy(desc("distance")).show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 2:
# MAGIC
# MAGIC サンフランシスコとシカゴの間で2時間より長い遅延のあったフライトを検索

# COMMAND ----------

spark.sql("""
SELECT date, delay, origin, destination 
FROM us_delay_flights_tbl 
WHERE delay > 120 AND ORIGIN = 'SFO' AND DESTINATION = 'ORD' 
ORDER by delay DESC
""").show(10, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query 3:
# MAGIC
# MAGIC SQLにおけるより複雑なクエリーとして、目的地に関係なく空港から出発する全てのUSフライトにラベル、_high_, _medium_, _low_, _no delays_をつけましょう。

# COMMAND ----------

spark.sql("""SELECT delay, origin, destination,
              CASE
                  WHEN delay > 360 THEN 'Very Long Delays'
                  WHEN delay > 120 AND delay < 360 THEN  'Long Delays '
                  WHEN delay > 60 AND delay < 120 THEN  'Short Delays'
                  WHEN delay > 0 and delay < 60  THEN   'Tolerable Delays'
                  WHEN delay = 0 THEN 'No Delays'
                  ELSE 'No Delays'
               END AS Flight_Delays
               FROM us_delay_flights_tbl
               ORDER BY origin, delay DESC""").show(10, truncate=False)

# COMMAND ----------

df = spark.sql("""SELECT delay, origin, destination,
              CASE
                  WHEN delay > 360 THEN 'Very Long Delays'
                  WHEN delay > 120 AND delay < 360 THEN  'Long Delays '
                  WHEN delay > 60 AND delay < 120 THEN  'Short Delays'
                  WHEN delay > 0 and delay < 60  THEN   'Tolerable Delays'
                  WHEN delay = 0 THEN 'No Delays'
                  ELSE 'No Delays'
               END AS Flight_Delays
               FROM us_delay_flights_tbl
               ORDER BY origin, delay DESC""")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Some Side Queries

# COMMAND ----------

df1 =  spark.sql("SELECT date, delay, origin, destination FROM us_delay_flights_tbl WHERE origin = 'SFO'")

# COMMAND ----------

df1.createOrReplaceGlobalTempView("us_origin_airport_SFO_tmp_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM global_temp.us_origin_airport_SFO_tmp_view

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS global_temp.us_origin_airport_JFK_tmp_view

# COMMAND ----------

df2 = spark.sql("SELECT date, delay, origin, destination from us_delay_flights_tbl WHERE origin = 'JFK'")

# COMMAND ----------

df2.createOrReplaceTempView("us_origin_airport_JFK_tmp_view")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM us_origin_airport_JFK_tmp_view

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS us_origin_airport_JFK_tmp_view

# COMMAND ----------

spark.catalog.listTables(dbName="global_temp")

# COMMAND ----------


