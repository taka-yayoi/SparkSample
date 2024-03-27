# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # Example 3.7

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import col, expr, when, concat, lit

# データのスキーマを定義
""""
schema = (StructType([
   StructField("Id", IntegerType(), False),
   StructField("First", StringType(), False),
   StructField("Last", StringType(), False),
   StructField("Url", StringType(), False),
   StructField("Published", StringType(), False),
   StructField("Hits", IntegerType(), False),
   StructField("Campaigns", ArrayType(StringType()), False)]))
   """

# DDL(Data Definition Language)によるスキーマ定義
ddl_schema = "`Id` INT,`First` STRING,`Last` STRING,`Url` STRING,`Published` STRING,`Hits` INT,`Campaigns` ARRAY<STRING>"

# データの作成
data = [[1, "Jules", "Damji", "https://tinyurl.1", "1/4/2016", 4535, ["twitter", "LinkedIn"]],
       [2, "Brooke","Wenig","https://tinyurl.2", "5/5/2018", 8908, ["twitter", "LinkedIn"]],
       [3, "Denny", "Lee", "https://tinyurl.3","6/7/2019",7659, ["web", "twitter", "FB", "LinkedIn"]],
       [4, "Tathagata", "Das","https://tinyurl.4", "5/12/2018", 10568, ["twitter", "FB"]],
       [5, "Matei","Zaharia", "https://tinyurl.5", "5/14/2014", 40578, ["web", "twitter", "FB", "LinkedIn"]],
       [6, "Reynold", "Xin", "https://tinyurl.6", "3/2/2015", 25568, ["twitter", "LinkedIn"]]
      ]

# COMMAND ----------

# 上で定義したDDLスキーマを用いてデータフレームを作成
blogs_df = spark.createDataFrame(data, ddl_schema)
# データフレームの表示。上のテーブルを反映しているはずです
blogs_df.show()

# COMMAND ----------

# データフレームから一時ビューの作成
blogs_df.createOrReplaceTempView("blogs")

# COMMAND ----------

# MAGIC %scala
# MAGIC // 一時ビューから読み込んでDDL形式でスキーマを表示
# MAGIC spark.table("blogs").schema.toDDL

# COMMAND ----------

# Hitsの値を2倍にして2行を表示
blogs_df.select(expr("Hits") * 2).show(2)

# COMMAND ----------

# HitsとIdを加算して表示
blogs_df.select(expr("Hits") + expr("Id")).show(truncate=False)

# COMMAND ----------

# Hitsが10000を越えるかどうかを示す列Big Hittersを追加して表示
blogs_df.withColumn("Big Hitters", (expr("Hits") > 10000)).show()

# COMMAND ----------

# First、Last、Id列を文字列結合しAuthorsId列を作成、AuthorsId列のみを4行分表示
blogs_df.withColumn(
    "AuthorsId", (concat(expr("First"), expr("Last"), expr("Id")))
).select(expr("AuthorsId")).show(n=4)
