# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # Spark Tables
# MAGIC
# MAGIC このノートブックでは、データベース、テーブル、カラムをクエリーするためのSpark Catalog Interface APIの使い方を説明します。
# MAGIC
# MAGIC 文書化されているメソッドの完全なリストは[こちら](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Catalog)です。

# COMMAND ----------

us_flights_file = "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ### マネージドテーブルの作成

# COMMAND ----------

# カタログ、データベース、マネージドテーブルの作成
spark.sql("USE CATALOG takaakiyayoi_catalog") 
spark.sql("DROP DATABASE IF EXISTS learn_spark_db CASCADE") 
spark.sql("CREATE DATABASE learn_spark_db")
spark.sql("USE learn_spark_db")
spark.sql("CREATE TABLE us_delay_flights_tbl(date STRING, delay INT, distance INT, origin STRING, destination STRING)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### データベースの表示

# COMMAND ----------

display(spark.catalog.listDatabases())

# COMMAND ----------

# MAGIC %md
# MAGIC ## US Flightsテーブルの読み込み

# COMMAND ----------

df = (spark.read.format("csv")
      .schema("date STRING, delay INT, distance INT, origin STRING, destination STRING")
      .option("header", "true")
      .option("path", "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv")
      .load())

# COMMAND ----------

# MAGIC %md
# MAGIC ## テーブルに保存

# COMMAND ----------

df.write.mode("overwrite").saveAsTable("us_delay_flights_tbl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## テーブルのキャッシュ

# COMMAND ----------

# MAGIC %sql
# MAGIC CACHE TABLE us_delay_flights_tbl

# COMMAND ----------

# MAGIC %md
# MAGIC テーブルがキャッシュされているかどうかを確認します。

# COMMAND ----------

spark.catalog.isCached("us_delay_flights_tbl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## データベースのテーブルの表示
# MAGIC
# MAGIC テーブルはSparkによって管理されている`MANAGED`であることに注意してください。

# COMMAND ----------

spark.catalog.listTables(dbName="learn_spark_db")

# COMMAND ----------

# MAGIC %md
# MAGIC ## テーブルのカラムの表示

# COMMAND ----------

spark.catalog.listColumns("us_delay_flights_tbl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## アンマネージドテーブルの作成

# COMMAND ----------

# Drop the database and create unmanaged tables
spark.sql("DROP DATABASE IF EXISTS learn_spark_db CASCADE")
spark.sql("CREATE DATABASE learn_spark_db")
spark.sql("USE learn_spark_db")
spark.sql("CREATE TABLE us_delay_flights_tbl (date STRING, delay INT, distance INT, origin STRING, destination STRING) USING csv OPTIONS (path '/databricks-datasets/learning-spark-v2/flights/departuredelays.csv')")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display Tables
# MAGIC
# MAGIC **Note**: The table type here that tableType='EXTERNAL', which indicates it's unmanaged by Spark, whereas above the tableType='MANAGED'

# COMMAND ----------

spark.catalog.listTables(dbName="learn_spark_db")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display Columns for a table

# COMMAND ----------

spark.catalog.listColumns("us_delay_flights_tbl")
