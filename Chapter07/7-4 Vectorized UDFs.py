# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # Vectorized User Defined Functions
# MAGIC
# MAGIC UDF、ベクトライズドUDF、ビルトインのメソッドのパフォーマンスを比較してみましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC ダミーデータを生成することからスタートしましょう。

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import col, count, rand, collect_list, explode, struct, count, pandas_udf

df = (spark
      .range(0, 10 * 1000 * 1000)
      .withColumn("id", (col("id") / 1000).cast("integer"))
      .withColumn("v", rand()))

df.cache()
df.count()

# COMMAND ----------

df.rdd.getNumPartitions()

# COMMAND ----------

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 列の値を1増加
# MAGIC
# MAGIC データフレームのそれぞれの値に1を加算するシンプルな例からスタートします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### PySpark UDF

# COMMAND ----------

@udf("double")
def plus_one(v):
    return v + 1

%timeit -n1 -r1 df.withColumn("v", plus_one(df.v)).agg(count(col("v"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC 別の構文 (SQL名前空間で利用可能)

# COMMAND ----------

from pyspark.sql.types import DoubleType

def plus_one(v):
    return v + 1
  
spark.udf.register("plus_one_udf", plus_one, DoubleType())

%timeit -n1 -r1 df.selectExpr("id", "plus_one_udf(v) as v").agg(count(col("v"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Scala UDF
# MAGIC
# MAGIC うわーっ！それぞれの値に1を加算するので時間がかかっています。Scala UDFでどのくらいの時間になるのかを見てみましょう。

# COMMAND ----------

df.createOrReplaceTempView("df")

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions._
# MAGIC
# MAGIC val df = spark.table("df")
# MAGIC
# MAGIC def plusOne: (Double => Double) = { v => v+1 }
# MAGIC val plus_one = udf(plusOne)

# COMMAND ----------

# MAGIC %scala
# MAGIC df.withColumn("v", plus_one($"v"))
# MAGIC   .agg(count(col("v")))
# MAGIC   .show()

# COMMAND ----------

# MAGIC %md
# MAGIC ワオ！Scala UDFの方がはるかに高速です。しかし、Spark 2.3時点では、Pythonでの処理を高速化する助けとなるベクトライズドUDFが利用できます。
# MAGIC
# MAGIC * [ブログ記事](https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html)
# MAGIC * [ドキュメント](https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow)
# MAGIC
# MAGIC ![Benchmark](https://databricks.com/wp-content/uploads/2017/10/image1-4.png)
# MAGIC
# MAGIC ベクトライズドUDFは処理を高速化するためにApache Arrowを活用します。どれだけ処理時間の改善になるのかを見てみましょう。

# COMMAND ----------

# MAGIC %md
# MAGIC [Apache Arrow](https://arrow.apache.org/)は、JVMとPythonプロセス間のデータ転送を効率的に行うためにSparkで利用されるインメモリの列指向データフォーマットです。詳細は[こちら](https://spark.apache.org/docs/latest/sql-pyspark-pandas-with-arrow.html)をご覧ください。
# MAGIC
# MAGIC Apache Arrowが有効化されている場合とされていない場合とで、SparkデータフレームからPandasへの変換にどのくらい時間がかかるのかを見てみましょう。

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

%timeit -n1 -r1 df.toPandas()

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "false")

%timeit -n1 -r1 df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ベクトライズドUDF

# COMMAND ----------

@pandas_udf("double")
def vectorized_plus_one(v):
    return v + 1

%timeit -n1 -r1 df.withColumn("v", vectorized_plus_one(df.v)).agg(count(col("v"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC いい感じです！Scala UDFほどではありませんが、少なくとも通常のPython UDFよりは優れています！
# MAGIC
# MAGIC Pandas UDFでは別の構文がいくつか存在します。

# COMMAND ----------

from pyspark.sql.functions import pandas_udf

def vectorized_plus_one(v):
    return v + 1

vectorized_plus_one_udf = pandas_udf(vectorized_plus_one, "double")

%timeit -n1 -r1 df.withColumn("v", vectorized_plus_one_udf(df.v)).agg(count(col("v"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ビルトインのメソッド
# MAGIC
# MAGIC ビルトインのメソッドとUDFのパフォーマンスを比較してみましょう。

# COMMAND ----------

from pyspark.sql.functions import lit

%timeit -n1 -r1 df.withColumn("v", df.v + lit(1)).agg(count(col("v"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## subtract meanの計算
# MAGIC
# MAGIC ここまでは、スカラーの戻り値を取り扱ってきました。ここでは、グルーピングされたUDFを活用します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### PySpark UDF

# COMMAND ----------

from pyspark.sql import Row
import pandas as pd

@udf(ArrayType(df.schema))
def subtract_mean(rows):
  vs = pd.Series([r.v for r in rows])
  vs = vs - vs.mean()
  return [Row(id=rows[i]["id"], v=float(vs[i])) for i in range(len(rows))]
  
%timeit -n1 -r1 (df.groupby("id").agg(collect_list(struct(df["id"], df["v"])).alias("rows")).withColumn("new_rows", subtract_mean(col("rows"))).withColumn("new_row", explode(col("new_rows"))).withColumn("id", col("new_row.id")).withColumn("v", col("new_row.v")).agg(count(col("v"))).show())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vectorized UDF

# COMMAND ----------

def vectorized_subtract_mean(pdf: pd.Series) -> pd.Series:
	return pdf.assign(v=pdf.v - pdf.v.mean())

%timeit -n1 -r1 df.groupby("id").applyInPandas(vectorized_subtract_mean, df.schema).agg(count(col("v"))).show()
