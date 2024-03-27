# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC # Caching Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### _cache()_ の使用

# COMMAND ----------

# MAGIC %md
# MAGIC いくつかのカラムを持つ大規模データセットを作成します。

# COMMAND ----------

from pyspark.sql.functions import col

df = spark.range(1 * 10000000).toDF("id").withColumn("square", col("id") * col("id"))
df.cache().count()

# COMMAND ----------

df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC どこにデータが格納されるのかを確認するためにSpark UIのstorageタブをチェックします。

# COMMAND ----------

df.unpersist() # unpersistしない場合、下のdf2はdfと同じクエリープランであるためdf2はキャッシュされません

# COMMAND ----------

# MAGIC %md
# MAGIC ### _persist(StorageLevel.Level)_ の使用

# COMMAND ----------

from pyspark import StorageLevel

df2 = spark.range(1 * 10000000).toDF("id").withColumn("square", col("id") * col("id"))
df2.persist(StorageLevel.DISK_ONLY).count()

# COMMAND ----------

df2.count()

# COMMAND ----------

# MAGIC %md
# MAGIC データがどこに格納されているのかを確認するためにSpark UIのStorageタブをチェックします。

# COMMAND ----------

df2.unpersist()

# COMMAND ----------

df.createOrReplaceTempView("dfTable")
spark.sql("CACHE TABLE dfTable")

# COMMAND ----------

# MAGIC %md
# MAGIC どこにデータが格納されるのかを確認するためにSpark UIのstorageタブをチェックします。

# COMMAND ----------

spark.sql("SELECT count(*) FROM dfTable").show()
