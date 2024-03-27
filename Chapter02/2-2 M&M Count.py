# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC ## Example 2-1 M&M Count

# COMMAND ----------

from pyspark.sql.functions import *

# CSVファイルのパス
mnm_file = "/databricks-datasets/learning-spark-v2/mnm_dataset.csv"

# COMMAND ----------

# MAGIC %md
# MAGIC ### CSVからの読み込みおよびスキーマの推定

# COMMAND ----------

mnm_df = (spark
          .read
          .format("csv") # フォーマット指定
          .option("header", "true") # ヘッダーあり
          .option("inferSchema", "true") # スキーマを推定
          .load(mnm_file))

display(mnm_df) # displayコマンドでデータフレームを表示

# COMMAND ----------

mnm_df.rdd.getNumPartitions()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 全ての色のカウントを集計し、州と色でgroupByし、カウントの降順でorderBy

# COMMAND ----------

count_mnm_df = (mnm_df
                .select("State", "Color", "Count") # State、Color、Countを選択
                .groupBy("State", "Color") # StateとColorでグルーピング
                .agg(count("Count").alias("Total")) # カウントを集計し列名をTotalに
                .orderBy("Total", ascending=False)) # Totalの降順でソート

count_mnm_df.show(n=60, truncate=False) # 先頭60件を表示
print(f"Total Rows = {count_mnm_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Stateでフィルタリングすることでカルフォルニアのカウント集計値を取得

# COMMAND ----------

ca_count_mnm_df = (mnm_df
                   .select("State", "Color", "Count") # State、Color、Countを選択
                   .where(mnm_df.State == "CA") # StateがCAであるものをフィルタリング
                   .groupBy("State", "Color") # StateとColorでグルーピング
                   .agg(count("Count").alias("Total")) # カウントを集計し列名をTotalに
                   .orderBy("Total", ascending=False)) # Totalの降順でソート

ca_count_mnm_df.show(n=10, truncate=False) # 先頭10件を表示
