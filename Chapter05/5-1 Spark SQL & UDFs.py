# Databricks notebook source
# MAGIC
# MAGIC %md
# MAGIC ## Chapter 5: Spark SQL and DataFrames: Interacting with External Data Sources
# MAGIC This notebook contains for code samples for *Chapter 5: Spark SQL and DataFrames: Interacting with External Data Sources*.

# COMMAND ----------

# MAGIC %md
# MAGIC ### ユーザー定義関数
# MAGIC
# MAGIC Apache Sparkでは数多くの関数を提供していますが、Sparkの柔軟性によってデータエンジニアやデータサイエンティストは自分の関数を定義することができます(すなわち、user-defined functionあるいはUDF)。

# COMMAND ----------

from pyspark.sql.types import LongType

# 3乗関数の作成
def cubed(s):
  return s * s * s

# UDFの登録
spark.udf.register("cubed", cubed, LongType())

# 一時ビューの作成
spark.range(1, 9).createOrReplaceTempView("udf_test")

# COMMAND ----------

spark.sql("SELECT id, cubed(id) AS id_cubed FROM udf_test").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pandas UDFを用いたPySpark UDFのスピードアップと分散処理
# MAGIC
# MAGIC PySpark UDFで不可避な問題の一つが、Scala UDFよりも遅いということです。これは、PySpark UDFでは、処理コストが非常に高いJVMとPythonの間でのデータ移動が必要なためです。この問題を解決するために、pandas UDF(ベクトル化UDFとも呼ばれます)がApache Spark 2.3の一部として導入されました。これは、データ転送にApache Sparkを活用し、データの操作でpandasを活用します。デコレーターとしてpandas_udfキーワードを用いるか、関数自身をラッピングするためにpandas UDFを定義するだけです。データがApache Arrowフォーマットになると、Pythonプロセスで利用可能なフォーマットになっているので、シリアライズ/pickleの必要がなくなります。個々の入力を行ごとに操作するのではなく、pandasのシリーズやデータフレームを操作することになります(すなわち、ベクトル化処理)。

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col, pandas_udf
from pyspark.sql.types import LongType

# 3乗関数の宣言 
def cubed(a: pd.Series) -> pd.Series:
    return a * a * a

# 3乗関数に対するpandas UDFの作成 
cubed_udf = pandas_udf(cubed, returnType=LongType())

# COMMAND ----------

# MAGIC %md
# MAGIC ### pandasデータフレームの使用

# COMMAND ----------

# Pandasシリーズの作成
x = pd.Series([1, 2, 3])

# ローカルのPandasデータに対して実行されるpandas_udfの関数
print(cubed(x))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sparkデータフレームの使用

# COMMAND ----------

# Sparkデータフレームの作成
df = spark.range(1, 4)

# Sparkベクトル化UDFとして関数を実行
df.select("id", cubed_udf(col("id"))).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## データフレームやSpark SQLにおける高階関数
# MAGIC
# MAGIC 複雑なデータタイプはシンプルなデータタイプの集合体なので、直接複雑なデータタイプを操作したくなるものです。記事*Introducing New Built-in and Higher-Order Functions for Complex Data Types in Apache Spark 2.4*で言及したように、複雑なデータ型の操作では、2つの典型的なソリューションがあります。
# MAGIC
# MAGIC 1. 以下のコードで示しているように、ネストされた構造を個々の行に分割し、何かしらの関数を適用し、ネストされた構造を再構築する(オプション1をご覧ください)
# MAGIC 1. ユーザー定義関数(UDF)を構築する

# COMMAND ----------

# 配列型データセットの作成
arrayData = [[1, (1, 2, 3)], [2, (2, 3, 4)], [3, (3, 4, 5)]]

# スキーマの作成
from pyspark.sql.types import *
arraySchema = (StructType([
      StructField("id", IntegerType(), True), 
      StructField("values", ArrayType(IntegerType()), True)
      ]))

# データフレームの作成
df = spark.createDataFrame(spark.sparkContext.parallelize(arrayData), arraySchema)
df.createOrReplaceTempView("table")
df.printSchema()
df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### オプション1: ExplodeとCollect
# MAGIC
# MAGIC このネストされたSQL文では、最初に値の中の個々の要素(`value`)に対応する(idを持つ)新規行を作成する`explode(values)`を実行します。

# COMMAND ----------

spark.sql("""
SELECT id, collect_list(value + 1) AS newValues
  FROM  (SELECT id, explode(values) AS value
        FROM table) x
 GROUP BY id
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### オプション2: ユーザー定義関数
# MAGIC
# MAGIC 同じタスク(`values`のそれぞれの要素の値に1を足す)を実行するために、加算のオペレーションを実行するためにそれぞれの要素(`value`)に対するイテレーションのためにmapを用いるユーザー定義関数(UDF)を作成することもできます。

# COMMAND ----------

from pyspark.sql.types import IntegerType
from pyspark.sql.types import ArrayType

# UDFの作成
def addOne(values):
  return [value + 1 for value in values]

# UDFの登録
spark.udf.register("plusOneIntPy", addOne, ArrayType(IntegerType()))  

# データのクエリー
spark.sql("SELECT id, plusOneIntPy(values) AS values FROM table").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 高階関数
# MAGIC
# MAGIC 上述したビルトインの関数に加え、引数として匿名のラムダ関数を受け取る高階関数があります。

# COMMAND ----------

from pyspark.sql.types import *
schema = StructType([StructField("celsius", ArrayType(IntegerType()))])

t_list = [[35, 36, 32, 30, 40, 42, 38]], [[31, 32, 34, 55, 56]]
t_c = spark.createDataFrame(t_list, schema)
t_c.createOrReplaceTempView("tC")

# データフレームの表示
t_c.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform
# MAGIC
# MAGIC `transform(array<T>, function<T, U>): array<U>`
# MAGIC
# MAGIC transform関数は、(map関数と同じように)入力配列のそれぞれの要素に関数を適用することで配列を生成します。

# COMMAND ----------

# 気温の配列に対して摂氏から華氏を計算
spark.sql("""SELECT celsius, transform(celsius, t -> ((t * 9) div 5) + 32) as fahrenheit FROM tC""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Filter
# MAGIC
# MAGIC `filter(array<T>, function<T, Boolean>): array<T>`
# MAGIC
# MAGIC filter関数はboolean関数がtrueになる要素を持つ配列を生成します。

# COMMAND ----------

# 気温配列を temperatures > 38C でフィルタリング
spark.sql("""SELECT celsius, filter(celsius, t -> t > 38) as high FROM tC""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Exists
# MAGIC
# MAGIC `exists(array<T>, function<T, V, Boolean>): Boolean`
# MAGIC
# MAGIC exists関数はboolean関数が入力配列のいずれかの要素でtrueになる場合にはtrueを返します。

# COMMAND ----------

# 気温の配列に38Cが含まれるかどうか
spark.sql("""
SELECT celsius, exists(celsius, t -> t = 38) as threshold
FROM tC
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Reduce
# MAGIC
# MAGIC `reduce(array<T>, B, function<B, T, B>, function<B, R>)`
# MAGIC
# MAGIC reduce関数は`function<B, T, B>`を用いてバッファ`B`に配列をマージし、最後のバッファに最終関数`function<B, R>`を適用することで、配列を単一の値に集約します。

# COMMAND ----------

# 平均気温を計算し、Fに変換
spark.sql("""
SELECT celsius, 
       reduce(
          celsius, 
          0, 
          (t, acc) -> t + acc, 
          acc -> (acc div size(celsius) * 9 div 5) + 32
        ) as avgFahrenheit 
  FROM tC
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## データフレームとSpark SQLの一般的なリレーショナルオペレーター
# MAGIC
# MAGIC Spark SQLのパワーは、数多くのデータフレームオペレーションを持っているということです(型なしデータセットオペレーションとも呼ばれます)。
# MAGIC
# MAGIC 完全なリストについては、[Spark SQL, Built-in Functions](https://spark.apache.org/docs/latest/api/sql/index.html)をご覧下さい。
# MAGIC
# MAGIC 次のセクションでは、以下の一般的なリレーショナルオペレーターにフォーカスします:
# MAGIC * UnionとJoin
# MAGIC * ウィンドウ処理
# MAGIC * 変更

# COMMAND ----------

from pyspark.sql.functions import expr

# ファイルパスの設定
delays_path = "/databricks-datasets/learning-spark-v2/flights/departuredelays.csv"
airports_path = "/databricks-datasets/learning-spark-v2/flights/airport-codes-na.txt"

# airportsデータセットの取得
airports = spark.read.options(header="true", inferSchema="true", sep="\t").csv(airports_path)
airports.createOrReplaceTempView("airports_na")

# 出発遅延データの取得
delays = spark.read.options(header="true").csv(delays_path)
delays = (delays
          .withColumn("delay", expr("CAST(delay as INT) as delay"))
          .withColumn("distance", expr("CAST(distance as INT) as distance")))

delays.createOrReplaceTempView("departureDelays")

# 一時的な小さいテーブルを作成
foo = delays.filter(expr("""
            origin == 'SEA' AND 
            destination == 'SFO' AND 
            date like '01010%' AND 
            delay > 0"""))

foo.createOrReplaceTempView("foo")

# COMMAND ----------

spark.sql("SELECT * FROM airports_na LIMIT 10").show()

# COMMAND ----------

spark.sql("SELECT * FROM departureDelays LIMIT 10").show()

# COMMAND ----------

spark.sql("SELECT * FROM foo LIMIT 10").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Union

# COMMAND ----------

# 二つのテーブルをUnion: delaysとbarを結合してbarを作成
bar = delays.union(foo)
bar.createOrReplaceTempView("bar")
# barに含まれるfooのデータを表示: 元々fooはdelaysから作っているのでレコードが重複している
bar.filter(expr("origin == 'SEA' AND destination == 'SFO' AND date LIKE '01010%' AND delay > 0")).show()

# COMMAND ----------

# Spark SQLでも確認
spark.sql("""
SELECT * 
FROM bar 
WHERE origin = 'SEA' 
   AND destination = 'SFO' 
   AND date LIKE '01010%' 
   AND delay > 0
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join
# MAGIC
# MAGIC デフォルトは`inner join`です。他のオプション`inner, cross, outer, full, full_outer, left, left_outer, right, right_outer, left_semi, left_anti`があります。
# MAGIC
# MAGIC 詳細はこちら:
# MAGIC * [PySpark Join](https://spark.apache.org/docs/latest/api/python/reference/pyspark.pandas/api/pyspark.pandas.DataFrame.join.html?highlight=join#pyspark.pandas.DataFrame.join)

# COMMAND ----------

# 到着遅延データ(foo)とフライト情報をjoin: キーは出発地の空港
foo.join(
  airports, 
  airports.IATA == foo.origin
).select("City", "State", "date", "delay", "distance", "destination").show()

# COMMAND ----------

spark.sql("""
SELECT a.City, a.State, f.date, f.delay, f.distance, f.destination 
  FROM foo f
  JOIN airports_na a
    ON a.IATA = f.origin
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## ウィンドウ関数
# MAGIC
# MAGIC 素晴らしいリファレンス: [Introduction Windowing Functions in Spark SQL](https://databricks.com/blog/2015/07/15/introducing-window-functions-in-spark-sql.html)
# MAGIC
# MAGIC > そのコアでは、ウィンドウ関数はフレームと呼ばれる行のグループをベースにして、テーブルのすべての入力行の値の戻り値を計算します。すべての入力行は関連づけられるユニークなフレームを持ちます。このウィンドウ関数の特性によって、他の関数よりもパワフルなものとなっており、ユーザーはウィンドウ関数なしには簡潔な方法で表現が困難(あるいは不可能)なさまざまなデータ処理タスクを表現することができます。

# COMMAND ----------

spark.sql("DROP TABLE IF EXISTS departureDelaysWindow")
spark.sql("""
CREATE TABLE departureDelaysWindow AS
SELECT origin, destination, sum(delay) as TotalDelays 
  FROM departureDelays 
 WHERE origin IN ('SEA', 'SFO', 'JFK') 
   AND destination IN ('SEA', 'SFO', 'JFK', 'DEN', 'ORD', 'LAX', 'ATL') 
 GROUP BY origin, destination
""")

spark.sql("""SELECT * FROM departureDelaysWindow""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC SEA、SFO、JFKが出発地の合計遅延がトップスリーの目的地はどこでしょうか？

# COMMAND ----------

spark.sql("""
SELECT origin, destination, sum(TotalDelays) as TotalDelays
 FROM departureDelaysWindow
WHERE origin = 'SEA'
GROUP BY origin, destination
ORDER BY TotalDelays DESC
LIMIT 3
""").show()

# COMMAND ----------

spark.sql("""
SELECT origin, destination, TotalDelays, rank 
  FROM ( 
     SELECT origin, destination, TotalDelays, dense_rank() 
       OVER (PARTITION BY origin ORDER BY TotalDelays DESC) as rank 
       FROM departureDelaysWindow
  ) t 
 WHERE rank <= 3
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 変更
# MAGIC
# MAGIC この他の一般的なデータフレームのオペレーションは、データフレームに対して変更を加えることです。背後にあるRDDは、Sparkオペレーションのデータリネージを保持できるように不変である(つまり、変更不可)ことを思い出してください。このため、データフレーム自体は不変ですが、例えば、異なる列を持つ別のデータフレームを新たに作成するオペレーションを通じて変更を行うことができます。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 新規列の追加

# COMMAND ----------

foo2 = foo.withColumn("status", expr("CASE WHEN delay <= 10 THEN 'On-time' ELSE 'Delayed' END"))
foo2.show()

# COMMAND ----------

spark.sql("""SELECT *, CASE WHEN delay <= 10 THEN 'On-time' ELSE 'Delayed' END AS status FROM foo""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 列の削除

# COMMAND ----------

foo3 = foo2.drop("delay")
foo3.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 列名の変更

# COMMAND ----------

foo4 = foo3.withColumnRenamed("status", "flight_status")
foo4.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### ピボット
# MAGIC
# MAGIC 素晴らしいリファレンス [SQL Pivot: Converting Rows to Columns](https://databricks.com/blog/2018/11/01/sql-pivot-converting-rows-to-columns.html)

# COMMAND ----------

spark.sql("""SELECT destination, CAST(SUBSTRING(date, 0, 2) AS int) AS month, delay FROM departureDelays WHERE origin = 'SEA'""").show(10)

# COMMAND ----------

spark.sql("""
SELECT * FROM (
SELECT destination, CAST(SUBSTRING(date, 0, 2) AS int) AS month, delay 
  FROM departureDelays WHERE origin = 'SEA' 
) 
PIVOT (
  CAST(AVG(delay) AS DECIMAL(4, 2)) as AvgDelay, MAX(delay) as MaxDelay
  FOR month IN (1 JAN, 2 FEB, 3 MAR)
)
ORDER BY destination
""").show()

# COMMAND ----------

spark.sql("""
SELECT * FROM (
SELECT destination, CAST(SUBSTRING(date, 0, 2) AS int) AS month, delay 
  FROM departureDelays WHERE origin = 'SEA' 
) 
PIVOT (
  CAST(AVG(delay) AS DECIMAL(4, 2)) as AvgDelay, MAX(delay) as MaxDelay
  FOR month IN (1 JAN, 2 FEB)
)
ORDER BY destination
""").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rollup
# MAGIC Refer to [What is the difference between cube, rollup and groupBy operators?](https://stackoverflow.com/questions/37975227/what-is-the-difference-between-cube-rollup-and-groupby-operators)
# MAGIC
