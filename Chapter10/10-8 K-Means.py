# Databricks notebook source
# MAGIC %md
# MAGIC # 分散K-Means
# MAGIC
# MAGIC このノートブックでは、データをクラスタリングするためにK-Meansを使用します。ラベル(Irisのタイプ)を持つIrisデータセットを使用しますが、トレーニングするために使用するのではなく、モデルの評価のためにラベルだけを使用します。
# MAGIC
# MAGIC 最後には、分散環境でどのように実装されているのかを見ていきます。

# COMMAND ----------

from sklearn.datasets import load_iris
import pandas as pd

# sklearnからデータセットをロードし、Sparkデータフレームに変換
iris = load_iris()
iris_pd = pd.concat([pd.DataFrame(iris.data, columns=iris.feature_names), pd.DataFrame(iris.target, columns=["label"])], axis=1)
irisDF = spark.createDataFrame(iris_pd)
display(irisDF)

# COMMAND ----------

# MAGIC %md
# MAGIC "特徴量"として4つの値があることに注意してください。(可視化できるように)これらを2つの値に削減し、`DenseVector`に変換します。これを行うためには`VectorAssembler`を使用します。

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=["sepal length (cm)", "sepal width (cm)"], outputCol="features")
irisTwoFeaturesDF = vecAssembler.transform(irisDF)
display(irisTwoFeaturesDF)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=3, seed=221, maxIter=20)

# estimatorのfitを呼び出し、irisTwoFeaturesDFを入力
model = kmeans.fit(irisTwoFeaturesDF)

# KMeansModelからclusterCentersを取得
centers = model.clusterCenters()

# クラスターの予測結果を追加することでデータフレームを変換するためにモデルを使用
transformedDF = model.transform(irisTwoFeaturesDF)

print(centers)

# COMMAND ----------

# MAGIC %md
# MAGIC イテレーションの数を変えてフィッティングします。

# COMMAND ----------

modelCenters = []
iterations = [0, 2, 4, 7, 10, 20]
for i in iterations:
    kmeans = KMeans(k=3, seed=221, maxIter=i)
    model = kmeans.fit(irisTwoFeaturesDF)
    modelCenters.append(model.clusterCenters())   

# COMMAND ----------

print("modelCenters:")
for centroids in modelCenters:
  print(centroids)

# COMMAND ----------

# MAGIC %md
# MAGIC データのtrueラベルに対してクラスタリングがどの程度のパフォーマンスであるのかを可視化しましょう。
# MAGIC
# MAGIC 覚えておいてください: K-meansではトレーニングの際にtrueラベルは使用しませんが、評価に使用することはできます。
# MAGIC
# MAGIC こちらでは、星マークがクラスターの中心です。

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def prepareSubplot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999', 
                gridWidth=1.0, subplots=(1, 1)):
    """プロットレイアウト生成のテンプレート"""
    plt.close()
    fig, axList = plt.subplots(subplots[0], subplots[1], figsize=figsize, facecolor='white', 
                               edgecolor='white')
    if not isinstance(axList, np.ndarray):
        axList = np.array([axList])
    
    for ax in axList.flatten():
        ax.axes.tick_params(labelcolor='#999999', labelsize='10')
        for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
            axis.set_ticks_position('none')
            axis.set_ticks(ticks)
            axis.label.set_color('#999999')
            if hideLabels: axis.set_ticklabels([])
        ax.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
        map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
        
    if axList.size == 1:
        axList = axList[0]  # 通常のプロットには単一のaxesオブジェクトを返却
    return fig, axList

# COMMAND ----------

data = irisTwoFeaturesDF.select("features", "label").collect()
features, labels = zip(*data)

x, y = zip(*features)
centers = modelCenters[5]
centroidX, centroidY = zip(*centers)
colorMap = 'Set1'  # was 'Set2', 'Set1', 'Dark2', 'winter'

fig, ax = prepareSubplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(8,6))
plt.scatter(x, y, s=14**2, c=labels, edgecolors='#8cbfd0', alpha=0.80, cmap=colorMap)
plt.scatter(centroidX, centroidY, s=22**2, marker='*', c='yellow')
cmap = cm.get_cmap(colorMap)

colorIndex = [.5, .99, .0]
for i, (x,y) in enumerate(centers):
    print(cmap(colorIndex[i]))
    for size in [.10, .20, .30, .40, .50]:
        circle1=plt.Circle((x,y),size,color=cmap(colorIndex[i]), alpha=.10, linewidth=2)
        ax.add_artist(circle1)

ax.set_xlabel('Sepal Length'), ax.set_ylabel('Sepal Width')
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC それぞれのイテレーションでクラスターのオーバーレイを確認することに加え、イテレーションごとにクラスターの中心がどのように動くのかを確認することができます(イテレーションが少ない場合に結果がどのようなものになるのかも確認できます)。

# COMMAND ----------

x, y = zip(*features)

oldCentroidX, oldCentroidY = None, None

fig, axList = prepareSubplot(np.arange(-1, 1.1, .4), np.arange(-1, 1.1, .4), figsize=(11, 15),
                             subplots=(3, 2))
axList = axList.flatten()

for i,ax in enumerate(axList[:]):
    ax.set_title('K-means for {0} iterations'.format(iterations[i]), color='#999999')
    centroids = modelCenters[i]
    centroidX, centroidY = zip(*centroids)
    
    ax.scatter(x, y, s=10**2, c=labels, edgecolors='#8cbfd0', alpha=0.80, cmap=colorMap, zorder=0)
    ax.scatter(centroidX, centroidY, s=16**2, marker='*', c='yellow', zorder=2)
    if oldCentroidX and oldCentroidY:
      ax.scatter(oldCentroidX, oldCentroidY, s=16**2, marker='*', c='grey', zorder=1)
    cmap = cm.get_cmap(colorMap)
    
    colorIndex = [.5, .99, 0.]
    for i, (x1,y1) in enumerate(centroids):
      print(cmap(colorIndex[i]))
      circle1=plt.Circle((x1,y1),.35,color=cmap(colorIndex[i]), alpha=.40)
      ax.add_artist(circle1)
    
    ax.set_xlabel('Sepal Length'), ax.set_ylabel('Sepal Width')
    oldCentroidX, oldCentroidY = centroidX, centroidY

plt.tight_layout()

display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC それでは、分散環境では何が起きているのかを見てみましょう。

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://files.training.databricks.com/images/Mapstage.png" height=500px>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://files.training.databricks.com/images/Mapstage2.png" height=500px>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://files.training.databricks.com/images/ReduceStage.png" height=500px>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <img src="https://files.training.databricks.com/images/Communication.png" height=500px>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Take Aways
# MAGIC
# MAGIC 分散MLアルゴリズムを設計/選択する際には:
# MAGIC * コミュニケーションが鍵となります！
# MAGIC * お使いのデータ/モデルの次元数とどれだけのデータを必要とするのかを検討します
# MAGIC * データのパーティショニングと構成が重要となります。
