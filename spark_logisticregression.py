from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ctr-prediction').getOrCreate()
spark_df = spark.read.csv('dataset/preprocessed_500000.csv', header = True, inferSchema = True)
spark_df = spark_df.drop('_c0')

#Handling Categorical Data
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
catergoricalFeature = ["banner_pos", "site_category", "app_category","device_type", "device_conn_type"]
stages = [] # stages in our Pipeline
for catergoricalFeat in catergoricalFeature:
    # StringIndexer for category Indexing
    strIndexer = StringIndexer(inputCol=catergoricalFeat, outputCol=catergoricalFeat + "Index").setHandleInvalid("skip")
    # Using OneHotEncoder
    ohencoder = OneHotEncoderEstimator(inputCols=[strIndexer.getOutputCol()], outputCols=[catergoricalFeat + "classVec"])
    # Add stages for pipeline
    stages += [strIndexer, ohencoder]

label_output = StringIndexer(inputCol="click", outputCol="label")
stages += [label_output]

numericCols = ["hour"]
assemblerInputs = [c + "classVec" for c in catergoricalFeature] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

# Applying Logistic regression
from pyspark.ml.classification import LogisticRegression
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(spark_df)
preppedDataDF = pipelineModel.transform(spark_df)

train, test = preppedDataDF.randomSplit([0.75, 0.25], seed = 2)
lrModel = LogisticRegression(featuresCol='features',labelCol = 'label', maxIter=10)
lrModel1 = lrModel.fit(train)

import matplotlib.pyplot as plt
import numpy as np
beta = np.sort(lrModel1.coefficients)
plt.plot(beta)
plt.ylabel('Beta Coefficients')
plt.show()

trainingSummary = lrModel1.summary
roc = trainingSummary.roc.toPandas()
plt.plot(roc['FPR'],roc['TPR'])
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

pr = trainingSummary.pr.toPandas()
plt.plot(pr['recall'],pr['precision'])
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()
