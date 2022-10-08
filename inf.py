import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import  Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import  IDF
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel
import sys 

create_data=int(sys.argv[1])
db_file=sys.argv[2]
model_path=sys.argv[3]
print(create_data)


if create_data:
    print("create data")
    conn = sqlite3.connect(db_file, isolation_level=None,
                           detect_types=sqlite3.PARSE_COLNAMES)
    db_df = pd.read_sql_query("SELECT * FROM Reviews", conn)
    db_df.to_csv('database_test.csv', index=False)


############### set config ###################
spark = SparkSession.builder \
    .config("spark.driver.memory", "33g") \
    .config("spark.sql.autoBroadcastJoinThreshold","100")\
    .appName('my-cool-app') \
    .getOrCreate()
df = spark.read.csv('database_test.csv', header = True, inferSchema = True)
##############################################

############ vis the data ####################
print("#"*10,"vis the data","#"*10)
df.printSchema()
pd.DataFrame(df.take(5), columns=df.columns).transpose()
results=df.groupby('Score').count().toPandas()
print(results)
score_dict={}
for idx,score_rate in enumerate(results["Score"]): # Score and count
  score_dict[score_rate]=results["count"][idx]

print(score_dict)
print("#" * 60)
# data is noised, we need to clean it 
# first option my option with any outlaier ( remove this )
# seconde option, replace it with a meaningfull value, in our case this is abusive
# https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrame.replace.html for replace 
#
print("#"*10,"clean the data","#"*10)
df=df.withColumn("Score",df.Score.cast('int'))
df.groupby('Score').count().toPandas() # we got Nan now represinting the text in score field
df=df.na.drop()
df.groupby('Score').count().toPandas() # holay we remove the nan
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().toPandas().transpose() # max score is 69 ,  this is a clear outlaier
#remove outlaiers score is between 0 and 5  based on the sorted count of scores
'''
idx score count
13	5	    361648
18	4	    80141
5	  1     52635
9	  3	    42502
32	2	    29877
33	0	    1111
21	7	    4
...
'''
df=df.where("Score<6")

print("#" * 60)
print("#"*10,"vis clean data","#"*10)
numeric_features = [t[0] for t in df.dtypes if t[1] == 'int']
df.select(numeric_features).describe().toPandas().transpose() # max score is 69 ,  this is a clear outlaier
print("#" * 60)


print("#"*10,"Drop unnecessary columns","#"*10)
dataset = df.drop('ProductId')
dataset = dataset.drop('UserId')
dataset = dataset.drop('ProfileName')
dataset = dataset.drop('HelpfulnessNumerator')
dataset = dataset.drop('HelpfulnessDenominator')
dataset = dataset.drop('Time')
dataset = dataset.drop('Summary')
dataset = dataset.drop('Id')
dataset.show()
print("#" * 60)

print("#"*10,"vis text data","#"*10)
row_list = dataset.collect()
dfx = dataset.withColumn("length_of_text", F.length("Text"))
dfx.show(truncate=False)
print("#" * 60)
print("#"*10,"Drop text lines too large columns","#"*10)
numeric_features = [t[0] for t in dfx.dtypes if t[1] == 'int']
dfx.select(numeric_features).describe().toPandas().transpose() # max score is 69 ,  this is a clear outlaier
dfx.groupby('length_of_text').count().toPandas()
dataset=dfx.where("length_of_text<2001")
dataset = dataset.drop('length_of_text')
print("#" * 60)



print("#"*10,"tokenize the data stage 1","#"*10)
tokenizer = Tokenizer(inputCol="Text", outputCol="words")
wordsData = tokenizer.transform(dataset)
wordsData=wordsData.withColumnRenamed("Score","label")
wordsData.show()
print("#" * 60)

print("#"*10,"tokenize the data stage 2","#"*10)
count = CountVectorizer (inputCol="words", outputCol="rawFeatures")
model = count.fit(wordsData)
featurizedData = model.transform(wordsData)
featurizedData.show()
print("#" * 60)


print("#"*10,"create test data","#"*10)
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
rescaledData.select("label", "features").show()  # We want only the label and features columns for our machine learning models
rescaledData=rescaledData.select("label", "features")
rescaledData.show()
seed = 0  # set seed for reproducibility
trainDF, testDF = rescaledData.randomSplit([0.0,1.0],seed)#100% test
print(testDF.count(),"test size")
print("#" * 60)

print("#"*10,"inf on test data","#"*10)

persistedModel = LogisticRegressionModel.load(f"{model_path}")
rf_predictions = persistedModel.transform(testDF)
print(rf_predictions,rf_predictions)
rf_predictions.show()
rf_predictions.select("prediction").show()
multi_evaluator = MulticlassClassificationEvaluator(labelCol = 'label', metricName = 'accuracy')
print(f'logistic regression classifier  Accuracy:', multi_evaluator.evaluate(rf_predictions))
print("#" * 60)

