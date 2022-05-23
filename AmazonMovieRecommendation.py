import findspark
findspark.init()
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
import pandas as pd



from pyspark.ml.feature import StringIndexer
spark = SparkSession \
    .builder \
    .appName("Recommendation App") \
    .config("spark.driver.host", "localhost") \
    .getOrCreate()
default_conf = spark.sparkContext._conf.getAll()
print(default_conf)

conf = spark.sparkContext._conf.setAll([('spark.executor.memory', '2g'),
                                        ('spark.app.name', 'HW2 Recommendation'),
                                        ('spark.executor.cores', '4'),
                                        ('spark.cores.max', '10'),
                                        ('spark.driver.memory','16g'),
                                        ('spark.kryoserializer.buffer.max','1g'),
                                        ('spark.default.parallelism','300'),
                                       ('spark.sql.shuffle.partitions','300')])
v = spark.sparkContext._conf.getAll()
print(v)

movies_ratings_df = spark.read.json("/Users/saheedadepoju/Documents/CSE272/HW2/Movies_and_TV.json.gz")

movies_ratings_df_subset = movies_ratings_df.limit(100000)

movies_asin_index_subset=movies_ratings_df_subset.select("asin").distinct().withColumn("asin_index", monotonically_increasing_id())

movies_training_data_merged_1 = movies_ratings_df_subset.join(movies_asin_index_subset.select('asin', 'asin_index'), ['asin'])
movies_review_subset = movies_training_data_merged_1.select("reviewerID").distinct().withColumn("reviewerID_index", monotonically_increasing_id())
movies_training_data_merged_2 = movies_training_data_merged_1.join(movies_review_subset.select('reviewerID', 'reviewerID_index'), ['reviewerID'])

(training,test)=movies_training_data_merged_2.randomSplit([0.8, 0.2])
als=ALS(maxIter=5,regParam=0.09,rank=25,userCol="reviewerID_index",itemCol="asin_index",ratingCol="overall",coldStartStrategy="drop",nonnegative=True)
model=als.fit(training)
evaluator=RegressionEvaluator(metricName="rmse",labelCol="overall",predictionCol="prediction")
evaluator_1=RegressionEvaluator(metricName="mae",labelCol="overall",predictionCol="prediction")

predictions=model.transform(test)
rmse=evaluator.evaluate(predictions)

print("RMSE="+str(rmse))
predictions.show()

mae = evaluator_1.evaluate(predictions)
print("MAE="+str(mae))
user_recs=model.recommendForAllUsers(20).show(10)


recs=model.recommendForAllUsers(10).toPandas()
nrecs=recs.recommendations.apply(pd.Series) \
            .merge(recs, right_index = True, left_index = True) \
            .drop(["recommendations"], axis = 1) \
            .melt(id_vars = ['reviewerID_index'], value_name = "recommendation") \
            .drop("variable", axis = 1) \
            .dropna()
nrecs=nrecs.sort_values('reviewerID_index')
nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['reviewerID_index']], axis = 1)
nrecs.columns = [

        'ProductID_index',
        'Rating',
        'UserID_index'

     ]
md=movies_training_data_merged_2.select(movies_training_data_merged_2['reviewerID'],movies_training_data_merged_2['reviewerID_index'],movies_training_data_merged_2['asin'],movies_training_data_merged_2['asin_index'])
md=md.toPandas()
dict1 =dict(zip(md['reviewerID_index'],md['reviewerID']))
dict2=dict(zip(md['asin_index'],md['asin']))
nrecs['reviewerID']=nrecs['UserID_index'].map(dict1)
nrecs['asin']=nrecs['ProductID_index'].map(dict2)
nrecs=nrecs.sort_values('reviewerID')
nrecs.reset_index(drop=True, inplace=True)
new=nrecs[['reviewerID','asin','Rating']]
new['recommendations'] = list(zip(new.asin, new.Rating))
res=new[['reviewerID','recommendations']]
res_new=res['recommendations'].groupby([res.reviewerID]).apply(list).reset_index()
review_df = spark.createDataFrame(res_new)
review_df.show(10)
##print(res_new)
