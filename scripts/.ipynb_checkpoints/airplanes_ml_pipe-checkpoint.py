from pyspark.sql.functions import col,count,when
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer,OneHotEncoder,VectorAssembler,StandardScaler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline,PipelineModel

import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class airplanes_ml_pipe:
    def __init__(self,spark: SparkSession):
        self.spark = spark

    def quality_check(self,df):
        try:
            logger.info("--checking for dataset quality--")

            # count null/empty values
            null_counts = df.select([count(when(col(c).isNull(),c)).alias(c) \
                          for c in df.columns]).collect()[0]

            logger.info("--dataset quality--")
            logger.info(f"total count: {df.count()}")
            logger.info(f"# of null entries: {len(null_counts)}")

            delay_dist = df.groupBy(col("DELAY_CAT")).count().collect()
            logger.info(f"class distribution: {delay_dist}")

            logger.info("--quality check complete--")
            return True
        except Exception as e:
            logger.error("--quality check failed--")
            return False

    def prepare_features(self,df):
        try:
            logger.info("--preparing features--")

            # convert categorical string data to unique integer indexes
            string_indexers = [StringIndexer(inputCol=col,outputCol=f"{col}_IDX",handleInvalid="keep") \
                             for col in ["OP_CARRIER","ORIGIN","DEST","DELAY_CAT"]]

            # one-hot encodings for categorical data
            one_hot_encoders = [OneHotEncoder(inputCol=f"{col}_IDX",outputCol=f"{col}_ENC") \
                                 for col in ["OP_CARRIER","ORIGIN","DEST"]]
            '''
                I one-hot encoded the categorical numeric features so they are not mistaken as ordinal data.
                Thus, eliminating any false ordinal relationships with binary vectors.
            '''
            
            # binary features
            binary_cols = ["IS_WEEKEND","IS_MORN_FLIGHT","IS_AFTERN_FLIGHT","IS_NI_FLIGHT"]

            # Numeric features
            numeric_cols = [
                "DEP_HOUR", "DAY_OF_WEEK", "DAILY_FLIGHTS",
                "AVERAGE_CARRIER_DELAY", "AVERAGE_ROUTE_DELAY", "DELAY_RISK_SCORE",
                "WEATHER_DELAY"
            ]

            feature_cols = binary_cols + numeric_cols + \
            ["OP_CARRIER_ENC","ORIGIN_ENC","DEST_ENC"]

            # Collapse all features into a single column vector, "features"
            assembler = VectorAssembler(
                            inputCols=feature_cols,
                            outputCol="features"
                            )

            # Normalize the features (sets SD to 1 and subtracts the mean to center the data[mean=0])
            scaler = StandardScaler(
                     inputCol="features",
                     outputCol="scaled_features",
                     withStd=True,
                     withMean=True
                   )

            logger.info("--features prepared successfully--")
            return string_indexers,one_hot_encoders,assembler,scaler
        except Exception as e:
            logger.error("--could not ready features--")
            raise

    def create_model(self):
        try:
            rf = RandomForestClassifier(
                    labelCol="DELAY_CAT_IDX",
                    featuresCol="scaled_features",
                    numTrees=100,
                    maxDepth=10
                    )
            logger.info("--model created successfully--")
            return rf
        except Exception as e:
            logger.error("--model could not be created--")
            raise

    def train_and_evaluate(self,df):
        try:
            train_df,test_df = df.randomSplit([0.8,0.2],seed=42)

            indexers,encoders,assembler,scaler = self.prepare_features(train_df)
            classifier = self.create_model()

            pipe = Pipeline(stages=indexers+encoders+[assembler,scaler,classifier])
            logger.info("--pipeline created successfully; generating PipelineModel--")
            model = pipe.fit(train_df)

            predictions = model.transform(test_df)

            evaluator = MulticlassClassificationEvaluator(
                labelCol="DELAY_CAT_IDX",
                predictionCol="prediction"
                )

            # evaluate model
            logger.info("--evaluating--")
            metrics={}
            for metric in ["f1","weightedPrecision","weightedRecall","accuracy"]:
                evaluator.setMetricName(metric)
                metrics[metric]=evaluator.evaluate(predictions)
            logger.info(f"Performance Metrics: {metrics}")
            
            
            return model,metrics            
        except Exception as e:
            logger.error("--model training failed--")
            raise

def main():
    try:
        logger.info("--loading data--")
        path="/home/jovyan/work/data"
        fname="processed_data.csv"
        number_of_records=500
        spark=SparkSession.builder.appName("AirplaneDataAnalyzer").getOrCreate()
        processed_df=spark.read.csv(f"{path}/processed/{fname}",header=True,inferSchema=True) \
        .limit(number_of_records)
        logger.info("--dataset loaded successfully--")
        processed_df = processed_df.na.drop(subset=["DELAY_CAT"]) # ensure no null values in label column

        ml_pipe=airplanes_ml_pipe(spark)

        logger.info("--starting--")
        if ml_pipe.quality_check(processed_df):
            model,metrics=ml_pipe.train_and_evaluate(processed_df)
            model.write().overwrite().save(f"{path}/models/delay_predictor/")
            logger.info("--model saved successfully--")
    except Exception as e:
        logger.error(f"--failed: {e}--")
        return
if __name__=="__main__":
    main()
            