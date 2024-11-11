from pyspark.sql import SparkSession,Window
from pyspark.sql.functions import (
    col,lag,concat,to_timestamp,hour,lit,count,
    avg,when,coalesce,dayofweek,year,month,lpad
)
import logging

logging.basicConfig(level=logging.INFO,format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

class airplanes_data_pipe:
    def __init__(self,spark: SparkSession):
        self.spark = spark

    def load_data(self,path:str):
        try:
            logger.info("--loading dataset--")
            df = self.spark.read.csv(path,header=True,inferSchema=True)
            logger.info("--dataset loaded successfully--")
            return True,df
        except Exception as e:
            logger.error(f"--dataset could not be loaded: {e}--")
            return False,None
    
    def generate_temporal_features(self,df):
        logger.info("--generating temporal features--")

        try:
            # add flight_date column
            df = df.withColumn("FLIGHT_DATE", \
            to_timestamp(col("FL_DATE"),\
                        "dd-MM-yyyy"))

            df = df.withColumn("CRS_DEP_TIME_STR",lpad(col("CRS_DEP_TIME").cast("string"),4,"0")) \
                .withColumn("SCHEDULED_DEPARTURE_TIME",to_timestamp(col("CRS_DEP_TIME_STR"),"HHmm")) \
                .withColumn("DEP_HOUR",hour(col("SCHEDULED_DEPARTURE_TIME"))) \
                .withColumn("DAY_OF_WEEK",dayofweek(col("FLIGHT_DATE"))) \
                .withColumn("MONTH",month(col("FLIGHT_DATE"))) \
                .withColumn("IS_WEEKEND",when(col("DAY_OF_WEEK").isin([6,7]),1).otherwise(0)) \
                .withColumn("IS_MORN_FLIGHT",when(col("DEP_HOUR").between(6,12),1).otherwise(0)) \
                .withColumn("IS_AFTERN_FLIGHT",when(col("DEP_HOUR").between(12,18),1).otherwise(0)) \
                .withColumn("IS_NI_FLIGHT",when(col("DEP_HOUR") > 18,1).otherwise(0))

            logger.info("--successful--")
            return df
        except Exception as e:
            logger.error("--could not generate time features--")
            raise

    def generate_airline_features(self,df):
        logger.info("--generating airline features--")

        try:
            airline_win = Window.partitionBy("OP_CARRIER")
            route_win = Window.partitionBy("OP_CARRIER","ORIGIN","DEST")
            date_win = Window.partitionBy("ORIGIN","FLIGHT_DATE").orderBy("FLIGHT_DATE")

            df = df.withColumn("AVERAGE_CARRIER_DELAY",avg("DEP_DELAY").over(airline_win)) \
                 .withColumn("AVERAGE_ROUTE_DELAY",avg("DEP_DELAY").over(route_win)) \
                 .withColumn("PREV_DAY_DEPARTURE_DELAY",lag("DEP_DELAY",1).over(date_win))
            '''
                AVERAGE_CARRIER_DELAY : departure delay time grouped by airline company
                AVERAGE_ROUTE_DELAY : departure delay time grouped by airline route(airline,origin,destination)
                PREV_DAY_DEPARTURE_DELAY : departure delay of previous day, first record is null
            '''

            # route congestion features
            df = df.withColumn("DAILY_FLIGHTS",count("*").over(Window.partitionBy("FLIGHT_DATE","ORIGIN"))) \
                 .withColumn("ROUTE_FREQ",count("*").over(route_win))

            logger.info("--successful--")
            return df
        except Exception as e:
            logger.error("--could not generate airline features--")
            raise

    def generate_delay_features(self,df):
        logger.info("--generating delay features--")
        try:

            # define delay categories
            df = df.withColumn("DELAY_CAT",when(col("DEP_DELAY") <= 0,"On time") \
                               .when(col("DEP_DELAY") <= 15,"Slight Delay") \
                               .when(col("DEP_DELAY") <= 45,"Major Delay") \
                               .otherwise("Severe Delay"))

            # define a weighted delay risk score
            df = df.withColumn("DELAY_RISK_SCORE",col("AVERAGE_CARRIER_DELAY")*0.2 + \
                               coalesce(col("WEATHER_DELAY"),lit(0))*0.2 + \
                               coalesce(col("PREV_DAY_DEPARTURE_DELAY"),lit(0))*0.2 + \
                               col("AVERAGE_ROUTE_DELAY")*0.2+
                               (col("DAILY_FLIGHTS")/100)*0.2)
            
            logger.info("--successful--")
            return df
        except Exception as e:
            logger.error("--could not generate delay features--")
            raise

    def pipe_driver(self,df):
        logger.info("--starting data processing pipeline--")

        try:
            # 1. Initial Cleaning
            df = df.na.fill({
                "DEP_DELAY": 0,
                "ARR_DELAY": 0,
                "CANCELLED": 0,
                "WEATHER_DELAY": 0,
                "NAS_DELAY": 0,
                "SECURITY_DELAY": 0,
                "CARRIER_DELAY": 0,
                "LATE_AIRCRAFT_DELAY": 0
            })

            # 2. Feature Extraction
            df = self.generate_temporal_features(df)
            df = self.generate_airline_features(df)
            df = self.generate_delay_features(df)

            # 3. Select Final Feature Set
            final_df = df.select(
                "FLIGHT_DATE", "OP_CARRIER", "ORIGIN", "DEST",
                "DEP_HOUR", "DAY_OF_WEEK", "MONTH",
                "IS_WEEKEND", "IS_MORN_FLIGHT", "IS_AFTERN_FLIGHT", "IS_NI_FLIGHT",
                "AVERAGE_CARRIER_DELAY", "AVERAGE_ROUTE_DELAY", "DAILY_FLIGHTS",
                "WEATHER_DELAY",
                "DELAY_CAT", "DELAY_RISK_SCORE", "DEP_DELAY"
            )

            logger.info("--dataset has been processed successfully--")
            return True,final_df
        except Exception as e:
            logger.error(f"--processing pipeline failed: {e}--")
            return False,None

    def save_dataset(self,df,dest_path: str):
        logger.info("--saving dataset--")

        try:
            df.coalesce(1).write.csv(dest_path,mode="overwrite",header=True)
            logger.info(f"--data saved successfully at {dest_path}")
            return True
        except Exception as e:
            logger.error("--failed--")
            return False

def main():
    path="/home/jovyan/work/data"
    dataset="2018.csv"
    spark=SparkSession.builder.appName("AirlineDataProcessor").getOrCreate()
    data_pipe=airplanes_data_pipe(spark)
    status,df=data_pipe.load_data(f"{path}/raw/{dataset}")
    if not status:
        return

    status,processed_df=data_pipe.pipe_driver(df)
    if not status:
        return

    status = data_pipe.save_dataset(df=processed_df,dest_path=f"{path}/processed/")  
    if not status:
        return
        
if __name__=="__main__":
    main()
        