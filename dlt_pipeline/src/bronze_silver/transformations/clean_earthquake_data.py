from pyspark.sql.types import *
from pyspark.sql import functions as F

import dlt


catalog_name = spark.conf.get('catalog_name')
volume_path = f'/Volumes/{catalog_name}/bronze/earthquake_data'
primary_key = 'id'


properties_schema = StructType([
    StructField("mag", StringType(), True),
    StructField("place", StringType(), True),
    StructField("time", StringType(), True),
    StructField("updated", StringType(), True),
    StructField("tz", StringType(), True),
    StructField("url", StringType(), True),
    StructField("detail", StringType(), True),
    StructField("felt", StringType(), True),
    StructField("cdi", StringType(), True),
    StructField("mmi", StringType(), True),
    StructField("alert", StringType(), True),
    StructField("status", StringType(), True),
    StructField("tsunami", StringType(), True),
    StructField("sig", StringType(), True),
    StructField("net", StringType(), True),
    StructField("code", StringType(), True),
    StructField("ids", StringType(), True),
    StructField("sources", StringType(), True),
    StructField("types", StringType(), True),
    StructField("nst", StringType(), True),
    StructField("dmin", StringType(), True),
    StructField("rms", StringType(), True),
    StructField("gap", StringType(), True),
    StructField("magType", StringType(), True),
    StructField("type", StringType(), True),
    StructField("title", StringType(), True)
])

geometry_schema = StructType([
    StructField("type", StringType(), True),
    StructField("coordinates", ArrayType(DoubleType()), True)
])

feature_schema = StructType([
    StructField("type", StringType(), True),
    StructField("properties", properties_schema, True),
    StructField("geometry", geometry_schema, True),
    StructField("id", StringType(), True)
])

schema = ArrayType(feature_schema)



@dlt.view(name = 'earthquake_view')
def earthquake_view():
    df = spark.readStream\
            .format('cloudFiles')\
            .option('cloudFiles.format', 'json')\
            .load(volume_path )
    
    df = df.withColumn(
        'parsed_data',
        F.from_json(F.col('features'), schema)).withColumn(
        '_loadtime', F.current_timestamp() )
    
    df = df.withColumn('feature', F.explode('parsed_data'))
    df = df.select(
        'feature.properties.*',
        'feature.id',
        F.col('feature.geometry.coordinates')[0].alias('longitude'),
        F.col('feature.geometry.coordinates')[1].alias('latitude'),
        F.col('feature.geometry.coordinates')[2].alias('depth'),
        '_loadtime')
    
    df = df.withColumn(
        'time',
        F.from_utc_timestamp(
            F.from_unixtime(F.col('time') / 1000),
            'Asia/Kolkata'
        ).cast('timestamp')
    ).withColumn(
        'mag', F.col('mag').cast('double')
    ).withColumn(
        'updated',
        F.from_utc_timestamp(
            F.from_unixtime(F.col('updated') / 1000),
            'Asia/Kolkata'
        ).cast('timestamp')
    ).withColumn(
        'nst', F.col('nst').cast('double')
    ).withColumn(
        'dmin', F.col('dmin').cast('double')
    ).withColumn(
        'rms', F.col('rms').cast('double')
    ).withColumn(
        'gap', F.col('gap').cast('double')
    )
            

    return df

dlt.create_streaming_table('earthquake_data')
dlt.apply_changes(
    target = 'earthquake_data',
    source = 'earthquake_view',
    keys = [primary_key],
    sequence_by = '_loadtime',
    stored_as_scd_type = '1'

)

