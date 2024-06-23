import time
import os
import resource
from pyspark.sql import SparkSession
import numpy as np
import gc
import psutil
import pickle
import pandas as pd
import tqdm

def preprocess_single_row(row, feature_names):
    row_df = pd.DataFrame([row], columns=feature_names)
    return row_df

def get_memory_usage():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss

# Загрузка модели
with open('/output/dt_model.pkl', 'rb') as file:
    dt_classifier = pickle.load(file)

# Создание Spark сессии
spark = SparkSession.builder \
    .appName("Forest Cover Type Prediction") \
    .getOrCreate()

time_vec = []
ram_vec = []

# Загрузка данных из HDFS
df = spark.read.csv("hdfs://namenode:9001/covtype.data", header=False)
pdf = df.toPandas()

# Присвоение имен столбцам
feature_names = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
                 "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
                 "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3",
                 "Wilderness_Area4", "Soil_Type1", "Soil_Type2", "Soil_Type3", "Soil_Type4", "Soil_Type5", "Soil_Type6",
                 "Soil_Type7", "Soil_Type8", "Soil_Type9", "Soil_Type10", "Soil_Type11", "Soil_Type12", "Soil_Type13",
                 "Soil_Type14", "Soil_Type15", "Soil_Type16", "Soil_Type17", "Soil_Type18", "Soil_Type19", "Soil_Type20",
                 "Soil_Type21", "Soil_Type22", "Soil_Type23", "Soil_Type24", "Soil_Type25", "Soil_Type26", "Soil_Type27",
                 "Soil_Type28", "Soil_Type29", "Soil_Type30", "Soil_Type31", "Soil_Type32", "Soil_Type33", "Soil_Type34",
                 "Soil_Type35", "Soil_Type36", "Soil_Type37", "Soil_Type38", "Soil_Type39", "Soil_Type40", "Cover_Type"]

pdf.columns = feature_names

batch_size = 512  # Размер батча

for iter in tqdm.tqdm(range(200)):
    start = time.time()
    initial_memory_usage = get_memory_usage()

    averaged_error = 0

    # Выбор случайных батчей данных
    sample_df = pdf.sample(n=batch_size, random_state=iter)

    for i in sample_df.index:
        data_loaded = sample_df.loc[i, feature_names[:-1]]
        preprocessed_data = preprocess_single_row(data_loaded, feature_names[:-1])
        predicted_cover_type = dt_classifier.predict(preprocessed_data)[0]
        real_cover_type = sample_df.loc[i, "Cover_Type"]
        if int(predicted_cover_type) != int(real_cover_type):
            averaged_error += 1

    accuracy = (batch_size - averaged_error) / batch_size
    end = time.time()
    time_vec.append(end - start)
    final_memory_usage = get_memory_usage()
    ram_vec.append(final_memory_usage - initial_memory_usage)

    df.unpersist()
    del sample_df
    gc.collect()

spark.stop()

with open('/output/single.csv', 'w') as file:
    file.write("Index,Execution Time (seconds),RAM Usage (B),Accuracy\n")
    for i in range(200):
        file.write(f"{i+1},{time_vec[i]},{ram_vec[i]},{accuracy}\n")