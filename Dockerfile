FROM bitnami/spark:latest

USER root
RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install streamlit
RUN pip3 install pandas
RUN pip3 install scikit-learn==1.3.2
RUN pip3 install psutil
RUN pip3 install tqdm

USER 1001

COPY ./ram_time_values.csv /output/ram_time_values.csv
COPY ./dt_model.pkl /output/dt_model.pkl
COPY ./parallel.py /scripts/parallel.py
COPY ./single.py /scripts/single.py

RUN chmod -R 777 /output

# Установка Spark
ENV SPARK_HOME /opt/bitnami/spark
ENV PATH $SPARK_HOME/bin:$PATH

CMD ["bin/spark-class", "org.apache.spark.deploy.master.Master"]
