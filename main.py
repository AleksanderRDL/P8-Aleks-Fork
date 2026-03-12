import os
import sys
import time
import platform
from pathlib import Path

import removeOutliers
import removeDuplications
import removeShiptypes
import ship_type
import trim_moving
import trimStationary

from hadoop_environment import configure_hadoop_environment
from java_environment import configure_java_environment
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_DIR = Path(__file__).resolve().parent
AISDATA_DIR = PROJECT_DIR / "AISDATA"
INPUT_FILE = AISDATA_DIR / "aisdk-2026-02-05.csv"
OUTPUT_PATH = AISDATA_DIR / "aisdk-2026-02-05.cleaned.csv"

# THIS WILL PROBABLY BE REMOVED OR MOVED INTO ITS OWN SETUP CONFIGURATION FILE
# Ensure Spark workers use the same Python interpreter (with pandas/pyarrow)
# On Windows, paths with spaces break PySpark worker launches, so use the
# short (8.3) path which contains no spaces.
if platform.system() == "Windows":
    import ctypes
    buf = ctypes.create_unicode_buffer(260)
    if ctypes.windll.kernel32.GetShortPathNameW(sys.executable, buf, 260):
        python_exec = buf.value

def configure_pyspark_python() -> None:
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

configure_java_environment(PROJECT_DIR)
configure_hadoop_environment(PROJECT_DIR)
configure_pyspark_python()

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

start_time = time.time()

spark = (SparkSession.builder
    .config("spark.local.dir", os.path.join(os.path.dirname(os.path.abspath(__file__)), "spark_temp"))
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate())

df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(str(INPUT_FILE))

timestamp_col = "# Timestamp"  # adjust if your column name differs
df = df.withColumn(timestamp_col, F.to_timestamp(F.col(timestamp_col), "dd/MM/yyyy HH:mm:ss"))


df = removeDuplications.deduplicate_and_filter(df)
df = trimStationary.trim_stationary(df)
df = ship_type.fill_ship_type(df)
df = ship_type.remove_undefined_ship_type(df)
df = removeShiptypes.remove_shiptypes(df)
df = removeOutliers.remove_gps_outliers(df)
df = trim_moving.trim_moving(df)

df.coalesce(1).write.format("csv").option("header", "true").mode("overwrite").save(str(OUTPUT_PATH))

elapsed_time = time.time() - start_time

print("elapsed_time:", elapsed_time)
if os.environ.get("PRINT_ROW_COUNT", "0") == "1":
    print("Count of rows after processing:", df.count())

spark.stop()
