from pyspark.sql import SparkSession
from pyspark.ml.feature import NGram

# creating spark session
spark = SparkSession.builder .appName("Ngram Example").getOrCreate()

#creating dataframe of input
wordDataFrame = spark.createDataFrame([
    (0, ["Hi", "I", "heard", "about", "Spark"]),
    (1, ["I", "wish", "Java", "could", "use", "case", "classes"]),
    (2, ["Logistic", "regression", "models", "are", "neat"])
], ["id", "words"])

#creating NGrams with n=2 (two words)
ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
ngramDataFrame = ngram.transform(wordDataFrame)

# displaying the results
ngramDataFrame.select("ngrams").show(truncate=False)

#closing the spark session
spark.stop()