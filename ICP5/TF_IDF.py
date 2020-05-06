from __future__ import print_function
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession

# Create the Spark session
spark = SparkSession.builder.appName("TF_IDF").getOrCreate()

# Create the dataframe with five text abstracts
abstracts = spark.read.text('abs*.txt')

# Tokenize the abstract texts
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(abstracts)

# Apply topic frequency on the abstracts
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

# Calculate the inverse document frequency
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Display the results
rescaledData.select("features").show(truncate=False)

spark.stop()
