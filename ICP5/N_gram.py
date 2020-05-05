from __future__ import print_function
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram
from pyspark.sql import SparkSession

# Create the Spark session
spark = SparkSession.builder.appName("Ngrams").getOrCreate()

# Create the dataframe with five text abstracts
abstracts = spark.read.text('abs*.txt')

# Tokenize the abstract texts
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(abstracts)

# Creating n-grams with n=5
ngram = NGram(n=5, inputCol="words", outputCol="ngrams")
ngramDataFrame = ngram.transform(wordsData)

# Apply topic frequency on the abstracts
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=40)
featurizedData = hashingTF.transform(ngramDataFrame)

# Calculate the inverse document frequency
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# Display the results
rescaledData.select("features").show(20, truncate=False)

spark.stop()
