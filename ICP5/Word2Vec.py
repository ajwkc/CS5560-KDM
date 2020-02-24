from pyspark.sql import SparkSession
from pyspark.ml.feature import Word2Vec

# creating spark session
spark = SparkSession.builder .appName("Ngram Example").getOrCreate()


# Input data: Each row is a bag of words from a sentence or document.
documentDF = spark.createDataFrame([
    ("McCarthy was asked to analyse the data from the first phase of trials of the vaccine.".split(" "), ),
    ("We have amassed the raw data and are about to begin analysing it.".split(" "), ),
    ("Without more data we cannot make a meaningful comparison of the two systems.".split(" "), ),
    ("Collecting data is a painfully slow process.".split(" "), ),
    ("You need a long series of data to be able to discern such a trend.".split(" "), )
], ["text"])


# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="text", outputCol="result")
model = word2Vec.fit(documentDF)
print(model.getVectors().collect())
result = model.getVectors().collect()


# showing the synonyms and cosine similarity of the word in input data
synonyms = model.findSynonyms("data", 5)   # its okay for certain words , real bad for others
synonyms.show(5)


#closing the spark session
spark.stop()