from __future__ import print_function
from pyspark.ml.feature import Word2Vec
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram
from pyspark.sql import SparkSession

# Create the Spark session
spark = SparkSession.builder.appName("W2V_Easy").getOrCreate()

# Create the dataframe with five text abstracts
abstracts = spark.read.text('abs*.txt')

# Tokenize the abstract texts
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(abstracts)

# Create a mapping from words to vectors
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="words", outputCol="result")
model = word2Vec.fit(wordsData)
print(model.getVectors().collect())
result = model.getVectors().collect()

# Show the synonyms and cosine similarity of the word in input data
synonyms = model.findSynonyms("physics", 5)
synonyms.show(5)

# Close the session
spark.stop()