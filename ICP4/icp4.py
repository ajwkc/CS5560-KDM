import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkFiles


def main():
    # Alias for SparkContext
    sc = pyspark.SparkContext()

    # Iterate through the file and count each word
    file = sc.textFile('/home/andrew/CS5560-KDM/ICP4/input.txt')
    counts = file.flatMap(lambda line: line.split(' ')).map(lambda word: (word, 1)).reduceByKey(lambda a, b: a + b)

    # Print the word count results
    for x in counts.collect():
        print(x)


if __name__ == '__main__':
    main()
