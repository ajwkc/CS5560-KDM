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

    # Alias for SQLContext
    sqlContext = SQLContext(sc)

    # Load the file for queries
    file = sqlContext.read.csv(SparkFiles.get('/home/andrew/CS5560-KDM/ICP4/data.csv'), header=True, inferSchema=True)

    # Transformation: a new RDD containing only female customers
    fCust = file.filter(file['gender'] == 'Female')

    # Transformation: a new RDD containing only senior citizen customers
    ssCust = file.filter(file['SeniorCitizen'] == '1')

    # Transformation: an intersection of these two new RDDs
    fssCust = fCust.intersect(ssCust)

    # Action: count the number of female customers
    print('There are {} female customers'.format(fCust.count()))

    # Action: count the number of female senior citizen customers
    print('There are {} female senior citizen customers'.format(fssCust.count()))

    # Action: show the average monthly charge for each gender of senior citizens
    ssCust.groupBy('gender').agg({'MonthlyCharges': 'avg', 'Gender': 'count'}).show()

    # Action: print an RDD's schema
    file.printSchema()

    # Action: show the first 10 entries in an RDD
    fssCust.show(10)

    # Action: group customers by contract type
    file.groupBy('Contract').count().show()

    # Action: show count, avg, stdev, min & max of each column
    file.describe().show()


if __name__ == '__main__':
    main()
