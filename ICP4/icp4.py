import pyspark
from pyspark.sql import SQLContext
from pyspark import SparkFiles


sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

df = sqlContext.read.csv(SparkFiles.get("/home/andrew/CS5560-KDM/ICP4/data.csv"), header=True, inferSchema= True)
df.printSchema()
#df.show(5, truncate = False)

#If you didn't set inderShema to True, here is what is happening to the type. There are all in string.
#df_string = sqlContext.read.csv(SparkFiles.get("C:/Users/shs6g/Desktop/KDM/code/ICP4_V2/data.csv"), header=True, inferSchema=  False)
#df_string.printSchema()


#You can select and show the rows with select and the names of the features. Below, gender and churn are selected.
#df.select('gender','churn').show(5)

#To get a summary statistics, of the data, you can use describe(). It will compute the :count, mean, standarddeviation, min, max
#df.describe().show()