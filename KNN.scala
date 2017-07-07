import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
//import org.apache.spark.SparkContext
//import org.apache.spark.SparkContext._

object KNN {

  def main(args: Array[String]) {
    val train = "hw4-3/training_data_prob3_1.txt"
    //val conf = new SparkConf().setAppName("KNN (K-nearest-neighbour)")
    sqlTesting(train)
  }

  def sqlTesting(dataFile: String) {
    val spark = SparkSession.builder
      .appName("KNN (K-nearest-neighbour)")
      .getOrCreate()
  }
}
