import org.apache.spark.SparkContext._
import scala.io._
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._
import org.apache.log4j.Logger
import org.apache.log4j.Level
import scala.collection._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression



object volatility {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("StockVolatility")
      .master("local[*]")
      .getOrCreate()

    // Load the CSV file into a DataFrame
    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/sp500_stocks.csv")

    // Select relevant columns and cast them to the correct data types
    val df = data.select(
      col("Date").cast("date"),
      col("Symbol"),
      col("High").cast("double"),
      col("Low").cast("double")
    )

    // Calculate daily volatility for each stock
    val dailyVolatility = df.withColumn("DailyVolatility", col("High") - col("Low"))

    // Calculate overall volatility for each stock
    val stockVolatility = dailyVolatility.groupBy("Symbol")
      .agg(stddev("DailyVolatility").alias("OverallVolatility"))
      .orderBy(desc("OverallVolatility"))

    // Find the top 10 most volatile stocks
    val top10VolatileStocks = stockVolatility.limit(10)
    top10VolatileStocks.show()

    val leastVolatileStocks = dailyVolatility.groupBy("Symbol")
      .agg(stddev("DailyVolatility").alias("OverallVolatility"))
      .orderBy(asc("OverallVolatility"))

    // Find the top 10 most volatile stocks
    val bot10VolatileStocks = leastVolatileStocks.limit(10)
    bot10VolatileStocks.show()

    // Calculate daily volatility for each stock
    val dailyVolatilitySum = dailyVolatility.groupBy("Date")
      .agg(sum("DailyVolatility").alias("TotalDailyVolatility"))
      .orderBy(desc("TotalDailyVolatility"))

    // Find the 10 days with the most stock volatility
    val top10VolatileDays = dailyVolatilitySum.limit(10)
    top10VolatileDays.show()
  }
}
