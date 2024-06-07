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

object VolatilityBySector {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("StockVolatility")
      .master("local[*]")
      .getOrCreate()

    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/sp500_stocks.csv")

    // Select columns and cast them to the correct data types
    val df = data.select(
      col("Date").cast("date"),
      col("Symbol"),
      col("Sector"),
      col("High").cast("double"),
      col("Low").cast("double"),
      col("Close").cast("double")
    )

    // Calculate daily volatility for each stock
    val dailyVolatility = df.withColumn("DailyVolatility", (col("High") - col("Low")) / col("Close") * 100)

    // Calculate overall volatility for each stock
    val stockVolatility = dailyVolatility.groupBy("Symbol", "Sector")
      .agg(stddev("DailyVolatility").alias("OverallVolatility"))

    // Calculate average volatility for each sector
    val sectorVolatility = stockVolatility.groupBy("Sector")
      .agg(avg("OverallVolatility").alias("AvgSectorVolatility"))
      .orderBy(desc("AvgSectorVolatility"))

    // Find the top 10 highest volatility sectors
    val top10VolatilitySectors = sectorVolatility.limit(10)
    top10VolatilitySectors.show()

    // Find the top 10 lowest volatility sectors
    val lowest10VolatilitySectors = sectorVolatility.orderBy(asc("AvgSectorVolatility")).limit(10)
    lowest10VolatilitySectors.show()

    spark.stop()
  }
}
