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
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.Pipeline

object industries {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("StockVolatility")
      .master("local[*]")
      .getOrCreate()

    val sectors = spark.read.option("header", "true").option("inferSchema", "true").csv("src/sp500_companies.csv")
    val stocks = spark.read.option("header", "true").option("inferSchema", "true").csv("src/sp500_stocks.csv")

    // Select columns and cast them to the correct data types
    val df1 = sectors.select(
      col("Symbol").cast("String"),
      col("Fulltimeemployees").cast("double")
    )

    val df2 = stocks.select(
      col("Date").cast("date"),
      col("High").cast("double"),
      col("Low").cast("double"),
      col("Close").cast("double"),
      col("Symbol").cast("String")
    )

    val joinedDF = df2.join(df1, "Symbol")
    val cleanDF = joinedDF.na.drop(Seq("Fulltimeemployees"))

    // Calculate daily volatility for each stock
    val dailyVolatility = cleanDF.withColumn("DailyVolatility", (col("High") - col("Low")) / col("Close") * 100)

    // Calculate overall volatility for each stock
    val stockVolatility = dailyVolatility.groupBy("Symbol", "Fulltimeemployees")
      .agg(stddev("DailyVolatility").alias("OverallVolatility"))

    // Prepare the data for linear regression
    val assembler = new VectorAssembler()
      .setInputCols(Array("Fulltimeemployees"))
      .setOutputCol("features")

    val assembledData = assembler.transform(stockVolatility)

    // Define the linear regression model
    val lr = new LinearRegression()
      .setLabelCol("OverallVolatility")
      .setFeaturesCol("features")

    // Create a pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, lr))

    // Fit the model
    val model = pipeline.fit(stockVolatility)

    // Print the coefficients and intercept for linear regression
    val lrModel = model.stages.last.asInstanceOf[LinearRegressionModel]
    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

    // Summarize the model over the training set and print out some metrics
    val trainingSummary = lrModel.summary
    println(s"numIterations: ${trainingSummary.totalIterations}")
    println(s"objectiveHistory: ${trainingSummary.objectiveHistory.toList}")
    trainingSummary.residuals.show()
    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
    println(s"r2: ${trainingSummary.r2}")
    println(s"T-values: ${trainingSummary.tValues.mkString(", ")}")
    println(s"P-values: ${trainingSummary.pValues.mkString(", ")}")

    spark.stop()
  }
}
