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



object stocks {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val spark = SparkSession.builder
      .appName("StockPrediction")
      .master("local[*]")
      .getOrCreate()

    val data = spark.read.option("header", "true").option("inferSchema", "true").csv("src/sp500_stocks.csv")
    val df = data.select(
      col("Date").cast("date"),
      col("Open").cast("double"),
      col("High").cast("double"),
      col("Low").cast("double"),
      col("Close").cast("double"),
      col("Volume").cast("double")
    )
    // Remove rows with null values
    val dfWithoutNulls = df.na.drop()
    // Create a new column for the label (e.g., predicting the closing price)
    val dfWithLabel = df.withColumn("label", col("Close"))

    // Assemble features into a feature vector
    val assembler = new VectorAssembler()
      .setInputCols(Array("Open", "High", "Low", "Volume"))
      .setOutputCol("features")
      .setHandleInvalid("skip")

    val output = assembler.transform(dfWithLabel).select("features", "label")

    val Array(trainingData, testData) = output.randomSplit(Array(0.7, 0.3))

    val lr = new LinearRegression()

    // Fit the model
    val lrModel = lr.fit(trainingData)

    // Make predictions
    val predictions = lrModel.transform(testData)

    // Evaluate the model
    import org.apache.spark.ml.evaluation.RegressionEvaluator

    val evaluator = new RegressionEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    // Show some sample predictions
    predictions.select("prediction", "label", "features").show(5)
    val summaryStats = dfWithLabel.describe("Close")
    summaryStats.show()
    val residuals = predictions.withColumn("residual", col("label") - col("prediction"))
    residuals.select("label", "prediction", "residual").show(5)
  }
}
