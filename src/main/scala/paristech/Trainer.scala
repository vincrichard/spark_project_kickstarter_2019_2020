package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.tuning._
import org.apache.spark.ml.{Pipeline, PipelineModel}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val dfClean : DataFrame = spark.read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .parquet("/media/vincent/C0FC3B20FC3B0FE0/MSBGD/Spark/TP/cours-spark-telecom/data/prepared_trainingset")

    //Stage 1 : Récupérer les mots
    val tokenizer : RegexTokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    //Stage 2 : Retirer les stops words
    val stopWordsRemover : StopWordsRemover = new StopWordsRemover()
      .setCaseSensitive(true)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("meaningful_tokens")

    //Stage 3 : Computer la partie TF
    val countVectorizer: CountVectorizer = new CountVectorizer()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("vector")

    //Stage 4 : Computer la partie IDF
    val idf : IDF = new IDF()
      .setInputCol(countVectorizer.getOutputCol)
      .setOutputCol("tfidf")

    //Stage 5 et 6: Index value of country2 and currency2
    val indexerCountry : StringIndexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")

    val indexerCurrency : StringIndexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")

    //Stage 7 : One-Hot encoder
    val encoder : OneHotEncoderEstimator = new OneHotEncoderEstimator()
      .setInputCols(Array(indexerCountry.getOutputCol, indexerCurrency.getOutputCol))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    //Stage 8 et 9: Assemblage de tous les features en un vecteur
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    //Stage 10 : Créer / instancier le modèle de classification
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(20)

    //Création du Pipeline
    val pipeline = new Pipeline()
      .setStages(Array(
        tokenizer,
        stopWordsRemover,
        countVectorizer,
        idf,
        indexerCountry,
        indexerCurrency,
        encoder,
        assembler,
        lr))

    //Split des données en partition 90% train / 10% test
    val Array(training, test) = dfClean.randomSplit(Array(0.9, 0.1), seed = 12345)

    //Entrainement du model
    val model: PipelineModel = pipeline.fit(training)

    //Test du model
    val dfWithSimplePredictions : DataFrame = model.transform(test)

    //Afficher le f1 score
    val f1Evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    println("f1 score : " + f1Evaluator.evaluate(dfWithSimplePredictions))
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

  }
}
