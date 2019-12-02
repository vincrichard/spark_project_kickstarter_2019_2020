package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._


object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /****************************************
      * Charger un fichier csv dans un dataFrame
      *********************************************/

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv("data/train_clean.csv")

//    println(s"Nombre de lignes : ${df.count}")
//    println(s"Nombre de colonnes : ${df.columns.length}")
//    df.show()
//    df.printSchema()

    /******************************
      * Typage de nos données
      ***************************/

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline" , $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))


    //Remove a column which is not useful
    val df2: DataFrame = dfCasted.drop("disable_communication")
    //Remove column with values concerning the future result of the campaign
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")


    /***********************************************************
      * Cleanning des colonnes currency et country via des udf
      *********************************************************/
    val cleanCountryUdf = udf(cleanCountry _)
    val cleanCurrencyUdf = udf(cleanCurrency _)

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    // ou encore, en utilisant sql.functions.when:
//    import sql.functions.when
//    dfNoFutur
//      .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
//      .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
//      .drop("country", "currency")

    //Keep the meaningful final_status variable
    val dfFinalStatus = dfCountry
      .filter($"final_status" === 1 or $"final_status" === 0)


    /*************************************
      * Ajout et manipuation de colonnes
      *************************************/

    //Ajout d'une colone days_campaign contenant la durée de la campagne
    val createDaysCampaignUdf = udf(createDaysCampaign _)
    val dfDayCampaign : DataFrame = dfFinalStatus
      .withColumn("days_campaign", createDaysCampaignUdf($"launched_at", $"deadline"))

    //Ajout d'une colonne hours_prepa contenant le temps de preparation de la campagne
    val createHoursPrepaUdf = udf(createHoursPrepa _)
    val dfHoursPrepa : DataFrame = dfDayCampaign
      .withColumn("hours_prepa", round(createHoursPrepaUdf($"created_at", $"launched_at"), 3))
      .drop("launched_at", "created_at", "launched_at")

    //Modification des champs de texte
    val dfText = dfHoursPrepa
      .withColumn("name", lower($"name"))
      .withColumn("desc", lower($"desc"))
      .withColumn("keywords", lower($"keywords"))

    val dfConcatenate = dfText
      .withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))
      .drop("name", "desc", "keywords")

    //Remove null values
    val dfClean = dfConcatenate
      .na.fill(-1, Array("days_campaign", "hours_prepa", "goal"))
      .na.fill("unknown", Array("country2", "currency2"))
    //Remove hours_prepa < 0
      .filter($"hours_prepa" >0)

    /**********************************
      * Exportation du Dataframe obtenu
      **********************************/
    dfClean.write.mode(SaveMode.Overwrite).parquet("data/out/")
  }

  /**
    * Methode utilisé pour Clean la colonne county
    */
  def cleanCountry(country: String, currency: String): String = {
    if (currency != null && (country == "False" || country.length != 2))
      null
    else
      country
  }

  /**
    * Methode utilisé pour Clean la colonne currency
    */
  def cleanCurrency(currency: String): String = {
    if (currency != null && currency.length != 3)
      null
    else
      currency
  }

  /**
    * Methode utilisé pour créer la colonne days_campaign qui représente la durée de la campagne en jours
    */
  def createDaysCampaign(launched_at: Int, deadline: Int): Int = {
    return (deadline - launched_at) / 86400
  }

  /**
    * Methode utilisé pour créer la colonne hours_prepa qui représente le nombre d’heures de préparation de la campagne
    */
  def createHoursPrepa(created_at: Int, launched_at: Int): Double = {
    return (launched_at.toFloat - created_at.toFloat) / 3600.0
  }

}
