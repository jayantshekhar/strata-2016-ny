/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package org.apache.spark

// $example on$

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature._
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types._

// $example off$
import org.apache.spark.sql.SparkSession


object Spam {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("Spam")
      .master("local")
      .getOrCreate()
    import spark.implicits._

    val customSchema = StructType(Array(
      StructField("spam", StringType, true),
      StructField("message", StringType, true)

    ))

    val ds = spark.read.option("inferSchema", "true").option("delimiter", "\t").schema(customSchema).csv("data/SMSSpamCollection.tsv")

    ds.printSchema()

    ds.show(8)

    // label
    val indexer = new StringIndexer()
      .setInputCol("spam")
      .setOutputCol("label")
    val indexed = indexer.fit(ds).transform(ds)

    indexed.show()

    // tokenize
    val tokenizer = new Tokenizer().setInputCol("message").setOutputCol("tokens")
    val tokdata = tokenizer.transform(indexed)

    tokdata.show()

    // tf
    val hashingTF = new HashingTF()
      .setInputCol("tokens").setOutputCol("tf").setNumFeatures(20)
    val tfdata = hashingTF.transform(tokdata)

    tfdata.show()

    // idf
    val idf = new IDF().setInputCol("tf").setOutputCol("idf")
    val idfModel = idf.fit(tfdata)
    val idfdata = idfModel.transform(tfdata)

    val assembler = new VectorAssembler()
      .setInputCols(Array("idf"))
      .setOutputCol("features")

    val assemdata = assembler.transform(idfdata)

    // split
    val Array(trainingData, testData) = assemdata.randomSplit(Array(0.7, 0.3))

    // lr
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol("label")
      .setFeaturesCol("features")


    // Fit the model
    val lrModel = lr.fit(assemdata)

    val str = lrModel.toString()
    println(str)

    // predict
    val predict = lrModel.transform(assemdata)


    predict.show(100)

    spark.stop()
  }
}
// scalastyle:on println


