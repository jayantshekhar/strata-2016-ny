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

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types._

// $example off$
import org.apache.spark.sql.SparkSession


object KMeans {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("KMeans")
      .master("local")
      .getOrCreate()
    import spark.implicits._

    val customSchema = StructType(Array(
      StructField("symboling", StringType, true),
      StructField("normalized-losses", StringType, true),
      StructField("make", StringType, true),

      StructField("fuel-type", StringType, true),
      StructField("aspiration", StringType, true),

      StructField("num-of-doors", StringType, true),
      StructField("body-style", StringType, true),

      StructField("drive-wheels", StringType, true),
      StructField("engine-location", StringType, true),

      StructField("wheel-base", StringType, true),
      StructField("length", StringType, true),
      StructField("width", StringType, true),
      StructField("height", StringType, true),
      StructField("curb-weight", StringType, true),

      StructField("engine-type", StringType, true),
      StructField("num-of-cylinders", StringType, true),
      StructField("engine-size", StringType, true),
      StructField("fuel-system", StringType, true),
      StructField("bore", StringType, true),
      StructField("stroke", StringType, true),

      StructField("compression-ratio", StringType, true),

      StructField("horsepower", StringType, true),
      StructField("peak-rpm", StringType, true),
      StructField("city-mpg", StringType, true),
      StructField("highway-mpg", StringType, true),
      StructField("price", StringType, true)

    ))

    val ds = spark.read.option("inferSchema", "true").schema(customSchema).csv("data/imports-85.data")

    ds.printSchema()

    ds.show()

    spark.stop()
  }
}
// scalastyle:on println


