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
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.types._

// $example off$
import org.apache.spark.sql.SparkSession


object Churn {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("ALSExample")
      .master("local")
      .getOrCreate()
    import spark.implicits._

    val customSchema = StructType(Array(
      StructField("state", StringType, true),
      StructField("account_length", DoubleType, true),
      StructField("area_code", StringType, true),

      StructField("phone_number", StringType, true),
      StructField("intl_plan", StringType, true),

      StructField("voice_mail_plan", StringType, true),
      StructField("number_vmail_messages", DoubleType, true),

      StructField("total_day_minutes", DoubleType, true),
      StructField("total_day_calls", DoubleType, true),

      StructField("total_day_charge", DoubleType, true),
      StructField("total_eve_minutes", DoubleType, true),
      StructField("total_eve_calls", DoubleType, true),
      StructField("total_eve_charge", DoubleType, true),
      StructField("total_night_minutes", DoubleType, true),

      StructField("total_night_calls", DoubleType, true),
      StructField("total_night_charge", DoubleType, true),
      StructField("total_intl_minutes", DoubleType, true),
      StructField("total_intl_calls", DoubleType, true),
      StructField("total_intl_charge", DoubleType, true),
      StructField("number_customer_service_calls", DoubleType, true),

      StructField("churned", StringType, true)

    ))

    val ds = spark.read.option("inferSchema", "true").schema(customSchema).csv("data/churn.all")

    ds.printSchema()

    val indexer = new StringIndexer()
      .setInputCol("intl_plan")
      .setOutputCol("intl_plan_idx")

    val indexed = indexer.fit(ds).transform(ds)

    indexed.printSchema()
    
    spark.stop()
  }
}
// scalastyle:on println


