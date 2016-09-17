package org.apache.spark

import org.apache.spark._
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

/**
 * Created by jayantshekhar on 9/16/16.
 */
object GraphXSample {

  def main(args: Array[String]): Unit = {

    val session = SparkSession
      .builder
      .appName("KMeans")
      .master("local")
      .getOrCreate()


    // Assume the SparkContext has already been constructed
    val sc: SparkContext = session.sparkContext

    // Create an RDD for the vertices
    val users: RDD[(VertexId, (String, String))] =
      sc.parallelize(Array((3L, ("rxin", "student")), (7L, ("jgonzal", "postdoc")),
        (5L, ("franklin", "prof")), (2L, ("istoica", "prof"))))

    // Create an RDD for edges
    val relationships: RDD[Edge[String]] =
      sc.parallelize(Array(Edge(3L, 7L, "collab"),    Edge(5L, 3L, "advisor"),
        Edge(2L, 5L, "colleague"), Edge(5L, 7L, "pi")))

    // Define a default user in case there are relationship with missing user
    val defaultUser = ("John Doe", "Missing")

    // Build the initial Graph
    val graph = Graph(users, relationships, defaultUser)

    // Count all users which are postdocs
    var count = graph.vertices.filter { case (id, (name, pos)) => pos == "postdoc" }.count
    println("count is " + count)

    // Count all the edges where src > dst
    count = graph.edges.filter(e => e.srcId > e.dstId).count

    println("count is " + count)

    // An edge triplet represents an edge along with the vertex attributes of its neighboring vertices.
    graph.triplets.map(
      triplet => triplet.srcAttr._1 + " is the " + triplet.attr + " of " + triplet.dstAttr._1
    ).collect.foreach(println(_))


    // The reverse operator returns a new graph with all the edge directions reversed.
    //graph.reverse


    // The subgraph operator takes vertex and edge predicates and returns the graph containing only the vertices that
    // satisfy the vertex predicate (evaluate to true) and edges that satisfy the edge predicate and connect vertices that satisfy the vertex predicate.


  }

}
