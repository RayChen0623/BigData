#!/usr/bin/env python
# coding: utf-8
# Yu-Jui Chen

from __future__ import print_function

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, concat
from graphframes import *


if __name__ == "__main__":
    vertices_input = sys.argv[1] #vertices input
    edges_input = sys.argv[2] #edges input
    spark = SparkSession.builder.appName("Assignment_4_Yu-Jui").getOrCreate()
    #set up log level
    spark.sparkContext.setLogLevel("ERROR") 
    #load data & alter header
    df_vertices = spark.read.csv(vertices_input, header = True).selectExpr("vertex_id as id")
    #set different src for two edges to create undirected edges
    df_edges = spark.read.csv(edges_input, header = True)\
                    .selectExpr("from_id as src", "to_id as dst", "edge_weight as edge_weight")
    df_edges2 = spark.read.csv("/Users/Ray/Desktop/BigData/graph_edges.txt", header = True
                         ).selectExpr("to_id as src", "from_id as dst", "edge_weight as edge_weight")
    #undirected edges
    df_edge_all = df_edges.union(df_edges2)

    #create graphframe                
    g = GraphFrame(df_vertices, df_edge_all)
    # Get pattern of triangle
    df_triangle = g.find("(a)-[]->(b); (b)-[]->(c); (a)-[]->(c)")
    # Filter the unique one
    df_triangle = df_triangle[(df_triangle.a < df_triangle.b) & (df_triangle.b < df_triangle.c)]

    #transform dtype from id:string to string 
    df_triangle = df_triangle.withColumn("a", df_triangle["a"].cast("string"))\
                             .withColumn("b", df_triangle["b"].cast("string"))\
                             .withColumn("c", df_triangle["c"].cast("string"))
    #Create the column 'al'
    df_triangle = df_triangle.withColumn('a1', df_triangle.a)
    #Format output
    df_triangle = df_triangle.withColumn('Undirected Unique Triangle', 
        concat(df_triangle.a, lit("-->"), df_triangle.b, lit("-->"), df_triangle.c, lit("-->"), df_triangle.a1))
    #print results
    print (df_triangle.select('Undirected Unique Triangle').show(truncate = False))

spark.stop()  
