name := "knn"

version := "1.0"

scalaVersion := "2.11.11"
val sparkVersion = "2.1.1"

//TODO: when scala version is bumped to 2.12, spark version must be bumped (Note:
// there is currently no 2.1.1 Spark version which will build in sbt with Scala 2.12

//resolvers ++= Seq(
//  "apache-snapshots" at "http://repository.apache.org/snapshots/"
//)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion
  // sparkVersion % "provided"
)


