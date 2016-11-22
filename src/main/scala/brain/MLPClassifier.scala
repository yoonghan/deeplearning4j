import java.io._
import org.datavec.api.records.reader.impl.csv._
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec._
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.Builder
import org.deeplearning4j.nn.conf._
import org.deeplearning4j.nn.conf.layers.{OutputLayer, DenseLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.nn.multilayer._
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4s.Implicits._

/**
  * Created by Lee Wan on 11/21/2016.
  */
object MLPClassifier {
  def classify(): Unit = {

    val seed = 123  //used as random seed generation
    val learningRate = 0.01
    val batchSize = 50
    val nEpoch = 30 //No of time the training get passed thru

    val numInput = 2  //No of input of columns that needs to be used as statistics
    val numOutput = 2 //Output based on labels difference
    val numHiddenNodes = 20 //How many hidden nodes are expanded

    //RecordReader(recordreader, batchsize, labelindex, numPossible labels)
    val recordReader = new CSVRecordReader();
    recordReader.initialize(new FileSplit(new File(getClass.getResource("/mlp/training.csv").getFile)))
    val trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 0, 2)

    val testRecordReader = new CSVRecordReader();
    testRecordReader.initialize(new FileSplit(new File(getClass.getResource("/mlp/evaluate.csv").getFile)))
    val testIter = new RecordReaderDataSetIterator(testRecordReader, batchSize, 0, 2)


    val conf = new Builder()
      .seed(seed)
      .iterations(1)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(learningRate)
      .updater(Updater.NESTEROVS)
      .momentum(0.9)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation("relu").build())
      .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nIn(numHiddenNodes).nOut(numOutput).weightInit(WeightInit.XAVIER).activation("softmax").build())
      .pretrain(false).backprop(true).build()

    val model = new MultiLayerNetwork(conf)
    model.init()
    model.setListeners(new ScoreIterationListener(10))

    for(n <- 0 until nEpoch) {
      model.fit(trainIter)
    }

    println("Evaluate model...")
    val eval = new Evaluation(numOutput)
    while(testIter.hasNext) {
      val t = testIter.next()
      val features = t.getFeatureMatrix
      val labels = t.getLabels
      val predicted = model.output(features, false)

      eval.eval(labels, predicted)
    }

    println(eval.stats())


    val predictArray = Array(0.561659196951965,0.302892925040985)
    val prediction = model.predict(predictArray.toNDArray)
    println(s"""Prediction for [${predictArray.mkString(",")}] is ${prediction(0)}""")
  }
}
