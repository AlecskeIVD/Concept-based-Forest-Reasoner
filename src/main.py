from src.MNIST.mnistExperiment import runExperiment, plotMnist, twoPhaseTraining, runExperimentOnePhase
#from src.CUB.cubExperiment import runExperiment, MLPExperiment
#from src.AWA.awaExperiment import runExperiment, runExperiment2, runExperiment3
#from src.AWA.awaAccuracy import computeAccuracyMainModel, extractAccuracyBestMainModel, MLPExperiment, MLPFinalAccuracy
#from src.MNIST.MNISTAccuracy import computeAccuracyMainModel,extractAccuracyBestMainModel, MLPExperiment, MLPComputeFinalAccuracy
from src.CUB.cubAccuracy import computeAccuracyMainModel, extractAccuracyBestMainModel, MLPExperiment, MLPFinalAccuracy
#from src.AWA.awaXGB import runXGBExperiment, ComputeAccuracy
#from src.CUB.cubXGB import runXGBExperiment, ComputeAccuracy
from src.MNIST.mnistXGB import runXGBExperiment, ComputeAccuracy
from src.AWA.awaCMR import awaCMRExperiment, extractAccuracyAWACMR
from src.MNIST.mnistCMR import mnistCMRExperiment, extractAccuracyMNISTCMR, computeAccuracyMNISTMoreDifficult
from src.CUB.cubCMR import cubCMRExperiment, extractAccuracyCUBCMR
from src.MNIST.mnistRegression import trainRegressionModelbase, trainRegressionModeldropout, trainRegressionModel2stage, trainRegressionModelMLP
#from src.MNIST.mnistGivenTrees import extractAccuracyBestMainModel
#from src.AWA.awaGivenTrees import extractAccuracyBestMainModelAWA
from src.CUB.cubGivenTrees import extractAccuracyBestMainModelCUB
from src.AWA.awaAccuracy import baseModelAwa
from src.CUB.cubAccuracy import baseModelCUB
from src.MNIST.MNISTAccuracy import baseModelMNIST
from src.MNIST.mnistFetchTrees import baseTrees, dropoutTrees, twoStage

from src.MNIST.MNISTAccuracy import computeAccuracyMainModelMoreDifficult
from src.MNIST.cBottleneck import bottleNeckExperiment2


if __name__ == "__main__":
    #runExperiment()
    # plotMnist()
    #twoPhaseTraining()
    #runExperimentOnePhase()
    #runExperiment()
    #MLPExperiment()

    #runExperiment2()
    #computeAccuracyMainModel()
    #extractAccuracyBestMainModel()
    #computeAccuracyMainModelMoreDifficult()
    #computeAccuracyMNISTMoreDifficult()
    #trainRegressionModel()
#    scatterPlot()
    #computeAccuracyMainModel()
    #MLPExperiment()
    #MLPFinalAccuracy()
    #runXGBExperiment()
    #ComputeAccuracy()
    #awaCMRExperiment()
    #extractAccuracyAWACMR()
    #extractAccuracyMNISTCMR()
    #extractAccuracyCUBCMR()
    #extractAccuracyBestMainModel()
    #baseModelCUB()
    #extractAccuracyBestMainModelCUB()
    #baseModelMNIST()
    #baseModelAwa()
    #baseModelCUB()
    #baseTrees()
    #dropoutTrees()
    #twoStage()
    #trainRegressionModelMLP()
    #trainRegressionModelbase()
    #trainRegressionModeldropout()
    #trainRegressionModel2stage()
    bottleNeckExperiment2()
