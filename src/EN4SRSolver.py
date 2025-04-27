import time

import SRConfig
from SRData import SRData
from MasterTopology import MasterTopology
from MOEA import MOEA
from utils import NNTopology


class EN4SRSolver():

    def run(self):
        startTime = time.time()

        # ---------------------------------------------
        # --- Create masterTopologies
        # ---------------------------------------------
        masterTopologies = []
        all_layer_defs, all_identities = NNTopology.get_topology_definitions(topology_file=SRConfig.topology_file)
        for el in zip(all_layer_defs, all_identities):
            masterTopology = MasterTopology(inputSize=SRData.x_data.shape[1], layer_defs=el[0], identities=el[1])
            masterTopologies.append(masterTopology)
        moea = MOEA(masterTopologies)

        # ---------------------------------------------
        # --- Initialization
        # ---------------------------------------------
        SRConfig.totalIters = SRConfig.totalItersInit
        SRConfig.reg2rmse = SRConfig.reg2rmseMainRun
        moea.randomInitialization()

        moea.printPopulation(pop=moea.population, text='Initial population - before training')
        moea.printPopulationPortion(pop=moea.population, start=0, end=0, printCheckPoints=True)
        moea.trainInitialPopulation()
        moea.printBestTopologies(printCheckPoints=False)
        moea.printPopulation(pop=moea.population, text='Initial population - after training', fileName=SRConfig.progress_file, mode='w')
        moea.printNondominated(fileName=None)
        initializationTime = time.time()
        print(f'\nInitialization execution time: {(initializationTime-startTime):2.1f} seconds\n')

        # ---------------------------------------------
        # --- Evolution
        # ---------------------------------------------
        SRConfig.totalIters = SRConfig.totalItersRun
        SRConfig.reg2rmse = SRConfig.reg2rmseMainRun
        moea.runEvolution()
        moea.finalTuning()

        # ---------------------------------------------
        # --- Reporting
        # ---------------------------------------------
        endTime = time.time()
        moea.getAnalyticFormulas(maxNbOfNodes=SRConfig.maxNbOfActiveNodesToSympyfy)
        moea.printBestTopologies(all=True, printFormula=True, printSimplifiedFormula=False, printCheckPoints=False)
        moea.printNondominated(fileName=SRConfig.progress_file)
        moea.saveBestSoFarToFile(population=moea.population, prefixText='final_solution_', nbSaved=15, savePickle=SRConfig.savePickleFinal)

        print(f'\nTotal execution time: {(endTime-startTime):2.1f} seconds\n')



