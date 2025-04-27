import time
import numpy as np
import statistics
import math
import pickle

from sympy import bell

from SubTopology import SubTopology
from MasterTopology import MasterTopology
from SRUnits import UnitIdent1
from SRData import SRData
from SRConstraints import SRConstraints
import SRConfig


# def randrange(n, vmin, vmax):
#     """
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     """
#     return (vmax - vmin)*np.random.rand(n) + vmin


def sortByRMSEValid(subTopology: SubTopology):
    """
    Function used in list.sort() to produce the key as the subTopology.losses['rmse_valid'].
    :return:
    """
    return subTopology.losses["rmse_valid"]


def sortByRMSEConstr(subTopology: SubTopology):
    """
    Function used in list.sort() to produce the key as the subTopology.losses['rmse_constr'].
    :return:
    """
    return subTopology.losses["rmse_constr"]


def sortByNbOfActiveNodes(subTopology: SubTopology):
    """
    Function used in list.sort() to produce the key as the subTopology.nbOfActiveNodes].
    :return:
    """
    return subTopology.nbOfActiveNodes


def sortByLoss(pop: list[SubTopology], lossName: str, reverse: bool = False):
    if lossName == "rmse_valid":
        pop.sort(key=sortByRMSEValid, reverse=reverse)
    elif lossName == "rmse_constr":
        pop.sort(key=sortByRMSEConstr, reverse=reverse)
    elif lossName == "nbOfActiveNodes":
        pop.sort(key=sortByNbOfActiveNodes, reverse=reverse)
    return pop


def getMutualDominance(first: SubTopology, second: SubTopology, criteria: list[str]):
    """
    Check mutual dominance of the two subTopologies w.r.t. the given criteria.
    :param other:
    :return:
    """
    l1IsBetter = False
    l2IsBetter = False
    l1Dominates = False
    l2Dominates = False
    l1Equalsl2 = False
    for key in criteria:
        if first.losses[key] < second.losses[key]:
            l1IsBetter = True
        if first.losses[key] > second.losses[key]:
            l2IsBetter = True
    if l1IsBetter and (not l2IsBetter):
        l1Dominates = True
    elif l2IsBetter and (not l1IsBetter):
        l2Dominates = True
    elif (not l1IsBetter) and (not l2IsBetter):
        l1Dominates = True
        l1Equalsl2 = True
    # ---
    return l1Dominates, l2Dominates, l1Equalsl2


def splitPopulationByDomination(pop: list[SubTopology], front, criteria: list[str]):
    """
    Divides the population into two sets of non-dominated and dominated individuals.
    Dominance is computed using the losses only.
    The non-dominated set is sorted by subTopology.nbOfActiveNodes.
    :param pop: population to be divided
    :param front: non-dominated front id
    :return: sets of non-dominated and dominated elements
    """
    nonDominatedSet: list[SubTopology] = []  # --- list of non-dominated SubTopologies
    equalIdx: set[int] = set()
    dominatedIdx: set[int] = set()  # --- list of dominated SubTopologies indexes
    for i in range(len(pop)):
        if i in equalIdx:
            continue
        if front == 0 and pop[i].nbOfActiveNodes == SRConfig.maxFrontNb:
            isDominated = True
        else:
            isDominated = False
            for j in range(len(pop)):
                if i == j:
                    continue
                iDominates, jDominates, iEqualsj = getMutualDominance(
                    pop[i], pop[j], criteria=criteria
                )
                if iDominates:
                    dominatedIdx.add(j)  # --- add j into the list of dominated indexes
                if iEqualsj:
                    equalIdx.add(j)
                isDominated = isDominated or jDominates
        if not isDominated:
            pop[i].front = front
            nonDominatedSet.append(
                pop[i]
            )  # --- add i-th element to the list non-dominated SubTopologies
        else:
            dominatedIdx.add(i)
    dominatedSet = set([pop[el] for el in dominatedIdx if el not in equalIdx])
    # --------------------------------
    if not nonDominatedSet:
        return [pop[0]], []

    # # ----------------------------------------------------------------------------
    # # --- A) Take the whole set of non-dominated solutions
    # # ----------------------------------------------------------------------------
    # filteredNonDominatedSet = nonDominatedSet

    # -----------------------------------------------------------------------------
    # --- B) Choose only extreme subTopologies for each value of nbOfActiveNodes
    # -----------------------------------------------------------------------------
    filteredNonDominatedSet = set()
    extremeSolutions = set()  # --- extreme solutions w.r.t. keysPerformance
    for k in SubTopology.keysPerformance:
        nonDominatedSet = sortByLoss(nonDominatedSet, k)
        usage = []
        extremeSolutions.add(nonDominatedSet[0])
        usage.append(nonDominatedSet[0].nbOfActiveNodes)
        for sp in nonDominatedSet:
            if sp.nbOfActiveNodes not in usage:
                filteredNonDominatedSet.add(sp)
                usage.append(sp.nbOfActiveNodes)
            elif SRConfig.enhancedDominated:
                dominatedSet.add(sp)  # --- TODO: add sp to dominatedSet
    filteredNonDominatedSet = [
        sp for sp in filteredNonDominatedSet if sp not in extremeSolutions
    ]
    filteredNonDominatedSet.sort(key=sortByNbOfActiveNodes)
    extremeSolutions = list(extremeSolutions)
    extremeSolutions.extend(filteredNonDominatedSet)
    filteredNonDominatedSet = extremeSolutions
    # ---
    # filteredNonDominatedSet = list(filteredNonDominatedSet)
    dominatedSet = [sp for sp in dominatedSet if sp not in filteredNonDominatedSet]
    # ---
    # filteredNonDominatedSet.sort(key=sortByNbOfActiveNodes)
    return filteredNonDominatedSet, dominatedSet


# def nonDominatedSorting(population: list[SubTopology], size: int):
#     """
#     Applies non-dominated sorting to the population and returns 'size'' best subset of the population.
#     Population is processed with respect to the solutions' 'topologyId'.
#     :param population: an input population
#     :param size: a maximum size of the resulting population, res
#     :return: res ... a population of a required 'size' as a subpopulation of the 'population' according to the non-dominated sorting
#     """
#     print('\n*** nonDominatedSorting ...')
#     for ind in population:
#         ind.resetFront()
#     # ---
#     dominatedSets = []
#     for topologyId in range(len(MOEA.masterTopologies)):
#         ds = [st  for st in population if st.masterTopologyId==topologyId]
#         dominatedSets.append(ds)
#     # ---
#     res = []
#     front = 0
#     availableSt = len(population)
#     while (len(res) < size) and (availableSt > 0):
#         availableSt = 0
#         for topologyId in range(len(dominatedSets)):    # --- process all dominatedSets with the same frontId
#             if dominatedSets[topologyId]:
#                 nonDominatedSet, dominatedSets[topologyId] = splitPopulationByDomination(dominatedSets[topologyId], front=front, criteria=SubTopology.keysAll)
#                 res.extend(nonDominatedSet)
#             availableSt += len(dominatedSets[topologyId])   # --- sum of all solutions in dominatedSets[]
#         front += 1
#
#     res = res[:size]
#     return res


def nonDominatedSorting(
    population: list[SubTopology],
    size: int,
    criteria: list[str] = SubTopology.keysPerfNodes,
):
    """
    Applies non-dominated sorting to the population and returns 'size'' best subset of the population.
    Population is processed with respect to the solutions' 'topologyId'.
    :param population: an input population
    :param size: a maximum size of the resulting population, res
    :return: res ... a population of a required 'size' as a subpopulation of the 'population' according to the non-dominated sorting
    """
    print("\n*** nonDominatedSorting per topologyId ...")
    for ind in population:
        ind.resetFront()
    # ---
    res = []
    partition = math.ceil(
        size / len(MOEA.masterTopologies)
    )  # --- number of solutions per topologyId
    for topologyId in range(len(MOEA.masterTopologies)):
        dominatedSet = [st for st in population if st.masterTopologyId == topologyId]
        tmp = []
        front = 0
        while (len(tmp) < partition) and dominatedSet:
            nonDominatedSet, dominatedSet = splitPopulationByDomination(
                dominatedSet, front=front, criteria=SubTopology.keysAll
            )
            tmp.extend(nonDominatedSet)
            front += 1
        res.extend(tmp[:partition])
    # ---
    res = res[:size]
    return res


def nonDominatedSortingIgnoreTopologyId(population: list[SubTopology], size: int):
    """
    Applies non-dominated sorting to the population and returns 'size'' best subset of the population.
    Solution 'topologyId' is ignored.
    :param population: the input population
    :param size: the maximum size of the resulting population
    :return: res ... a population of a required 'size' as a subpopulation of the 'population' according to the non-dominated sorting
    """
    print("\n*** nonDominatedSorting while ignoring topologyIds ...")
    for ind in population:
        ind.resetFront()
    # ---
    dominatedSet = population
    res = []
    front = 0
    while (len(res) < size) and dominatedSet:
        nonDominatedSet, dominatedSet = splitPopulationByDomination(
            dominatedSet, front=front, criteria=SubTopology.keysAll
        )
        res.extend(nonDominatedSet)
        front += 1
    # ---
    res = res[:size]
    return res


def intersection(lst1, lst2):
    res = [el1 for el1 in lst1 if el1 in lst2]
    return res


def dictIsAllBelowMedian(dict1: dict[str, float], medianV: dict[str, float]):
    for key in list(set(dict1) & set(medianV)):
        if dict1[key] > medianV[key]:
            return False
    return True


def dictIsAtLeastOneBelowMedian(dict1: dict[str, float], medianV: dict[str, float]):
    for key in list(set(dict1) & set(medianV)):
        if dict1[key] <= medianV[key]:
            return True
    return False


def myQuantileIdx(data, n: int = 1):
    """
    Calculates the index of an item corresponding to the specific quantile, 0, 1 or 2.
    :param data:
    :param n:
    :return:
    """
    d = sorted(set(data))
    denom = 1
    numer = 1
    if n == -1:
        idx = SRConfig.r.randint(0, len(d) - 1)
        return idx, d
    elif n == 0:
        denom = 1
        numer = 4
    elif n == 1:
        denom = 1
        numer = 2
    elif n == 2:
        denom = 3
        numer = 4
    # ---
    if len(d) % 2 == 0:
        idx = min(int(np.ceil(len(d) * denom / numer)), len(d) - 1)
    else:
        idx = int(np.ceil(len(d) * denom / numer)) - 1
    if n > -1 and len(d) > 3:
        idx = min(idx, len(d) - 2)
    return idx, d


# ======================================================================================================================
class MOEA:
    masterTopologies: list[MasterTopology]

    def __init__(self, masterTopologies: list[MasterTopology]):
        self.population: list[SubTopology] = []
        self.archivedNondominatedFront: list[SubTopology] = []
        self.archiveBest: list[SubTopology] = []
        MOEA.masterTopologies = masterTopologies
        self.finalBest: dict[str, float] = {
            "rmse_train": 1e10,
            "rmse_valid": 1e10,
            "rmse_constr": 1e10,
        }
        self.lastNondominatedMedianActiveNodes: list[float] = len(
            self.masterTopologies
        ) * [-1]
        for topologyId in range(len(self.masterTopologies)):
            self.lastNondominatedMedianActiveNodes[topologyId] = 100.0
        self.noregularizationPhaseStart: int = 0
        self.epoch: int = -1

    def printNodeUtilizations(self, fileName: str = None, mode: str = "a"):
        nondominated, _ = splitPopulationByDomination(
            self.population, front=0, criteria=SubTopology.keysAll
        )
        print(f"\nNode utilizations: {len(self.population)} / {len(nondominated)}")
        if fileName:
            f = open(fileName, mode=mode)
            f.write(
                f"\nNode utilizations: {len(self.population)} / {len(nondominated)}\n"
            )
        # -----------------------------------------------------------------
        # --- Hidden layers
        # -----------------------------------------------------------------
        for l, layer in enumerate(MOEA.masterTopologies[0].nnLayers):
            if l == len(MOEA.masterTopologies[0].nnLayers) - 1:
                break
            print(f"    Layer {l}:")
            if fileName:
                f.write(f"    Layer {l}:\n")
            for n, node in enumerate(layer.units):
                counterAll = 0
                listAll: list[int] = []
                for i, ind in enumerate(self.population):
                    if ind.activeNodeCoordinates[l][n]:
                        counterAll += 1
                        listAll.append(i)
                # ---
                counterNonDominated = 0
                listNonDominated: list[int] = []
                for i, ind in enumerate(nondominated):
                    if ind.activeNodeCoordinates[l][n]:
                        counterNonDominated += 1
                        listNonDominated.append(i)
                # ---
                nodeName: str = MOEA.masterTopologies[0].nnLayers[l].units[n].name
                if "UnitIdent1" in MOEA.masterTopologies[0].nnLayers[l].units[n].name:
                    proxyNode: UnitIdent1 = (
                        MOEA.masterTopologies[0].nnLayers[l].units[n]
                    )
                    proxyId = proxyNode.inId
                    if l == 0:
                        proxyNode.proxyName = f"x[{proxyId}]"
                    elif (
                        "UnitIdent1"
                        not in MOEA.masterTopologies[0]
                        .nnLayers[l - 1]
                        .units[proxyId]
                        .name
                    ):
                        proxyNode.proxyName = (
                            MOEA.masterTopologies[0].nnLayers[l - 1].units[proxyId].name
                        )
                    else:
                        proxyNode.proxyName = (
                            MOEA.masterTopologies[0]
                            .nnLayers[l - 1]
                            .units[proxyId]
                            .proxyName
                        )
                    nodeName = nodeName + f" ({proxyNode.proxyName})"
                print(f"        {n}. {nodeName}:")
                print(f"            {counterAll}: {listAll}")
                print(f"            {counterNonDominated}: {listNonDominated}")
                if fileName:
                    f.write(f"        {n}. {nodeName}:\n")
                    f.write(f"            {counterAll}: {listAll}\n")
                    f.write(f"            {counterNonDominated}: {listNonDominated}\n")
        # -----------------------------------------------------------------
        # --- Output layer
        # -----------------------------------------------------------------
        print(f"    Layer {len(MOEA.masterTopologies[0].nnLayers) - 1}:")
        if fileName:
            f.write(f"    Layer {len(MOEA.masterTopologies[0].nnLayers) - 1}:\n")
        for mt in MOEA.masterTopologies:
            outNode = mt.nnLayers[-1].units[0]
            counterAll = 0
            listAll: list[int] = []
            for i, ind in enumerate(self.population):
                if ind.nnLayers[-1].units[0].name == outNode.name:
                    counterAll += 1
                    listAll.append(i)
            # ---
            counterNonDominated = 0
            listNonDominated: list[int] = []
            for i, ind in enumerate(nondominated):
                if ind.nnLayers[-1].units[0].name == outNode.name:
                    counterNonDominated += 1
                    listNonDominated.append(i)
            # ---
            print(f"        {outNode.name}:")
            print(f"            {counterAll}: {listAll}")
            print(f"            {counterNonDominated}: {listNonDominated}")
            if fileName:
                f.write(f"        {outNode.name}:\n")
                f.write(f"            {counterAll}: {listAll}\n")
                f.write(f"            {counterNonDominated}: {listNonDominated}\n")
        # ---
        if fileName:
            f.close()

    def printPopulationPortion(
        self,
        pop: list[SubTopology],
        start=0,
        end=1,
        printSimplified=False,
        printCheckPoints=False,
    ):
        for i in range(start, end):
            pop[i].printSubtopology(
                printSimplified=printSimplified, printCheckPoints=printCheckPoints
            )

    def printPopulation(
        self, pop: list[SubTopology], text="", fileName: str = None, mode: str = "a"
    ):
        if fileName:
            f = open(fileName, mode=mode)
            f.write(f"\n{text}\n")
        print(f"\n{text}")
        for i, s in enumerate(pop):
            txt = (
                f"\t {i}. complexity={s.complexity}, "
                f"activeNodes={s.nbOfActiveNodes}, "
                f'rmse_valid={s.losses["rmse_valid"]:2.7e}, '
                f'rmse_constr={s.losses["rmse_constr"]:2.7e}, '
                f"masterTopology: {s.masterTopologyId}"
            )
            print(txt)
            if fileName:
                f.write(txt + "\n")
        if fileName:
            f.close()

    def printNondominated(self, fileName: str = None, mode: str = "a"):
        if fileName:
            f = open(fileName, mode=mode)
            f.write("\nPopulation's non-dominated:\n")
        print("\nPopulation's non-dominated:")
        nondominated, _ = splitPopulationByDomination(
            self.population, front=0, criteria=SubTopology.keysAll
        )
        for i, s in enumerate(nondominated):
            txt = (
                f"\t {i}. activeNodes={s.complexity}, "
                f'rmse_valid={s.losses["rmse_valid"]:2.7e}, '
                f'rmse_constr={s.losses["rmse_constr"]:2.7e}, '
                f"nonDominatedImprovement={s.nonDominatedImprovement}, "
                f"masterTopology: {s.masterTopologyId}"
            )
            print(txt)
            if fileName:
                f.write(txt + "\n")
        if fileName:
            f.write(f"\nBackprop iterations: {SubTopology.backpropCount}\n\n")
            f.close()

    def saveBestSoFarToFile(
        self,
        population: list[SubTopology],
        prefixText: str = "",
        nbSaved: int = 4,
        savePickle=False,
    ):
        nondominated, _ = splitPopulationByDomination(
            population, front=0, criteria=SubTopology.keysAll
        )
        for i, s in enumerate(nondominated):
            if i == nbSaved:
                break
            s: SubTopology
            fileName = prefixText + str(i) + SRConfig.fileSuffix
            s.writeSubtopologyToFile(fileName=fileName, savePickle=savePickle)

    def printBestTopologies(
        self,
        all: bool = False,
        printFormula: bool = False,
        printSimplifiedFormula: bool = False,
        printCheckPoints: bool = False,
    ):
        nondominated: list[SubTopology]
        nondominated, _ = splitPopulationByDomination(
            self.population, front=0, criteria=SubTopology.keysAll
        )
        nondominated.sort(key=sortByRMSEValid)
        # --- find max acceptable number of active nodes
        vals: set[int] = set()
        for sol in nondominated:
            vals.add(sol.nbOfActiveNodes)
        n = np.maximum(np.median(list(vals)), SRConfig.maxNbOfActiveNodesToSympyfy)
        # ---
        for sol in nondominated:
            if sol.nbOfActiveNodes > n:
                continue
            sol.set_forwardpass_masks()
            rmse_valid_before = sol.calculateValidationRMSE()
            sol.printActiveNodes()
            if printFormula:
                sol.printAnalyticFormula(printSimplified=printSimplifiedFormula)
            sol.printPerformanceMetrics()
            rmse_valid_after = sol.calculateValidationRMSE()
            # ---
            if printCheckPoints:
                print(
                    f"!!! CHECK (printBestTopologies): RMSE_valid: orig={sol.losses['rmse_valid']}, before={rmse_valid_before}, after={rmse_valid_after}"
                )
                x_in, y_gt, y_hat, y_formula = sol.calculateCheckPoints()
                print("Check points:")
                for x, y, z, u in zip(x_in, y_gt, y_hat, y_formula):
                    print(f"\t{x[0]},{x[1]}: {y[0]} / {z[0]:2.3} / {u[0]:2.3}")
            # ---
            if not all:
                break

    def getAnalyticFormulas(self, maxNbOfNodes=6):
        print("getAnalyticFormulas() ...")
        nondominated: list[SubTopology]
        nondominated, _ = splitPopulationByDomination(
            self.population, front=0, criteria=SubTopology.keysAll
        )
        for ind in nondominated:
            if ind.nbOfActiveNodes <= maxNbOfNodes:
                ind.set_forwardpass_masks()
                # rmse_valid_before = ind.calculateValidationRMSE()
                ind.getAnalyticFormula(
                    simplify=ind.nbOfActiveNodes <= SRConfig.maxNbOfActiveNodesToSympyfy
                )
                # rmse_valid_after = ind.calculateValidationRMSE()
                # print(f"!!! CHECK (getAnalyticFormulas): RMSE_valid: orig={ind.losses['rmse_valid']}, before={rmse_valid_before}, after={rmse_valid_after}")

    def chooseSubTopologiesForHistoryUpdate_A(self, pop: list[SubTopology]):
        """
        Chooses a list of subTopologies to be used to update network history.
        1. Extracts a non-dominated front based on the PERFORMANCE MEASURES ONLY.
        :return:
        """
        res: list[SubTopology]
        res, _ = splitPopulationByDomination(pop, 0, criteria=SubTopology.keysAll)
        return res

    def chooseSubTopologiesForHistoryUpdate_B(self, pop: list[SubTopology]):
        """
        Chooses a list of subTopologies to be used to update network history.
        1. Extracts a non-dominated front based on the ALL MEASURES, including the complexity.
        2. Out of the subTopologies in the non-dominated front, chooses
           a) the ones with the best keyPerformance values
           b) the ones with the smallest complexity (nb. of active nodes) out of all subTopologies that have
              all performance metrics values less than the respective median value.
        :return:
        """
        # res: list[SubTopology] = []
        interRes: set[SubTopology] = set()
        nondominated, _ = splitPopulationByDomination(
            pop, 0, criteria=SubTopology.keysAll
        )
        # -----------------
        # --- a)
        # -----------------
        tmp = []
        for k in SubTopology.keysPerformance:
            minValue = np.min([sp.losses[k] for sp in nondominated])
            tmp = [
                sp
                for sp in nondominated
                if (sp.losses[k] == minValue and sp not in interRes and sp not in tmp)
            ]
            interRes.update(tmp)
        # -----------------
        # --- b)
        # -----------------
        medianValues: dict[str, float] = {}
        for k in SubTopology.keysPerformance:
            a = [sp.losses[k] for sp in nondominated]
            if len(a) > 1:
                medianValues[k] = statistics.quantiles(a, n=4)[1]
            else:
                medianValues[k] = a[0]
        # belowMedian = [st for st in nondominated if dictIsAllBelowMedian(st.losses, medianValues)]
        belowMedian = [
            st
            for st in nondominated
            if dictIsAtLeastOneBelowMedian(st.losses, medianValues)
        ]
        # --- All below-median
        tmp = belowMedian
        # --- Smallest below-median
        # spComplexity = [st.nbOfActiveNodes for st in belowMedian]
        # tmp = [st for st in belowMedian if st.nbOfActiveNodes == min(spComplexity)]
        # ---
        interRes.update(tmp)
        res: list[SubTopology] = list(interRes)
        # ---
        print(f"\nchooseSubTopologiesForHistoryUpdate_B:")
        for subTopology in res:
            print(
                f"\tRMSE_valid={subTopology.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={subTopology.losses['rmse_constr']:2.8f}, "
                f"\tcomplexity={subTopology.losses['complexity']}"
            )
        # ---
        return res

    def chooseSubTopologiesForHistoryUpdate_C(self, pop: list[SubTopology]):
        """
        Chooses a list of subTopologies to be used to update network history.
        1. Extracts a non-dominated front based on the ALL MEASURES, including the complexity.
        2. Out of the subTopologies in the non-dominated front, chooses
           a) the ones with the best keyPerformance values
           b) the ones with the smallest complexity (nb. of active nodes) out of all subTopologies that have
              all performance metrics values less than the respective median value.
        :return:
        """
        interRes: set[SubTopology] = set()
        nondominated, _ = splitPopulationByDomination(
            pop, 0, criteria=SubTopology.keysAll
        )
        # -------------------------------------------------
        # --- 1) Choose only moderate size subTopologies
        # -------------------------------------------------
        spComplexity = [st.nbOfActiveNodes for st in nondominated]
        # spComplexity = list(set(spComplexity))
        spComplexity.sort()
        if len(spComplexity) > 4:
            maxSpComplexity = np.ceil(statistics.quantiles(spComplexity, n=4)[2])
        else:
            maxSpComplexity = spComplexity[-1]
        nondominated = [
            st for st in nondominated if st.nbOfActiveNodes <= maxSpComplexity
        ]
        # --------------------------------------------
        # --- 2a) Calculate median values using all
        # ---     moderate-size subTopologies
        # --------------------------------------------
        # medianValues: dict[str,float] = {}
        # for k in SubTopology.keysPerformance:
        #     a = [sp.losses[k] for sp in nondominated]
        #     if len(a) > 1:
        #         medianValues[k] = statistics.quantiles(a, n=4)[1]
        #     else:
        #         medianValues[k] = a[0]
        # -----------------------------------------------------------------------------
        # --- 3) Filter out subTopologies with self.nonDominatedImprovement == False
        # -----------------------------------------------------------------------------
        nondominated = [st for st in nondominated if st.nonDominatedImprovement]
        if not nondominated:
            return None  # --- no subTopology can be used
        # -----------------------------------------------
        # --- 2b) Calculate median values using all
        # ---     moderate-size and True subTopologies
        # -----------------------------------------------
        medianValues: dict[str, float] = {}
        for k in SubTopology.keysPerformance:
            a = [sp.losses[k] for sp in nondominated]
            if len(a) > 2:
                medianValues[k] = statistics.quantiles(a, n=4)[
                    SRConfig.historyQualityQuantile
                ]
            else:
                medianValues[k] = np.max(a)
        print(f"\nMedian values: quantiles[{SRConfig.historyQualityQuantile}]")
        for k in SubTopology.keysPerformance:
            print(f"\t{k}: {medianValues[k]:2.8f}")
        # -------------------------------------------
        # --- 4) Select below-median subTopologies
        # -------------------------------------------
        belowMedian = [
            st for st in nondominated if dictIsAllBelowMedian(st.losses, medianValues)
        ]
        # belowMedian = [st for st in nondominated if dictIsAtLeastOneBelowMedian(st.losses, medianValues)]
        print("\nBelowMedian solutions")
        for st in belowMedian:
            print(
                f"\tcomplexity={st.losses['complexity']}, "
                f"\tRMSE_valid={st.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={st.losses['rmse_constr']:2.8f}"
            )
        # ---
        res: list[SubTopology] = []
        if belowMedian:
            tmp = []
            # --------------------------------------------
            # --- a) Extreme below-median subTopologies
            # --------------------------------------------
            for k in SubTopology.keysPerformance:
                minValue = np.min([sp.losses[k] for sp in belowMedian])
                tmp = [
                    sp
                    for sp in belowMedian
                    if (
                        sp.losses[k] == minValue
                        and sp not in interRes
                        and sp not in tmp
                    )
                ]
                interRes.update(tmp)
            # ------------------------------------------
            # --- b) Other below-median subTopologies
            # ------------------------------------------
            # --- All below-median
            # tmp = belowMedian
            # --- Smallest below-median
            spComplexity = [st.nbOfActiveNodes for st in belowMedian]
            tmp = [st for st in belowMedian if st.nbOfActiveNodes == min(spComplexity)]
            # ---
            interRes.update(tmp)
            res = list(interRes)
        # ---
        print(f"\nchooseSubTopologiesForHistoryUpdate_C: {maxSpComplexity}")
        for subTopology in res:
            print(
                f"\tcomplexity={subTopology.losses['complexity']}, "
                f"\tRMSE_valid={subTopology.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={subTopology.losses['rmse_constr']:2.8f}"
            )
        # ---
        return res

    def chooseSubTopologiesForHistoryUpdate_D(
        self, pop: list[SubTopology], wholePopMedians: dict[str, float]
    ):
        """
        Chooses a list of subTopologies to be used to update network history.
        1. Extracts a non-dominated front based on the ALL MEASURES, including the complexity.
        2. Out of the subTopologies in the non-dominated front, chooses
           a) the ones with the best keyPerformance values
           b) the ones with the smallest complexity (nb. of active nodes) out of all subTopologies that have
              all performance metrics values less than the respective median value.
        :return:
        """
        # -------------------------------------------------
        # --- 0) Select non-dominated solutions
        # -------------------------------------------------
        # print('\nlastSubTopologiesAddedToHistory:', end='')
        # for el in self.masterTopologies[0].lastSubTopologiesAddedToHistory:
        #     print(f' {el}', end='')
        # print('')
        nondominated, _ = splitPopulationByDomination(
            pop, 0, criteria=SubTopology.keysAll
        )
        # -------------------------------------------------
        # --- 1) Choose only moderate size subTopologies
        # -------------------------------------------------
        spComplexity = [st.nbOfActiveNodes for st in nondominated]
        spComplexity.sort()
        if len(spComplexity) > 4 and SRConfig.historySizeQuantile < 4:
            maxSpComplexity = np.ceil(
                statistics.quantiles(spComplexity, n=4)[SRConfig.historySizeQuantile]
            )  # --- 2
        else:
            maxSpComplexity = spComplexity[-1]
        print(f"\nmaxSpComplexity: {maxSpComplexity}")
        nondominated = [
            st for st in nondominated if st.nbOfActiveNodes <= maxSpComplexity
        ]
        # -----------------------------------------------------------------------
        # --- 3) Calculate median values using all solutions in 'nondominated'
        # -----------------------------------------------------------------------
        print("")
        medianValues: dict[str, float] = {}
        for k in SubTopology.keysPerformance:
            a = [sp.losses[k] for sp in nondominated if sp.losses[k] > 0.0]
            print(f"{k}: {a}")
            if not a:
                medianValues[k] = (
                    0.0  # --- there is no solution with a positive losses[k] value
                )
            elif len(a) > 2:
                medianValues[k] = statistics.quantiles(a, n=4)[
                    SRConfig.historyQualityQuantile
                ]  # --- 0
            else:
                medianValues[k] = np.max(a)
        print(f"Median values: quantiles[{SRConfig.historyQualityQuantile}]")
        for k in SubTopology.keysPerformance:
            print(f"\t{k}: {medianValues[k]:2.8f}")
        # -------------------------------------------
        # --- 4) Select below-median subTopologies
        # -------------------------------------------
        selected = [
            st for st in nondominated if dictIsAllBelowMedian(st.losses, medianValues)
        ]
        selected = [
            st for st in selected if dictIsAllBelowMedian(st.losses, wholePopMedians)
        ]
        print("\nBelowMedian solutions")
        for st in selected:
            print(
                f"\tcomplexity={st.losses['complexity']}, "
                f"\tRMSE_valid={st.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={st.losses['rmse_constr']:2.8f}"
            )
        # ---
        # if not selected:    # --- ORIGINAL
        # --------------------------------------------------
        # --- Take the extreme nondominated subTopologies
        # --------------------------------------------------
        # for k in SubTopology.keysPerformance:
        #     minValue = np.min([sp.losses[k] for sp in nondominated])
        #     tmp = [sp for sp in nondominated if (sp.losses[k] == minValue and sp not in selected)]
        #     selected.extend(tmp)
        # ---
        res: list[SubTopology] = []
        if selected:
            interRes: set[SubTopology] = (
                set()
            )  # --- set of the extreme solutions in 'selected', the below-median solutions, with the smallest nbOfActiveNodes
            tmp = []
            # --------------------------------------------
            # --- a) Extreme below-median subTopologies     TODO: useless
            # --------------------------------------------
            # for k in SubTopology.keysPerformance:
            #     minValue = np.min([sp.losses[k] for sp in selected])    # --- ORIGINAL: selected; TODO: nondominated
            #     tmp = [sp for sp in selected if (sp.losses[k] == minValue and sp not in interRes and sp not in tmp)]    # --- ORIGINAL: selected; TODO: nondominated
            #     tmp = sortByLoss(pop=tmp, lossName='nbOfActiveNodes')   # --- take the one with the smallest nbOfActiveNodes
            #     interRes.add(tmp[0])
            # ------------------------------------------
            # --- b) Other below-median subTopologies
            # ------------------------------------------
            # --- All below-median
            # tmp.extend(sortByLoss(pop=selected, lossName='nbOfActiveNodes', reverse=True))
            tmp = selected
            # --- Smallest below-median
            # spComplexity = [st.nbOfActiveNodes for st in selected]
            # tmp = [st for st in selected if st.nbOfActiveNodes == min(spComplexity)]
            # ---
            # res.extend(interRes)      TODO
            res = sortByLoss(
                pop=tmp,
                lossName="nbOfActiveNodes",
                reverse=SRConfig.historySortedDescending,
            )  # --- ORIGINAL: reverse=True; TODO: reverse=False
            # for st in interRes:   TODO: useless
            #     if st not in res:
            #         res.append(st)
            # interRes.update(tmp)
            # res = list(interRes)
            # res = sortByLoss(pop=res, lossName='nbOfActiveNodes', reverse=True)
        # ------------------------------------------------------------------------------------
        # --- Add the extreme nondominated subTopologies w.r.t. SubTopology.keysPerformance
        # ------------------------------------------------------------------------------------
        if not selected:  # --- ORIGINAL
            nondominated = [
                st
                for st in nondominated
                if dictIsAllBelowMedian(st.losses, wholePopMedians)
            ]
            if nondominated:
                for k in SubTopology.keysPerformance:
                    minValue = np.min([sp.losses[k] for sp in nondominated])
                    tmp = [
                        sp
                        for sp in nondominated
                        if (sp.losses[k] == minValue and sp not in res)
                    ]
                    res.extend(tmp)
        # ------------------------------------------------------------------------------------
        # --- Add the extreme nondominated subTopologies w.r.t. 'nbOfActiveNodes'
        # ------------------------------------------------------------------------------------
        # minValue = np.min([sp.losses['nbOfActiveNodes'] for sp in nondominated])
        # tmp = [sp for sp in nondominated if (sp.losses['nbOfActiveNodes'] == minValue and sp.losses['rmse_valid'] <= medianValues['rmse_valid'] and sp not in res)]
        # res.extend(tmp)
        # tmp = [sp for sp in nondominated if (sp.losses['nbOfActiveNodes'] == minValue and sp.losses['rmse_constr'] <= medianValues['rmse_constr'] and sp not in res)]
        # res.extend(tmp)
        # ---
        print(f"\nchooseSubTopologiesForHistoryUpdate_D: {maxSpComplexity}")
        for subTopology in res:
            print(
                f"\tcomplexity={subTopology.losses['complexity']}, "
                f"\tRMSE_valid={subTopology.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={subTopology.losses['rmse_constr']:2.8f}"
            )
        # ---
        return res

    def chooseSubTopologiesForHistoryUpdate_E(
        self, pop: list[SubTopology], scalingFactor: float = 1.0
    ):
        """
        Chooses a list of subTopologies to be used to update network history.
        1. Extracts a non-dominated front based on the ALL MEASURES, including the complexity.
        2. Out of the subTopologies in the non-dominated front, chooses the ones that have all performance metric values
           no worse than the specific quantileQuality value.
        3. Out of the constructed set of high-quality subTopologies, takes the ones with the smallest complexity (nb. of active nodes).
        :return:
        """
        # ------------------------------------------------------------
        # --- 1) Select non-dominated solutions w.r.t. all criteria
        # ------------------------------------------------------------
        # nondominated, _ = splitPopulationByDomination(pop, 0, criteria=SubTopology.keysAll)
        nondominated = pop

        # ------------------------------------------------------------------------------------
        # --- 2) Calculate the quantileQuality values using all solutions in 'nondominated'
        # ------------------------------------------------------------------------------------
        quantileValues: dict[str, float] = {}
        quantileQuality = (
            SRConfig.historyQualityQuantile
        )  # --- ORIGINAL: 2    # --- 1 ... 50%, 2 ... 75%
        for k in SubTopology.keysPerformance:
            a = [st.losses[k] for st in nondominated if st.losses[k] > 0.0]
            print(f"{k}: {a}")
            if not a:
                quantileValues[k] = (
                    0.0  # --- there is no solution with a positive losses[k] value
                )
            elif len(a) <= 2:
                quantileValues[k] = np.max(a)
            else:
                # quantileValues[k] = statistics.quantiles(a, n=4)[quantileQuality]     # --- ORIGINAL
                idx, a = myQuantileIdx(a, quantileQuality)
                quantileValues[k] = scalingFactor * a[idx]  # --- TODO: NEW
        print(f"\nQuantile[{quantileQuality}] values:")
        for k in SubTopology.keysPerformance:
            print(f"\t{k}: {quantileValues[k]:2.8f}")

        # ----------------------------------------------------
        # --- 3) Select below-quantileQuality subTopologies
        # ----------------------------------------------------
        selected = [
            st for st in nondominated if dictIsAllBelowMedian(st.losses, quantileValues)
        ]
        print("\nBelow-quantileQuality solutions")
        for st in selected:
            print(
                f"\tcomplexity={st.losses['complexity']}, "
                f"\tRMSE_valid={st.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={st.losses['rmse_constr']:2.8f}"
            )

        # -----------------------------------------------------------------
        # --- 4a) Select the simplest SRConfig.historySize subTopologies
        # -----------------------------------------------------------------
        # selected: list[SubTopology] = sortByLoss(pop=selected, lossName='nbOfActiveNodes')
        # res = []
        # for st in selected:
        #     res.append(st)
        #     if len(res) == SRConfig.historySize:
        #         break
        # --------------------------------------------------------------
        # --- 4b) Select the solutions with below-quantile complexity
        # --------------------------------------------------------------
        quantileSize = (
            SRConfig.historySizeQuantile
        )  # --- ORIGINAL: 1    # --- -1 ... random, 0 ... 25%, 1 ... 50%, 2 ... 75%
        complexities = [st.nbOfActiveNodes for st in selected]
        res = []
        if complexities:
            idx, complexities = myQuantileIdx(complexities, quantileSize)
            maxSize = complexities[idx]
            res = [st for st in selected if st.nbOfActiveNodes <= maxSize]
            res = res[: SRConfig.historySize]
            print(
                f"\nchooseSubTopologiesForHistoryUpdate_E (quantileValues={quantileQuality}, quantileSize={quantileSize}, maxSize={maxSize}):"
            )
        # ---
        for subTopology in res:
            print(
                f"\tcomplexity={subTopology.losses['complexity']}, "
                f"\tRMSE_valid={subTopology.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={subTopology.losses['rmse_constr']:2.8f}"
            )
        # ---
        return res

    def chooseSubTopologiesForHistoryPerSizeUpdate(self, pop: list[SubTopology]):
        """
        Chooses a list of subTopologies to be used to update network history.
        1. Extracts a non-dominated front based on the ALL MEASURES, including the complexity.
        2. Out of the subTopologies in the non-dominated front, chooses only the moderate size ones
        :return:
        """
        # -------------------------------------------------
        # --- 0) Select non-dominated solutions
        # -------------------------------------------------
        print("\nlastSubTopologiesAddedToHistory:", end="")
        for el in self.masterTopologies[0].lastSubTopologiesAddedToHistory:
            print(f" {el}", end="")
        print("")
        nondominated, _ = splitPopulationByDomination(
            pop, 0, criteria=SubTopology.keysAll
        )
        # -------------------------------------------------
        # --- 1) Choose only moderate size subTopologies
        # -------------------------------------------------
        spComplexity = [st.nbOfActiveNodes for st in nondominated]
        spComplexity.sort()
        if len(spComplexity) > 4 and SRConfig.historySizeQuantile < 4:
            maxSpComplexity = np.ceil(
                statistics.quantiles(spComplexity, n=4)[SRConfig.historySizeQuantile]
            )  # --- 2
        else:
            maxSpComplexity = spComplexity[-1]
        print(f"\nmaxSpComplexity: {maxSpComplexity}")
        nondominated = [
            st for st in nondominated if st.nbOfActiveNodes <= maxSpComplexity
        ]
        # -----------------------------------------------------------------------------
        # --- 2) Filter out subTopologies with self.nonDominatedImprovement == False
        # -----------------------------------------------------------------------------
        # nondominated = [st for st in nondominated if st.nonDominatedImprovement]
        # # ---
        # if not nondominated:
        #     return None     # --- no subTopology can be used
        # -----------------------------------------------
        # --- 3) Calculate median values using all
        # ---    non-dominated subTopologies
        # -----------------------------------------------
        print("")
        medianValues: dict[str, float] = {}
        for k in SubTopology.keysPerformance:
            a = [sp.losses[k] for sp in nondominated]
            print(f"{k}: {a}")
            if len(a) > 2:
                medianValues[k] = statistics.quantiles(a, n=4)[
                    SRConfig.historyQualityQuantile
                ]  # --- 0
            else:
                medianValues[k] = np.max(a)
        print(f"Median values: quantiles[{SRConfig.historyQualityQuantile}]")
        for k in SubTopology.keysPerformance:
            print(f"\t{k}: {medianValues[k]:2.8f}")
        # -------------------------------------------
        # --- 4) Select below-median subTopologies
        # -------------------------------------------
        selected = [
            st for st in nondominated if dictIsAllBelowMedian(st.losses, medianValues)
        ]
        print("\nBelowMedian solutions")
        for st in selected:
            print(
                f"\tcomplexity={st.losses['complexity']}, "
                f"\tRMSE_valid={st.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={st.losses['rmse_constr']:2.8f}"
            )
        # ---
        if not selected:
            # return None     # --- no subTopology can be used
            # --------------------------------------------
            # --- Take the extreme below-median subTopologies
            # --------------------------------------------
            for k in SubTopology.keysPerformance:
                minValue = np.min([sp.losses[k] for sp in nondominated])
                tmp = [
                    sp
                    for sp in nondominated
                    if (sp.losses[k] == minValue and sp not in selected)
                ]
                selected.extend(tmp)
        # ---
        print(f"\nchooseSubTopologiesForHistoryPerSizeUpdate: {maxSpComplexity}")
        for subTopology in selected:
            print(
                f"\tcomplexity={subTopology.losses['complexity']}, "
                f"\tRMSE_valid={subTopology.losses['rmse_valid']:2.8f}, "
                f"\tRMSE_constr={subTopology.losses['rmse_constr']:2.8f}"
            )
        # ---
        return selected

    def randomInitialization(self):
        """
        Randomly initializes the population of SubTopologies derived from the masterTopology.
        :return:
        """
        startTime = time.time()
        self.population = []
        masterId: int
        bestValue = {
            "rmse_train": 1.0e6,
            "rmse_valid": 1.0e6,
            "rmse_constr": 1.0e6,
            "nbOfActiveNodes": 1e6,
        }
        for i in range(SRConfig.initialPopsize):
            print(f"\n{i}. subTopology")
            masterId = i % len(self.masterTopologies)
            subTopology = self.masterTopologies[masterId].generateRandomSubTopology()
            subTopology.masterTopologyId = masterId
            subTopology.train_nn(
                learningSteps=0,
                learningRate=SRConfig.learning_rate_newborn,
                regularize=False,
                constrain=False,
                clipWeights=False,
                reduceLearningRate=False,
                alternatePrefs=False,
                printStep=0,
            )
            self.population.append(subTopology)
            # ---
            print(
                f"RMSE_train={subTopology.losses['rmse_train']:2.8f}, "
                f"RMSE_valid={subTopology.losses['rmse_valid']:2.8f}, "
                f"RMSE_constr={subTopology.losses['rmse_constr']:2.8f}, "
                f"complexity={subTopology.losses['complexity']}"
            )
            if bestValue["rmse_valid"] > subTopology.losses["rmse_valid"]:
                bestValue["rmse_valid"] = subTopology.losses["rmse_valid"]
                bestValue["rmse_constr"] = subTopology.losses["rmse_constr"]
                bestValue["complexity"] = subTopology.losses["complexity"]
            if self.finalBest["rmse_valid"] > subTopology.losses["rmse_valid"]:
                self.finalBest["rmse_valid"] = subTopology.losses["rmse_valid"]
                self.finalBest["rmse_constr"] = subTopology.losses["rmse_constr"]
                self.finalBest["complexity"] = subTopology.losses["complexity"]
        # ---
        endTime = time.time()
        print(
            f"\nBest initial subTopology: nbOfActiveNodes={bestValue['nbOfActiveNodes']}, rmse_valid={bestValue['rmse_valid']:2.3f}, rmse_constr={bestValue['rmse_constr']:2.3f}"
        )
        print(f"Initialization phase time: {(endTime-startTime):2.1f} seconds")

    def trainInitialPopulation(self):
        """
        Trains SubTopologies of the population.
        :return:
        """
        startTime = time.time()
        # self.population = []
        bestValue = {
            "rmse_train": 1.0e6,
            "rmse_valid": 1.0e6,
            "rmse_constr": 1.0e6,
            "nbOfActiveNodes": 1e6,
        }
        for i, subTopology in enumerate(self.population):
            print(f"\n{i}. subTopology")
            # ----------------------------------------------------------------------
            # --- Learning phase WITHOUT constraints
            # ----------------------------------------------------------------------
            subTopology.train_nn(
                learningSteps=int(SRConfig.totalItersInit / 2),
                learningRate=SRConfig.learning_rate_newborn,
                regularize=False,
                clipWeights=False,
                constrain=False,
                reduceLearningRate=False,
                alternatePrefs=False,
                printStep=0,
            )
            # ---------------------------------------------------------------------------------------
            # --- Learning phase WITH constraints (if available) or regularization (no constraints)
            # ---------------------------------------------------------------------------------------
            if SRConfig.constraints_file:
                subTopology.train_nn(
                    learningSteps=int(SRConfig.totalItersInit / 2),
                    learningRate=SRConfig.learning_rate_newborn,
                    regularize=False,
                    clipWeights=False,
                    constrain=False,
                    reduceLearningRate=False,
                    alternatePrefs=False,
                    printStep=0,
                )
            else:
                subTopology.train_nn(
                    learningSteps=int(SRConfig.totalItersInit / 2),
                    learningRate=SRConfig.learning_rate_newborn,
                    regularize=False,  # --- ORIGINAL: True
                    clipWeights=False,
                    constrain=False,
                    reduceLearningRate=False,
                    alternatePrefs=False,
                    printStep=0,
                )
            # ---
            print(
                f"RMSE_train={subTopology.losses['rmse_train']:2.8f}, "
                f"RMSE_valid={subTopology.losses['rmse_valid']:2.8f}, "
                f"RMSE_constr={subTopology.losses['rmse_constr']:2.8f}, "
                f"complexity={subTopology.losses['complexity']}"
            )
            if bestValue["rmse_valid"] > subTopology.losses["rmse_valid"]:
                bestValue["rmse_valid"] = subTopology.losses["rmse_valid"]
                bestValue["rmse_constr"] = subTopology.losses["rmse_constr"]
                bestValue["complexity"] = subTopology.losses["complexity"]
            if self.finalBest["rmse_valid"] > subTopology.losses["rmse_valid"]:
                self.finalBest["rmse_valid"] = subTopology.losses["rmse_valid"]
                self.finalBest["rmse_constr"] = subTopology.losses["rmse_constr"]
                self.finalBest["complexity"] = subTopology.losses["complexity"]
        # ---
        endTime = time.time()
        print(
            f"\nBest initial subTopology: nbOfActiveNodes={bestValue['nbOfActiveNodes']}, rmse_valid={bestValue['rmse_valid']:2.3f}, rmse_constr={bestValue['rmse_constr']:2.3f}"
        )
        print(f"Initialization phase time: {(endTime-startTime):2.1f} seconds")

        # ----------------------------------------
        # --- Choose the best popSize solutions
        # ----------------------------------------
        self.population = nonDominatedSorting(self.population, SRConfig.popSize)

        # ----------------------------------------
        # --- History update: TODO - do not update history with the initial population solutions
        # ----------------------------------------
        # if SRConfig.historyType == 0:
        #     # --- Perform history update for each masterTopology
        #     populationMedianValues: dict[str, float] = self.calculatePopulationMedianValues(self.population)
        #     for topologyId in range(len(MOEA.masterTopologies)):
        #         # updateList: list[SubTopology] = self.chooseSubTopologiesForHistoryUpdate_D(pop=self.population)
        #         popForUpdate = [st for st in self.population if st.masterTopologyId == topologyId]
        #         updateList: list[SubTopology] = self.chooseSubTopologiesForHistoryUpdate_D(pop=popForUpdate, wholePopMedians=populationMedianValues)
        #         if updateList is not None:
        #             partitionSize = int(SRConfig.historySize / len(MOEA.masterTopologies))
        #             updateList = updateList[:partitionSize]
        #             for st in updateList:
        #                 self.masterTopologies[0].updateNetworkHistory(st, partitionSize=partitionSize)
        # else:
        #     updateList: list[SubTopology] = self.chooseSubTopologiesForHistoryPerSizeUpdate(pop=self.population)
        #     if updateList is not None:
        #         for st in updateList:
        #             self.masterTopologies[0].updateNetworkHistoryPerSize(st)

    def tournamentSelection(
        self, pop: list[SubTopology], requiredMaster: int = -1, size=1
    ):
        i = 1
        while True:
            winner = pop[SRConfig.r.integers(low=0, high=len(pop))]
            if requiredMaster == -1 or winner.masterTopologyId == requiredMaster:
                requiredMaster = (
                    winner.masterTopologyId
                )  # --- assign the requiredMaster to the first candidate's topologyId
                break
        while i < size:
            while True:
                cand = pop[SRConfig.r.integers(low=0, high=len(pop))]
                if requiredMaster == -1 or cand.masterTopologyId == requiredMaster:
                    break
            # ---
            if winner.front > cand.front:
                winner = cand
            i += 1
        return winner

    def runCrossoverMutationStep(
        self,
        id: int,
        pop: list[SubTopology],
        regularize: bool,
        nbOfGenerated: int,
        learningSteps: int = 2,
        reduceLearningRate=False,
    ):
        """
        Generates an intermediate population of SubTopologies using crossover.
        :return: A population of new SubTopologies
        """
        print("\n*** runCrossoverMutationStep ...")
        startTime = time.time()
        newPopulation = []
        topologyId: int = 0
        while len(newPopulation) < nbOfGenerated:
            # historyId = SRConfig.r.integers(low=0, high=len(self.masterTopologies))
            # p = SRConfig.r.random()
            # if (not regularize) or (p < SRConfig.historyUseParentHistoryId):
            #     # --- Use the same historyId as the topologyId
            #     historyId = topologyId
            # else:
            #     # --- Choose some historyId different than the topologyId
            #     while True:
            #         historyId = SRConfig.r.integers(low=0, high=len(self.masterTopologies))
            #         if historyId != topologyId:
            #             break

            # ----------------------------------------------------------
            # --- SubTopology generation
            # ----------------------------------------------------------
            while True:
                p1: SubTopology = self.tournamentSelection(
                    pop=pop, requiredMaster=topologyId, size=SRConfig.tournamentSize1
                )  # --- ORIGINAL: size=SRConfig.tournamentSize
                if p1.nbOfActiveNodes < SRConfig.maxFrontNb:
                    break
            while True:
                p2: SubTopology = self.tournamentSelection(
                    pop=pop, requiredMaster=-1, size=SRConfig.tournamentSize2
                )  # --- ORIGINAL: requiredMaster=p1.masterTopologyId
                if p2.nbOfActiveNodes < SRConfig.maxFrontNb:
                    break
            child: SubTopology
            if SRConfig.r.random() < SRConfig.crossoverRate:
                child = self.masterTopologies[
                    p1.masterTopologyId
                ].subTopologyCrossoverB(
                    p1,
                    p2,
                    MOEA.masterTopologies[p1.masterTopologyId].networkHistory,
                    MOEA.masterTopologies[p2.masterTopologyId].networkHistory,
                )
            else:
                child = p1.cloneSubTopology()
                changed: bool
                if SRConfig.r.random() < SRConfig.applyActivityOrConfigMutation:
                    changed = self.masterTopologies[
                        child.masterTopologyId
                    ].networkActivityMutation(
                        child
                    )  # --- historyId
                else:
                    changed = self.masterTopologies[
                        child.masterTopologyId
                    ].networkConfigMutation_B(
                        child
                    )  # --- historyId
            # ----------------------------------------------------------
            # --- Evaluation
            # ----------------------------------------------------------
            self.masterTopologies[
                child.masterTopologyId
            ].rectifyActiveNodesConnectivity(child)
            # --- TODO: remove, just for debugging
            # if child.nnLayers[0].units[8].weights[0].v[0] == -1. or child.nnLayers[0].units[8].weights[0].v[1] == -1.:
            #     print('!!! ERROR !!!')
            # ---
            child.train_nn(
                learningSteps=learningSteps,
                learningRate=SRConfig.learning_rate_newborn,
                regularize=regularize,
                constrain=regularize,
                clipWeights=regularize,
                reduceLearningRate=reduceLearningRate,
                alternatePrefs=False,
            )
            # ---   TODO: remove, just for debugging
            if child.checkUnitIdent1Nodes():
                newPopulation.append(child)
            else:
                print("\n!!! Invalid solution generated !!!\n")
            # ---
            topologyId = (topologyId + 1) % len(MOEA.masterTopologies)
        endTime = time.time()
        print(
            f"\n{id}.\tCrossover-mutation phase time: {(endTime-startTime):2.1f} seconds"
        )
        return newPopulation

    def runNSGAPhase(self, regularize=False, fileName: str = None):
        """
        Evolves a population of SubTopologies using crossover and mutation for a number of generations.
        :return: A population of new SubTopologies
        """
        bestValue = {
            "nbOfActiveNodes": 1e6,
            "rmse_train": 1.0e6,
            "rmse_valid": 1.0e6,
            "rmse_constr": 1.0e6,
        }
        startTime = time.time()

        # ---------------------------------------------------------------------------------------------------
        # --- 0-th self.runCrossoverMutationStep()
        # ---------------------------------------------------------------------------------------------------
        newPopulation: list[SubTopology]
        newPopulation = self.runCrossoverMutationStep(
            id=0,
            pop=self.population,
            regularize=regularize,
            nbOfGenerated=SRConfig.popSize,
            learningSteps=SRConfig.totalItersNewbornNoPerturbations,
        )
        newPopulation = nonDominatedSorting(
            newPopulation, len(newPopulation)
        )  # --- TODO: unnecessary
        # ---------------------------------------------------------------------------------------------------
        # --- Main loop
        # ---------------------------------------------------------------------------------------------------
        for i in range(1, SRConfig.nsgaGenerationsNoPerturbations):
            # --------------------------------------------------------------------------------------------
            # --- Create an intermediate population of subTopologies using crossover and mutation
            # --------------------------------------------------------------------------------------------
            interPop = self.runCrossoverMutationStep(
                id=i,
                pop=newPopulation,
                regularize=regularize,
                nbOfGenerated=SRConfig.generatedPopSize,
                learningSteps=SRConfig.totalItersNewbornNoPerturbations,
            )

            # --------------------------------------------------------------------------------------------
            # --- Apply non-dominated sorting to the newPopulation to get the best subset of solutions
            # --------------------------------------------------------------------------------------------
            newPopulation.extend(interPop)
            newPopulation = nonDominatedSorting(
                newPopulation, SRConfig.popSize
            )  # --- ORIGINAL: True
        endTime = time.time()

        print(f"\nNSGA phase time: {(endTime-startTime):2.1f} seconds")
        return newPopulation

    def runWeightTuningPhase(
        self,
        pop: list[SubTopology],
        regularize: bool = True,
        deactivateBelowThresholdUnits: bool = False,
        fileName: str = None,
        mode: str = "a",
    ):
        """
        Applies a gradient descent algorithm to all SubTopologies in the 'pop'
        :param pop:
        :return: interPopulation with tuned weights.
        """
        if fileName:
            f = open(fileName, mode=mode)
        startTime = time.time()
        bestValue = {
            "nbOfActiveNodes": 1e6,
            "rmse_train": 1.0e6,
            "rmse_valid": 1.0e6,
            "rmse_constr": 1.0e6,
        }
        newPop: list[SubTopology] = []
        # --- find max acceptable number of active nodes
        vals: set[float] = set()
        for sol in pop:
            vals.add(sol.nbOfActiveNodes)
        # n = np.maximum(np.median(list(vals)), SRConfig.minNbOfActiveNodesToTune)
        if SRConfig.constraints_file:
            # n = np.maximum(self.lastNondominatedMedianActiveNodes, SRConfig.minNbOfActiveNodesToTune)
            n = [
                np.maximum(el, SRConfig.minNbOfActiveNodesToTune)
                for el in self.lastNondominatedMedianActiveNodes
            ]
        else:
            n = len(self.masterTopologies) * [1000.0]
        # --- Print all newly generated solutions
        if fileName:
            f.write("\nAll newly generated subTopologies:\n")
        print("\nAll newly generated subTopologies")
        for individual in pop:
            print(
                f' \t{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.12f} / topology: {individual.masterTopologyId}'
            )
            if fileName:
                f.write(
                    f' \t{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.12f}\n'
                )
        # ---
        print(f"\nTuned subTopologies ({n[pop[0].masterTopologyId]}))")
        if fileName:
            f.write("\nTuned subTopologies:\n")
        for i, individual_orig in enumerate(pop):
            if individual_orig.nbOfActiveNodes > n[individual_orig.masterTopologyId]:
                continue
            print(f"{i}. ", end="")
            individual = individual_orig.cloneSubTopology()
            print(
                f' \t{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f}  ->  ',
                end="",
            )
            if fileName:
                f.write(
                    f' \t{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f}  ->  '
                )
            individual.train_nn(
                learningSteps=SRConfig.totalIters,
                learningRate=SRConfig.learning_rate_newborn,
                regularize=regularize,
                constrain=regularize,
                clipWeights=regularize,
                deactivateBelowThresholdUnits=deactivateBelowThresholdUnits,
                # constrain=True,     # --- ORIGINAL
                reduceLearningRate=False,  # --- TODO
                alternatePrefs=SRConfig.alternatePrefs,
                printStep=0,
            )
            # individual_orig.nonDominatedImprovement = nonDominatedImprovement
            print(
                f'{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f} / topology: {individual.masterTopologyId}'
            )
            if fileName:
                f.write(
                    f'{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f}\n'
                )
            newPop.append(individual_orig)
            newPop.append(individual)
        endTime = time.time()
        print(f"Weights-tuning phase time: {(endTime-startTime):2.1f} seconds\n")
        # ---
        if fileName:
            f.write(f"\n")
            f.close()
        # ---
        return newPop

    def setNodeUtilizations(self):
        """
        Sets the node utilization statistics according to the current population
        :return:
        """
        nonDominatedSet, _ = splitPopulationByDomination(
            self.population, front=0, criteria=SubTopology.keysAll
        )
        self.masterTopologies[0].resetNodeUtilizations()
        for ind in nonDominatedSet:
            self.masterTopologies[0].updateNodeUtilizations(ind)

    def mergePopulations(self, popA, popB, size: int):
        """
        Merges two populations and takes 'size' best solutions while taking into account solutions 'topologyId'
        :param popA:
        :param popB:
        :param size: size of the resulting population
        """
        res = []
        res.extend(popA)
        res.extend(popB)
        # --- Apply non-dominated sorting to the 'res' population and take 'size' best solutions
        res = nonDominatedSorting(res, size)
        return res

    def updateArchive(self, currNondominatedFront, currArchive, size: int):
        """
        Merges two populations and takes 'size' best solutions while ignoring solutions 'topologyId'
        :param size: size of the resulting population
        """
        res = []
        res.extend(currNondominatedFront)
        res.extend(currArchive)
        # --- Apply non-dominated sorting to the 'res' population and take 'size' best solutions
        res = nonDominatedSortingIgnoreTopologyId(res, size)
        return res

    def fineTuneNonDominatedFront(
        self,
        pop: list[SubTopology],
        topologyId: int = 0,
        regularize: bool = False,
        generation: int = 0,
    ):
        """
        :param pop: This is already a selection of subtopologies with the given topologyId
        :param topologyId:
        :param regularize:
        :param generation:
        :return:
        """
        print(f"\nFine-tuning the non-dominated set", end="")
        startTime = time.time()
        nonDominatedSet, _ = splitPopulationByDomination(
            pop, front=0, criteria=SubTopology.keysAll
        )
        tmp = [el for el in nonDominatedSet]
        tunedNonDominatedSet = []
        # --- find max acceptable number of active nodes
        vals: set[float] = set()
        for sol in nonDominatedSet:
            vals.add(sol.nbOfActiveNodes)
        if len(vals) > 2 and SRConfig.historySizeQuantile < 4:  # --- ORIGINAL: 1, 0
            self.lastNondominatedMedianActiveNodes[topologyId] = np.round(
                statistics.quantiles(vals, n=4)[SRConfig.historySizeQuantile] + 0.5
            )
        else:
            valsList = list(set(vals))
            valsList.sort()
            self.lastNondominatedMedianActiveNodes[topologyId] = valsList[-1]
        print(
            f"\nlastNondominatedMedianActiveNodes[{topologyId}]: {self.lastNondominatedMedianActiveNodes[topologyId]}",
            end="",
        )
        n = np.maximum(
            self.lastNondominatedMedianActiveNodes[topologyId],
            SRConfig.minNbOfActiveNodesToTune,
        )
        # ---
        if SRConfig.fineTuningIters > 0:
            nonDominatedSet = [el for el in nonDominatedSet if el.nbOfActiveNodes <= n]
            if len(nonDominatedSet) < 3:
                tmp = sortByLoss(tmp, "nbOfActiveNodes")
                k = 1
                while k < len(tmp) and k < 4:
                    nonDominatedSet.append(tmp[k])
                    k += 1
            # ---
            for i, individual_orig in enumerate(nonDominatedSet):
                if individual_orig.nbOfActiveNodes > n:
                    continue
                individual = individual_orig.cloneSubTopology()
                print(
                    f'\n{i}.\t{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f}  ->  ',
                    end="",
                )
                printLoss = False
                if generation > 3:
                    printStep = 0
                else:
                    printStep = 0
                nonDominatedImprovement, _ = individual.train_nn(
                    learning_subset_size=SRConfig.learning_subset_size,
                    learningSteps=SRConfig.fineTuningIters,
                    learningRate=SRConfig.learning_rate_finetune,
                    regularize=regularize,  # --- ORIGINAL: False
                    clipWeights=regularize,
                    constrain=regularize,
                    # constrain=True,     # --- ORIGINAL
                    reduceLearningRate=False,
                    alternatePrefs=SRConfig.alternatePrefs,
                    printStep=printStep,
                )
                individual_orig.nonDominatedImprovement = nonDominatedImprovement
                print(
                    f'{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f} / topologyId: {individual.masterTopologyId}',
                    end="",
                )
                tunedNonDominatedSet.append(individual)
        # ---
        # tunedNonDominatedSet = self.mergePopulations(tunedNonDominatedSet, nonDominatedSet, SRConfig.popSize)   # --- TODO: Skip it, take it all
        endTime = time.time()
        print(f"\nFine-tuning phase time: {(endTime-startTime):2.1f} seconds\n")
        return tunedNonDominatedSet

    def pruneAndTunePopulation(
        self, pop: list[SubTopology], regularize: bool = False, generation: int = 0
    ):
        print(f"\nPruning the non-dominated set", end="")
        startTime = time.time()
        nonDominatedSet, _ = splitPopulationByDomination(
            pop, front=0, criteria=SubTopology.keysAll
        )
        nonDominatedSet = pop
        prunedNonDominatedSet = []
        # ---
        if SRConfig.pruningIters > 0:
            for i, individual_orig in enumerate(nonDominatedSet):
                individual = individual_orig.cloneSubTopology()
                print(
                    f'\n{i}.\t{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f}  ->  ',
                    end="",
                )
                # --- apply pruning strategies
                pruned = individual.pruneWeights()
                pruned = pruned or individual.pruneUnitsWithBelowThresholdOutput()
                # ---
                if pruned:
                    individual.train_nn(
                        learning_subset_size=SRConfig.learning_subset_size,
                        learningSteps=SRConfig.pruningIters,
                        learningRate=SRConfig.learning_rate_newborn,
                        regularize=regularize,
                        clipWeights=regularize,
                        constrain=True,
                        reduceLearningRate=False,
                        alternatePrefs=SRConfig.alternatePrefs,
                        printStep=0,
                    )
                    prunedNonDominatedSet.append(individual)
                    print(
                        f'{individual.complexity} / {individual.losses["rmse_valid"]:2.8f} / {individual.losses["rmse_constr"]:2.8f} / topologyId: {individual.masterTopologyId}',
                        end="",
                    )
                else:
                    print(f"unchanged", end="")
        # ---
        endTime = time.time()
        print(f"\nPruning phase time: {(endTime - startTime):2.1f} seconds\n")
        return prunedNonDominatedSet

    def finalTuning(self):
        print(f"\nFinal tuning of the non-dominated set", end="")
        # --- TODO: remove, just for debugging
        # individual = self.archiveBest[0]
        # print(f'\noriginal: {individual.losses["rmse_valid"]:2.7f} / {individual.losses["rmse_constr"]:2.7f}')
        # individual.train_nn(learningSteps=0,
        #                    learningRate=SRConfig.learning_rate_finaltune,
        #                    finalTuning=True,
        #                    regularize=False,
        #                    clipWeights=True,
        #                    constrain=True,
        #                    reduceLearningRate=False,
        #                    alternatePrefs=False,
        #                    printStep=1)
        # print(f'recalculated: {individual.losses["rmse_valid"]:2.7f} / {individual.losses["rmse_constr"]:2.7f}')
        # ---
        SRData.set_data_for_final_tuning(train_val_file=SRConfig.train_data)
        SRConstraints.load_constraints(path=SRConfig.constraints_file)
        startTime = time.time()
        tunedNonDominatedSet = []
        # --- TODO: remove, just for debugging
        # individual = self.archiveBest[0]
        # newIndividual = individual.cloneSubTopology()
        # newIndividual.train_nn(learningSteps=0,
        #                        learningRate=SRConfig.learning_rate_finaltune,
        #                        finalTuning=True,
        #                        regularize=False,
        #                        clipWeights=True,
        #                        constrain=True,
        #                        reduceLearningRate=False,
        #                        alternatePrefs=False,
        #                        printStep=1)
        # print(f'newIndividual: {newIndividual.losses["rmse_valid"]:2.7f} / {newIndividual.losses["rmse_constr"]:2.7f}')

        # ---
        if SRConfig.finalIters > 0:
            for i, individual in enumerate(self.archiveBest):
                if i == 15:
                    break
                # --- NO regularization
                newIndividual = individual.cloneSubTopology()
                print(
                    f'\n{i}. NO regularization: \t{individual.complexity} / {individual.losses["rmse_valid"]:2.7f} / {individual.losses["rmse_constr"]:2.7f}  ->  ',
                    end="",
                )
                newIndividual.train_nn(
                    learningSteps=SRConfig.finalIters,
                    learningRate=SRConfig.learning_rate_finaltune,
                    finalTuning=True,
                    regularize=False,
                    clipWeights=True,
                    constrain=True,
                    reduceLearningRate=False,
                    alternatePrefs=False,
                    printStep=1,
                )
                print(
                    f'{newIndividual.complexity} / {newIndividual.losses["rmse_valid"]:2.7f} / {newIndividual.losses["rmse_constr"]:2.7f}',
                    end="",
                )
                tunedNonDominatedSet.append(newIndividual)
        # self.population = self.mergePopulations(nonDominatedSet, tunedNonDominatedSet, SRConfig.popSize)
        # self.population = self.mergePopulations(tunedNonDominatedSet, self.archivedNondominatedFront, SRConfig.popSize)
        # self.population, _ = splitPopulationByDomination(self.population, front=0, criteria=SubTopology.keysAll)
        self.population = tunedNonDominatedSet
        endTime = time.time()
        print(f"\nFinal tuning phase time: {(endTime-startTime):2.1f} seconds\n")

    def calculatePopulationMedianPerformanceValues(self, population: list[SubTopology]):
        populationMedianValues: dict[str, float] = {}
        for k in SubTopology.keysPerformance:
            a = [sp.losses[k] for sp in population if sp.losses[k] > 0.0]
            # print(f'{k}: {a}')
            if not a:
                populationMedianValues[k] = (
                    0.0  # --- there is no solution with a positive losses[k] value
                )
            elif len(a) > 2:
                populationMedianValues[k] = statistics.quantiles(a, n=4)[
                    SRConfig.historyQualityQuantile
                ]  # --- 0
            else:
                populationMedianValues[k] = np.max(a)
        return populationMedianValues

    def calculatePopulationMaximumPerformanceValues(
        self, population: list[SubTopology]
    ):
        populationMaximumValues: dict[str, float] = {}
        for k in SubTopology.keysPerformance:
            a = [sp.losses[k] for sp in population if sp.losses[k] > 0.0]
            print(f"{k}: {a}")
            if not a:
                populationMaximumValues[k] = (
                    1.0e10  # --- there is no solution with a positive losses[k] value
                )
            else:
                populationMaximumValues[k] = np.max(a)
        # print(f"Whole population's maximum values:")
        # for k in SubTopology.keysPerformance:
        #     print(f"\t{k}: {populationMaximumValues[k]:2.8f}")
        return populationMaximumValues

    def resetHistoryAndArchivedNondominatedFront(self):
        self.archivedNondominatedFront: list[SubTopology] = []
        for topologyId in range(len(MOEA.masterTopologies)):
            self.masterTopologies[topologyId].initializeNetworkHistory()
        self.archivedNondominatedFront = []

    def runGeneration(self, generation: int, fileName: str = None, mode: str = "a"):
        startTime = time.time()
        startBackprops: int = SubTopology.backpropCount
        self.setNodeUtilizations()

        # ----------------------------------------------------------------------
        # --- Update currNondominatedFrontMaximumValues
        # --- No distinction is made between individual classes of solutions.
        # ----------------------------------------------------------------------
        # currNondominatedFront, _ = splitPopulationByDomination(self.population, front=0, criteria=SubTopology.keysAll)
        # if generation < self.noregularizationPhaseStart + SRConfig.regFreeInitialPhase:
        #     currNondominatedFrontMaximumValues = self.calculatePopulationMedianPerformanceValues(self.population)
        # else:
        #     currNondominatedFrontMaximumValues = self.calculatePopulationMaximumPerformanceValues(currNondominatedFront)
        #     # currNondominatedFrontMaximumValues = self.calculatePopulationMaximumPerformanceValues(self.archivedNondominatedFront)
        # for k in currNondominatedFrontMaximumValues.keys():
        #     currNondominatedFrontMaximumValues[k] *= SRConfig.nsgaAcceptanceScalingFactor
        # # ---
        # print(f"\nCurrent population's nondominated front maximum values")
        # for k in currNondominatedFrontMaximumValues.keys():
        #     print(f"\t{k}: {currNondominatedFrontMaximumValues[k]:2.8f}")
        # -------------------------------------------------------
        # --- Update currNondominatedFrontMaximumValues[]
        # --- Individual topologies are considered separately.
        # -------------------------------------------------------
        currNondominatedFront: list[list] = [
            [] for _ in range(len(MOEA.masterTopologies))
        ]  # --- list of fronts for each topologyId
        currNondominatedFrontMaximumValues = [
            {} for _ in range(len(MOEA.masterTopologies))
        ]  # --- list of dictionaries with the performance values for each topologyId
        # ----------------------------------------------------
        # --- Set the constraint violation terms
        # ----------------------------------------------------
        if (
            generation == self.noregularizationPhaseStart + SRConfig.regFreeInitialPhase
        ) and SRConstraints.constraintModules:
            for st in self.population:
                c_raw, c_weighted = st.calculateConstrainViolationsRMSE()
                st.losses["rmse_constr"] = c_raw
        # ---
        for topologyId in range(len(MOEA.masterTopologies)):
            tmpPop = [st for st in self.population if st.masterTopologyId == topologyId]
            currNondominatedFront[topologyId], _ = splitPopulationByDomination(
                tmpPop, front=0, criteria=SubTopology.keysAll
            )
            if (
                generation
                < self.noregularizationPhaseStart + SRConfig.regFreeInitialPhase
            ):
                currNondominatedFrontMaximumValues[topologyId] = (
                    self.calculatePopulationMedianPerformanceValues(tmpPop)
                )
            else:
                currNondominatedFrontMaximumValues[topologyId] = (
                    self.calculatePopulationMaximumPerformanceValues(
                        currNondominatedFront[topologyId]
                    )
                )
            for k in currNondominatedFrontMaximumValues[topologyId].keys():
                currNondominatedFrontMaximumValues[topologyId][
                    k
                ] *= SRConfig.nsgaAcceptanceScalingFactor
            # ---
            print(f"\nCurrent population's nondominated front maximum values")
            for k in currNondominatedFrontMaximumValues[topologyId].keys():
                print(
                    f"\t{k}: {currNondominatedFrontMaximumValues[topologyId][k]:2.8f}"
                )
        # ---
        # self.archivedNondominatedFront = self.updateArchive(currNondominatedFront, self.archivedNondominatedFront, 2*SRConfig.popSize)
        # self.archivedNondominatedFront, _ = splitPopulationByDomination(self.archivedNondominatedFront, front=0, criteria=SubTopology.keysAll)
        # ---
        print("\nCurrent archivedNondominatedFront:")
        if fileName:
            f = open(fileName, mode=mode)
            f.write("\nCurrent archivedNondominatedFront:\n")
        for st in self.archivedNondominatedFront:
            print(
                f' \t{st.complexity} / {st.losses["rmse_valid"]:2.8f} / {st.losses["rmse_constr"]:2.12f} / {st.masterTopologyId}'
            )
            if fileName:
                f.write(
                    f' \t{st.complexity} / {st.losses["rmse_valid"]:2.8f} / {st.losses["rmse_constr"]:2.12f} / {st.masterTopologyId}\n'
                )
        if fileName:
            f.write(f"\n")
            f.close()

        # -----------------------------------------------
        # --- Create intermediate population
        # --- and take the non-dominated front
        # -----------------------------------------------
        perturb: bool = (
            (SRConfig.perturbation > 0.0)
            and (generation > 0)
            and (generation % SRConfig.perturbationStep == 0)
        )
        if generation < self.noregularizationPhaseStart + SRConfig.regFreeInitialPhase:
            regularize = False  # --- ORIGINAL: False
        else:
            regularize = True
        if perturb:
            # ------------------------------------------------------------------------------------------
            # --- Compose a new self.population using non-dominated subTopologies from the last epoch
            # --- and non-dominated perturbed subTopologies
            # ------------------------------------------------------------------------------------------
            # if SRConfig.perturbationUseArchive:
            #     if self.archivedNondominatedFront:
            #         self.population = self.mergePopulations(self.population, self.archivedNondominatedFront, SRConfig.popSize)
            # ---------------------------------------------------------------------------------------------------
            interpop: list[SubTopology] = []
            while len(interpop) < 2 * SRConfig.popSize:
                for topologyId in range(len(MOEA.masterTopologies)):
                    for k, st in enumerate(
                        currNondominatedFront[topologyId]
                    ):  # --- ORIGINAL: self.population, TODO: currNondominatedFront
                        pertSt: SubTopology = st.cloneSubTopology()
                        pertSt.activateAllInactiveNodes()
                        self.masterTopologies[
                            pertSt.masterTopologyId
                        ].rectifyActiveNodesConnectivity(pertSt)
                        pertSt.train_nn(
                            learningSteps=SRConfig.totalItersNewbornPerturbed,
                            learningRate=SRConfig.learning_rate_newborn,
                            reduceLearningRate=False,
                            regularize=False,  # --- ORIGINAL: True, TODO: False
                            clipWeights=False,
                            constrain=False,
                        )
                        interpop.append(pertSt)
            self.population = nonDominatedSorting(
                interpop, len(self.population)
            )  # --- ORIGINAL
            # ---
            for topologyId in range(len(self.masterTopologies)):
                self.lastNondominatedMedianActiveNodes[topologyId] = 100.0
            self.noregularizationPhaseStart = generation
            # ------------------------------------------------
            # --- History reset: for all masterTopologies
            # ------------------------------------------------
            if SRConfig.perturbationResetHistory:
                for topologyId in range(len(MOEA.masterTopologies)):
                    self.masterTopologies[topologyId].initializeNetworkHistory()
                self.archivedNondominatedFront = []
        else:
            interPop = self.runNSGAPhase(regularize=regularize)
            # ----------------------------------------------------------
            # --- Run runWeightTuningPhase() per topologyId
            # ----------------------------------------------------------
            tunedInterPop = []
            for topologyId in range(len(MOEA.masterTopologies)):
                tmpPop = [st for st in interPop if st.masterTopologyId == topologyId]
                nonDominatedSet, _ = splitPopulationByDomination(
                    tmpPop, front=0, criteria=SubTopology.keysAll
                )
                tunedPerTopologyId = self.runWeightTuningPhase(
                    nonDominatedSet,
                    regularize=regularize,
                    deactivateBelowThresholdUnits=regularize,
                    fileName=fileName,
                )  # --- ORIGINAL: tune only non-dominated individuals
                tunedInterPop.extend(tunedPerTopologyId)

            # ------------------------------------------------------------
            # --- Take only nondominated solutions of the tunedInterPop
            # ------------------------------------------------------------
            # tunedInterPop, _ = splitPopulationByDomination(tunedInterPop, front=0, criteria=SubTopology.keysAll)

            # ------------------------------------------------------------------
            # --- Take new subTopologies that have all its performance values
            # --- no worse than the currNondominatedFrontMaximumValues
            # ------------------------------------------------------------------
            tunedInterPopAllPerformancesBelowMax: list[SubTopology] = []
            for st in tunedInterPop:
                if dictIsAllBelowMedian(
                    st.losses, currNondominatedFrontMaximumValues[st.masterTopologyId]
                ):
                    tunedInterPopAllPerformancesBelowMax.append(st)
            print(
                f"\ntunedInterPopAllPerformancesBelowMax ({len(tunedInterPopAllPerformancesBelowMax)})"
            )
            for st in tunedInterPopAllPerformancesBelowMax:
                print(
                    f'\t{st.complexity} / {st.losses["rmse_valid"]:2.7e} / {st.losses["rmse_constr"]:2.7e} / {st.masterTopologyId}'
                )

            # ----------------------------------------------------------------------
            # --- Take new subTopologies that have at least one performance value
            # --- no worse than the currNondominatedFrontMaximumValues
            # ----------------------------------------------------------------------
            if not tunedInterPopAllPerformancesBelowMax:
                tunedInterPopAllPerformancesBelowMax: list[SubTopology] = []
                for st in tunedInterPop:
                    if dictIsAtLeastOneBelowMedian(
                        st.losses,
                        currNondominatedFrontMaximumValues[st.masterTopologyId],
                    ):
                        tunedInterPopAllPerformancesBelowMax.append(st)
                print(
                    f"\ntunedInterPopSomePerformancesBelowMax ({len(tunedInterPopAllPerformancesBelowMax)})"
                )
                for st in tunedInterPopAllPerformancesBelowMax:
                    print(
                        f'\t{st.complexity} / {st.losses["rmse_valid"]:2.7e} / {st.losses["rmse_constr"]:2.7e} / {st.masterTopologyId}'
                    )

            # ----------------------------------------------------------------------------------
            # --- Take new subTopologies that dominate some of the currNondominatedFront ones
            # --- w.r.t. performance criteria
            # ------------------------------------------------------------------
            tunedInterPopDominating: list[SubTopology] = []
            for interPopInd in tunedInterPop:
                if (interPopInd not in tunedInterPopDominating) and (
                    interPopInd not in tunedInterPopAllPerformancesBelowMax
                ):
                    for nonDominatedFrontInd in currNondominatedFront[
                        interPopInd.masterTopologyId
                    ]:
                        interPopDominates, nonDominatedfrontDominates, iEqualsj = (
                            getMutualDominance(
                                interPopInd,
                                nonDominatedFrontInd,
                                criteria=SubTopology.keysPerformance,
                            )
                        )
                        if interPopDominates:
                            tunedInterPopDominating.append(interPopInd)
                            break
            print(f"\ntunedInterPopDominating ({len(tunedInterPopDominating)})")
            for st in tunedInterPopDominating:
                print(
                    f'\t{st.complexity} / {st.losses["rmse_valid"]:2.7e} / {st.losses["rmse_constr"]:2.7e} / {st.masterTopologyId}'
                )

            # ----------------------------------------------------------
            # --- Run fineTuneNonDominatedFront() per topologyId
            # ----------------------------------------------------------
            fineTunedNonDominatedSet = []
            for topologyId in range(len(MOEA.masterTopologies)):
                populationToTune = [
                    st for st in self.population if st.masterTopologyId == topologyId
                ]
                tunedPerTopologyId = self.fineTuneNonDominatedFront(
                    pop=populationToTune,
                    topologyId=topologyId,
                    regularize=regularize,
                    generation=generation,
                )
                fineTunedNonDominatedSet.extend(tunedPerTopologyId)
            # ---
            # tunedPerTopologyId = self.fineTuneNonDominatedFront(pop=self.population, topologyId=topologyId, regularize=regularize, generation=generation)
            # fineTunedNonDominatedSet.extend(tunedPerTopologyId)

            # -----------------------------------------------------------------------------------------
            # --- Apply non-dominated sorting to the 'res' population and take 'size' best solutions
            # -----------------------------------------------------------------------------------------
            self.population.extend(tunedInterPopAllPerformancesBelowMax)
            self.population.extend(fineTunedNonDominatedSet)
            for st in tunedInterPopDominating:
                if st not in self.population:
                    self.population.append(st)
            self.population = nonDominatedSorting(
                self.population, SRConfig.popSize, criteria=SubTopology.keysPerfNodes
            )  # --- takes into account topologyIds

        # ---------------------------------------------------------------------------------------
        # ------------------------------------- History -----------------------------------------
        # ---------------------------------------------------------------------------------------
        if generation >= self.noregularizationPhaseStart + SRConfig.regFreeInitialPhase:
            # ----------------------------------------------------------------------------------------------------------
            # --- Update: currNondominatedFront, currNondominatedFrontMaximumValues, self.archivedNondominatedFront
            # ----------------------------------------------------------------------------------------------------------
            currNondominatedFront: list[list] = [
                [] for _ in range(len(MOEA.masterTopologies))
            ]  # --- list of fronts for each topologyId
            currNondominatedFrontMaximumValues = [
                {} for _ in range(len(MOEA.masterTopologies))
            ]  # --- list of dictionaries with the performance values for each topologyId
            for topologyId in range(len(MOEA.masterTopologies)):
                tmpPop = [
                    st for st in self.population if st.masterTopologyId == topologyId
                ]
                currNondominatedFront[topologyId], _ = splitPopulationByDomination(
                    tmpPop, front=0, criteria=SubTopology.keysAll
                )
                currNondominatedFrontMaximumValues[topologyId] = (
                    self.calculatePopulationMaximumPerformanceValues(
                        currNondominatedFront[topologyId]
                    )
                )
                # for k in currNondominatedFrontMaximumValues[topologyId].keys():
                #     currNondominatedFrontMaximumValues[topologyId][k] *= SRConfig.nsgaAcceptanceScalingFactor
            # ---------------------------------------------------------------
            # --- Update: self.archivedNondominatedFront, self.archiveBest
            # ---------------------------------------------------------------
            populationToUpdateArchive: list[SubTopology] = []
            for topologyId in range(len(MOEA.masterTopologies)):
                for st in self.population:
                    if dictIsAllBelowMedian(
                        st.losses,
                        currNondominatedFrontMaximumValues[st.masterTopologyId],
                    ):
                        populationToUpdateArchive.append(st)
            self.archivedNondominatedFront = self.updateArchive(
                populationToUpdateArchive,
                self.archivedNondominatedFront,
                2 * SRConfig.popSize,
            )
            self.archivedNondominatedFront, _ = splitPopulationByDomination(
                self.archivedNondominatedFront, front=0, criteria=SubTopology.keysAll
            )
            print("\nNew archivedNondominatedFront:")
            for st in self.archivedNondominatedFront:
                print(
                    f' \t{st.complexity} / {st.losses["rmse_valid"]:2.8f} / {st.losses["rmse_constr"]:2.8f} / {st.masterTopologyId}'
                )
            # --- Update the best-so-far archive
            self.archiveBest.extend(self.archivedNondominatedFront)
            self.archiveBest, _ = splitPopulationByDomination(
                self.archiveBest, front=0, criteria=SubTopology.keysAll
            )
            # ---
            if len(MOEA.masterTopologies) == 1:
                # --------------------------------------
                # --- History update for topologyId=0
                # --------------------------------------
                historyUpdateList: list[SubTopology] = (
                    self.chooseSubTopologiesForHistoryUpdate_E(
                        pop=currNondominatedFront[0],
                        scalingFactor=SRConfig.historyScalingFactor,
                    )
                )
                if historyUpdateList is not None:
                    # --- History reset
                    MOEA.masterTopologies[0].initializeNetworkHistory()
                    # ---
                    partitionSize = SRConfig.historySize
                    for st in historyUpdateList:
                        MOEA.masterTopologies[0].updateNetworkHistory(
                            st, partitionSize=partitionSize, byTopologyId=-1
                        )
            else:
                # ---------------------------------------------
                # --- History update for each masterTopology
                # ---------------------------------------------
                for topologyId in range(len(MOEA.masterTopologies)):
                    # popForUpdate = [st for st in self.population if st.masterTopologyId == topologyId]    # --- take only individuals with the given topologyId
                    # currNondominatedFront, _ = splitPopulationByDomination(popForUpdate, front=0, criteria=SubTopology.keysAll)
                    historyUpdateList: list[SubTopology] = (
                        self.chooseSubTopologiesForHistoryUpdate_E(
                            pop=currNondominatedFront[topologyId],
                            scalingFactor=SRConfig.historyScalingFactor,
                        )
                    )
                    if historyUpdateList is not None:
                        # --- History reset
                        self.masterTopologies[topologyId].initializeNetworkHistory()
                        # ---
                        partitionSize = SRConfig.historySize
                        for st in historyUpdateList:
                            self.masterTopologies[topologyId].updateNetworkHistory(
                                st, partitionSize=partitionSize, byTopologyId=topologyId
                            )
        # ---------------------------------
        # --- Re-assign masterTopologyId
        # ---------------------------------
        if (
            (SRConfig.nichingInterval > 0)
            and (generation > 0)
            and (generation % SRConfig.nichingInterval == 0)
        ):
            id = 0
            for st in self.population:
                st.masterTopologyId = id % len(self.masterTopologies)
                id += 1

        # -------------------------------------------------------------------------------------
        endTime = time.time()
        print(f"\nGeneration")
        print(f"\texecution time: {(endTime-startTime):2.1f} seconds")
        print(f"\tbackprops: {SubTopology.backpropCount-startBackprops}")
        print(f"\nbackpropCount: {SubTopology.backpropCount}\n")

    def runEvolution(self):
        self.noregularizationPhaseStart = 0
        generationBackprops: list[int] = [0]
        startBackprops = SubTopology.backpropCount
        for g in range(SRConfig.eaGenerations):
            medianGenerationBackprops = int(np.median(generationBackprops))
            consumedBackprops = (
                SubTopology.backpropCount
                + int(medianGenerationBackprops / 3)
                + 5 * SRConfig.finalIters
            )
            if consumedBackprops >= SRConfig.maxBackpropIters:
                print(f"\n\n========================================================")
                print(
                    f"Maximum number of backprop iterations {SRConfig.maxBackpropIters} reached at "
                    f"\n\t - generation: {g}"
                    f"\n\t - backpropCount: {SubTopology.backpropCount}"
                    f"\n\t - len(self.archiveBest): {len(self.archiveBest)}"
                    f"\n\t - medianGenerationBackprops: {medianGenerationBackprops}"
                )
                print(f"========================================================\n\n")
                break
            print(f"\n---------------------\nGeneration: {g}")
            self.runGeneration(generation=g, fileName=SRConfig.progress_file)
            self.printPopulation(
                pop=self.population,
                text=f"{g}. population",
                fileName=SRConfig.progress_file,
            )
            self.printNondominated(fileName=SRConfig.progress_file)
            self.printNodeUtilizations(fileName=SRConfig.progress_file)
            if SRConfig.historyType == 0:
                print("\n--------------------------------")
                print("\n--- Master topologies' history:")
                for topologyId in range(len(self.masterTopologies)):
                    self.masterTopologies[topologyId].printNetworkHistory()
            else:
                self.masterTopologies[0].printNetworkHistoryPerSize()
            if (
                (g + 1) % SRConfig.saveBestTopologiesStep == 0
            ) or g == SRConfig.eaGenerations - 1:
                if self.archivedNondominatedFront:
                    tmpPopulation = self.mergePopulations(
                        self.population,
                        self.archivedNondominatedFront,
                        SRConfig.popSize,
                    )
                else:
                    tmpPopulation = self.population
                prefixText = ""
                if self.epoch >= 0:
                    prefixText += f"solution_e={self.epoch}_g={g}_"
                else:
                    prefixText += f"solution_g={g}_"
                # --- ORIGINAL
                # if self.archiveBest:
                #     self.saveBestSoFarToFile(population=self.archiveBest, prefixText=prefixText, nbSaved=10, savePickle=SRConfig.savePickleIntermediate)
                # else:
                #     nondominated, _ = splitPopulationByDomination(self.population, front=0, criteria=SubTopology.keysAll)
                #     self.saveBestSoFarToFile(population=nondominated, prefixText=prefixText, nbSaved=10, savePickle=SRConfig.savePickleIntermediate)
                # --- PREFER: self.archivedNondominatedFront
                if self.archivedNondominatedFront:
                    self.saveBestSoFarToFile(
                        population=self.archivedNondominatedFront,
                        prefixText=prefixText,
                        nbSaved=20,
                        savePickle=SRConfig.savePickleIntermediate,
                    )
                else:
                    nondominated, _ = splitPopulationByDomination(
                        self.population, front=0, criteria=SubTopology.keysAll
                    )
                    self.saveBestSoFarToFile(
                        population=nondominated,
                        prefixText=prefixText,
                        nbSaved=10,
                        savePickle=SRConfig.savePickleIntermediate,
                    )
            # ---
            k: int = SubTopology.backpropCount - startBackprops
            generationBackprops.append(k)
            startBackprops = SubTopology.backpropCount

        # --------------------------------------------------
        # --- Update MOEA.archivedNondominatedFront
        # --------------------------------------------------
        # if self.archivedNondominatedFront:
        #     self.population = self.mergePopulations(self.population, self.archivedNondominatedFront, SRConfig.popSize)
        # self.archivedNondominatedFront, _ = splitPopulationByDomination(self.population, front=0, criteria=SubTopology.keysAll)
        # ---
        currNondominatedFront, _ = splitPopulationByDomination(
            self.population, front=0, criteria=SubTopology.keysAll
        )
        self.archivedNondominatedFront = self.updateArchive(
            currNondominatedFront, self.archivedNondominatedFront, 2 * SRConfig.popSize
        )
        self.archivedNondominatedFront, _ = splitPopulationByDomination(
            self.archivedNondominatedFront, front=0, criteria=SubTopology.keysAll
        )

        # --- Update the best-so-far archive
        self.archiveBest.extend(self.archivedNondominatedFront)
        self.archiveBest, _ = splitPopulationByDomination(
            self.archiveBest, front=0, criteria=SubTopology.keysAll
        )

        # ---
        self.setNodeUtilizations()
        self.printNodeUtilizations(fileName=SRConfig.progress_file)
