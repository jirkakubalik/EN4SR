import copy
import numpy as np

import SRConfig
import SRData
from NNLayer import NNLayer
from SubTopology import SubTopology
# from utils import nnTopology
from NodeHistory import NodeHistory
from NodeHistoryPerSize import NodeHistoryPerSize
from InputConfiguration import InputConfiguration
from SRUnits import NNUnit, UnitMultiply, UnitDivide, \
    UnitSin, UnitCos, UnitTanh, UnitArcTan, \
    UnitSquare, UnitSqrt, UnitCube, \
    UnitIdent, UnitIdent1, MyTFVariable


def checkConfigShape(nodeWeights: list[MyTFVariable], configWeights: list[np.ndarray]):
    for w, c in zip(nodeWeights, configWeights):
        if w.v.shape != c.shape:
            return False
    return True


class NodeUtilizations():
    def __init__(self, nnLayers: list[NNLayer]):
        self.k = 0   # --- the number of subTopologies collected in the nodeUtilizations
        self.nodeUtilizationsAbs: list[list[int]] = []      # --- absolute utilization values for each node
        self.nodeUtilizationsRel: list[list[float]] = []    # --- utilization value from interval [0, 1], i.e., a proportion
                                                            # --- of active node occurences in the population
        for layer in nnLayers:
            self.nodeUtilizationsAbs.append(len(layer.units) * [0])
            self.nodeUtilizationsRel.append(len(layer.units) * [0.])


    def reset(self):
        self.k = 0.
        newUtilsAbs = []
        newUtilsRel = []
        for layer in self.nodeUtilizationsAbs:
            newUtilsAbs.append(len(layer) * [0])
            newUtilsRel.append(len(layer) * [0.])
        self.nodeUtilizationsAbs = newUtilsAbs
        self.nodeUtilizationsRel = newUtilsRel


    def updateNodeUnitilizations(self, subTopology: SubTopology):
        self.k += 1.
        for l, layer in enumerate(subTopology.activeNodeCoordinates):
            for n, node in enumerate(layer):
                self.nodeUtilizationsAbs[l][n] += subTopology.activeNodeCoordinates[l][n]
                self.nodeUtilizationsRel[l][n] = self.nodeUtilizationsAbs[l][n] / self.k


    def printNodeUtilizations(self):
        print(f'\nNode utilizations: {self.k} subTopologies', end='')
        for l, layer in enumerate(self.nodeUtilizationsAbs):
            print('\n\t', end='')
            for n, node in enumerate(layer):
                print(f'  {self.nodeUtilizationsRel[l][n]:1.2f}', end='')
        print('')


class MasterTopology():
    # networkHistory: list[list[NodeHistory]] = []  # --- a list of history configurations for each node
    #                                               # --- outer list of layers, inner list of layer'nodes and their configs
    networkHistoryPerSize: list[list[NodeHistoryPerSize]] = []
    lastSubTopologiesAddedToHistory: list[int] = [1000] * SRConfig.historySize  # --- last SRConfig.historySize subTopologies' nbOfActiveNodes

    def __init__(self, inputSize=0, layer_defs=None, identities=None):
        self.nnLayers: list[NNLayer] = []   # --- outer list of layers, inner list of layer'nodes
        self.networkHistory = []            # --- a list of history configurations for each node
                                            # --- outer list of layers, inner list of layer'nodes and their configs
        MasterTopology.networkHistoryPerSize = []
        MasterTopology.lastSubTopologiesAddedToHistory = [1000] * SRConfig.historySize
        # --- Forward pass
        for l in range(len(layer_defs)):
            layer_i = self.create_layer(l, in_size=inputSize, unit_types=layer_defs[l], identities=identities[l])
            inputSize = len(layer_i.units)
            newHistoryLayer = list(NodeHistory([]) for _ in layer_i.units)
            self.networkHistory.append(newHistoryLayer)
            newHistoryLayer = list(NodeHistoryPerSize({}) for _ in layer_i.units)
            MasterTopology.networkHistoryPerSize.append(newHistoryLayer)
        self.nodeUtilizations = NodeUtilizations(self.nnLayers)


    def updateNodeUtilizations(self, subTopology: SubTopology):
        self.nodeUtilizations.updateNodeUnitilizations(subTopology)


    def printNodeUtilizations(self):
        self.nodeUtilizations.printNodeUtilizations()


    def resetNodeUtilizations(self):
        self.nodeUtilizations.reset()


    def create_layer(self, layerId, in_size, unit_types=None, identities=True):
        nnLayer = NNLayer(in_size, layerId)
        # --- Units
        for u in unit_types.keys():
            for i in range(unit_types[u]):
                nnLayer.add_unit(u())
        # --- Identities
        if identities:
            for i in range(in_size):
                newUnit = UnitIdent1(i)
                if (layerId == 0) and (i < SRData.SRData.x_data.shape[1]):
                    newUnit.isVariable = True
                elif (layerId > 0) and (self.nnLayers[layerId-1].units[i].isVariable):    # --- current UnitIdent1 points to a previous layer UnitIdent1 that represents a variable
                    newUnit.isVariable = True
                nnLayer.add_unit(newUnit)
        # ---
        self.nnLayers.append(nnLayer)
        print(f'nnLayer_{layerId}: units = {len(nnLayer.units)}, learnableParams = {len(nnLayer.learnableParams)}')
        return nnLayer


    def generateRandomSubTopology(self):
        """
        Generates randomly a subTopology starting from the output layer and proceeding backwards.
        Backward pass, from the output layer to the first one.
        :return: A clone of the MasterTopology with its weights set.
        """
        subTopology = SubTopology()
        subTopology.activeNodeCoordinates = []
        # ---
        activeNodes: list[bool] = [True]
        # -------------------------------------------------
        # --- Backward pass, start with the output layer
        # -------------------------------------------------
        for l in reversed(range(len(self.nnLayers))):
            subTopology.activeNodeCoordinates.insert(0, activeNodes)
            if l > 0:
                prevLayer = self.nnLayers[l-1]
            else:
                prevLayer = None
            newLayer, activeNodes = self.nnLayers[l].randomly_generate_layer(layerId=l,
                                                                             layerActiveNodes=activeNodes,
                                                                             inputLayer=prevLayer)
            subTopology.learnableParams.extend(newLayer.learnableParams)
            subTopology.nnLayers.insert(0, newLayer)
        # ---------------------------------------------------
        # --- Rectify connectivity and update active nodes
        # ---------------------------------------------------
        subTopology.setProxyNames()
        self.rectifyActiveNodesConnectivity(subTopology)
        subTopology.updateActiveNodes()
        # ---
        return subTopology


    def subTopologyCrossoverB(self, subTopologyA: SubTopology=None, subTopologyB: SubTopology=None, historyA=None, historyB=None):
        """
        Forward pass!
        Inherits activeNodeCoordinates[l][n] either from subTopologyA or from subTopologyB.
        Weights are inherited with a probability SRConfig.inheritWeights.
        Otherwise they are taken from an existing config, if applicable, or randomly regenerated anew.
        :param subTopologyA:
        :param subTopologyB:
        :return: offspring ... a new subTopology as a combinantion of the two parents
                 changed ... True, if the offspring differs from both parents. False, otherwise.
        """
        res: SubTopology = subTopologyA.cloneSubTopology()
        # ---
        activeNodes = []
        for l, layer in enumerate(res.nnLayers):
            activeNodesLayer = []
            # ---
            for n, node in enumerate(layer.units):
                if SRConfig.r.random() < SRConfig.inheritFromFirstParent:
                    # -------------------------
                    # --- Use subTopologyA
                    # -------------------------
                    currValue = subTopologyA.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    # ---
                    if currValue:
                        if (l < len(res.nnLayers)-1) and SRConfig.r.random() > SRConfig.inheritWeights:
                            # ---------------------------------------------
                            # --- DON'T inherit weights; choose new ones
                            # ---------------------------------------------
                            config: InputConfiguration
                            if SRConfig.r.random() > SRConfig.historyUsageProb:     # --- ORIGINAL: commented
                                config = None
                            elif SRConfig.historyType == 0:
                                config = self.networkHistory[l][n].getRandomConfig()
                            else:
                                if SRConfig.historyConfigPreferences == 0:
                                    config = MasterTopology.networkHistoryPerSize[l][n].getConfigRandom()
                                elif SRConfig.historyConfigPreferences == -1:
                                    config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=True)
                                else:
                                    config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=False)
                            # ---
                            if config:
                                for w, c in zip(node.weights, config.nodeWeights):
                                    w.set_value_mask(c)
                            else:
                                # --------------------------------------------
                                # --- UnitIdent1 node: set its default
                                # --------------------------------------------
                                if 'UnitIdent1' in node.name:
                                    node.reset_weights_and_bias()
                                else:
                                    # --------------------------------------------
                                    # --- Ordinary unit: generate random values
                                    # --------------------------------------------
                                    for w in node.weights:
                                        inputLayerActiveNodes: list[bool]
                                        if ' b' in w.v.name:    # --- bias
                                            inputLayerActiveNodes = [True]
                                        elif l == 0:
                                            inputLayerActiveNodes = layer.inSize * [True]
                                        else:
                                            # inputLayerActiveNodes = subTopologyA.activeNodeCoordinates[l-1]
                                            inputLayerActiveNodes = activeNodes[l-1]
                                        w.setRandomWeights(layerId=l, inputLayerActiveNodes=inputLayerActiveNodes)
                    else:
                        # ----------------------------------------------------
                        # --- It's an inactive node, set all values to zero
                        # ----------------------------------------------------
                        for w in node.weights:
                            w.set_allzero_value_mask()
                else:
                    # -----------------------
                    # --- Inherit from B
                    # -----------------------
                    currValue = subTopologyB.activeNodeCoordinates[l][n]
                    activeNodesLayer.append(currValue)
                    # --- Change node weights to the weights of the second parent, subTopologyB
                    if not currValue:
                        for w in node.weights:
                            w.set_allzero_value_mask()
                    else:
                        if (l == len(res.nnLayers)-1) or SRConfig.r.random() < SRConfig.inheritWeights:
                            # --- INHERIT; use the old subTopologyB weights, if applicable
                            for wA, wB in zip(node.weights, subTopologyB.nnLayers[l].units[n].weights):
                                if wA.mask.shape == wB.mask.shape:
                                    wA.set_value_mask(wB.v.numpy())
                        else:
                            # --- DON'T inherit weights; choose new ones
                            config: InputConfiguration
                            if SRConfig.r.random() > SRConfig.historyUsageProb:     # --- ORIGINAL: commented
                                config = None
                            elif SRConfig.historyType == 0:
                                # config = self.networkHistory[l][n].getRandomConfig()        # --- ORIGINAL
                                config = historyB[l][n].getRandomConfig()
                            else:
                                if SRConfig.historyConfigPreferences == 0:
                                    config = MasterTopology.networkHistoryPerSize[l][n].getConfigRandom()
                                elif SRConfig.historyConfigPreferences == -1:
                                    config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=True)
                                else:
                                    config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=False)
                            # ---
                            if config:
                                for w, c in zip(node.weights, config.nodeWeights):
                                    w.set_value_mask(c)
                            else:
                                if 'UnitIdent1' in node.name:
                                    # --------------------------------------------
                                    # --- UnitIdent1 node: set its default
                                    # --------------------------------------------
                                    node.reset_weights_and_bias()
                                else:
                                    # --------------------------------------------
                                    # --- Ordinary unit: generate random values
                                    # --------------------------------------------
                                    for w in node.weights:
                                        inputLayerActiveNodes: list[bool]
                                        if ' b' in w.v.name:    # --- bias
                                            inputLayerActiveNodes = [True]
                                        elif l == 0:
                                            inputLayerActiveNodes = layer.inSize * [True]
                                        else:
                                            # inputLayerActiveNodes = subTopologyA.activeNodeCoordinates[l-1]
                                            inputLayerActiveNodes = activeNodes[l-1]
                                        w.setRandomWeights(layerId=l, inputLayerActiveNodes=inputLayerActiveNodes)
            activeNodes.append(activeNodesLayer)
        res.activeNodeCoordinates = activeNodes
        # ------------------------------------
        # --- Final check
        # ------------------------------------
        for l, activeNodesLayer in enumerate(res.activeNodeCoordinates):
            if sum(activeNodesLayer) == 0:
                # --- Activate some node
                n = SRConfig.r.integers(0, len(activeNodesLayer))
                activeNodesLayer[n] = True
                self.activateNode(res, l, n)
        # ---
        return res


    def nodeActivityMutation(self, l: int, n: int, value: bool):
        """
        Mutate the given activeNodeCoordinates element according to the corresponding node utilization.
        :param l: layer id
        :param n: position in the layer
        :param value: current value
        :return: new value
        """
        # --- mutate only positions with extreme node utilization value
        if self.nodeUtilizations.nodeUtilizationsRel[l][n] <= SRConfig.utilLow:
            if (not value) and (SRConfig.r.random() < SRConfig.nodeUndersampledMutationRate):
                return True
        elif self.nodeUtilizations.nodeUtilizationsRel[l][n] >= SRConfig.utilHigh:
            if (value) and (SRConfig.r.random() < SRConfig.nodeOversampledMutationRate):
                return False
        return value


    def networkActivityMutation(self, ind: SubTopology):
        """
        Mutate all activeNodeCoordinates elements with certain probability.
        :return:
        """
        changed: bool = False
        # for l in range(len(ind.activeNodeCoordinates)-1):     # --- ORIGINAL
        for l in range(len(ind.activeNodeCoordinates)):         # --- TODO
            for n in range(len(ind.activeNodeCoordinates[l])):
                value = ind.activeNodeCoordinates[l][n]
                if ((self.nodeUtilizations.nodeUtilizationsRel[l][n] <= SRConfig.utilLow and not value) or
                   (self.nodeUtilizations.nodeUtilizationsRel[l][n] >= SRConfig.utilHigh and value)) \
                    and (SRConfig.r.random() < SRConfig.nodeActivityMutationRate):
                    changed = True
                    ind.activeNodeCoordinates[l][n] = not value
                    if ind.activeNodeCoordinates[l][n]:
                        # --- node became active
                        config: InputConfiguration
                        if SRConfig.r.random() > SRConfig.historyUsageProb:
                            config = None
                        elif SRConfig.historyType == 0:
                            config = self.networkHistory[l][n].getRandomConfig()
                        else:
                            if SRConfig.historyConfigPreferences == 0:
                                config = MasterTopology.networkHistoryPerSize[l][n].getConfigRandom()
                            elif SRConfig.historyConfigPreferences == -1:
                                config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=True)
                            else:
                                config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=False)
                        # ---
                        if config:
                            for w, c in zip(ind.nnLayers[l].units[n].weights, config.nodeWeights):
                                w.set_value_mask(c)
                        else:
                            if 'UnitIdent1' in ind.nnLayers[l].units[n].name:
                                # --------------------------------------------
                                # --- UnitIdent1 node: set its default
                                # --------------------------------------------
                                ind.nnLayers[l].units[n].reset_weights_and_bias()
                            else:
                                # --------------------------------------------
                                # --- Ordinary unit: generate random values
                                # --------------------------------------------
                                for w in ind.nnLayers[l].units[n].weights:
                                    inputLayerActiveNodes: list[bool]
                                    if ' b' in w.v.name:    # --- bias
                                        inputLayerActiveNodes = [True]
                                    elif l == 0:
                                        inputLayerActiveNodes = ind.nnLayers[l].inSize * [True]
                                    else:
                                        inputLayerActiveNodes = ind.activeNodeCoordinates[l-1]
                                    if sum(inputLayerActiveNodes) > 0:
                                        w.setRandomWeights(layerId=l, inputLayerActiveNodes=inputLayerActiveNodes)
                    else:
                        # --- node became inactive
                        for w in ind.nnLayers[l].units[n].weights:
                            w.set_allzero_value_mask()
        # ------------------------------------
        # --- Final check
        # ------------------------------------
        for l, activeNodesLayer in enumerate(ind.activeNodeCoordinates):
            if sum(activeNodesLayer) == 0:
                # --- Activate some node
                n = SRConfig.r.integers(0, len(activeNodesLayer))
                activeNodesLayer[n] = True
                self.activateNode(ind, l, n)
        # ---
        return changed


    def networkConfigMutation_A(self, ind: SubTopology):
        """
        Forward pass!
        Mutates active nodes weights.
        :param ind:
        :return:
        """
        for l in range(len(ind.activeNodeCoordinates)):
            for n in range(len(ind.activeNodeCoordinates[l])):
                if ind.activeNodeCoordinates[l][n] and (SRConfig.r.random() < SRConfig.nodeConfigMutationRate):
                    # --- choose randomly one config
                    config: InputConfiguration
                    if SRConfig.r.random() > SRConfig.historyUsageProb:
                        config = None
                    elif SRConfig.historyType == 0:
                        config = self.networkHistory[l][n].getRandomConfig()
                    else:
                        if SRConfig.historyConfigPreferences == 0:
                            config = MasterTopology.networkHistoryPerSize[l][n].getConfigRandom()
                        elif SRConfig.historyConfigPreferences == -1:
                            config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=True)
                        else:
                            config = MasterTopology.networkHistoryPerSize[l][n].getConfigRouletteWheel(preferSmall=False)
                    # ---
                    if config:
                        for w, c in zip(ind.nnLayers[l].units[n].weights, config.nodeWeights):
                            w.set_value_mask(c)
                    # --- generate random values
                    else:
                        for w in ind.nnLayers[l].units[n].weights:
                            # --- generate random values
                            inputLayerActiveNodes: list[bool]
                            if ' b' in w.v.name:    # --- bias
                                inputLayerActiveNodes = [True]
                            elif l == 0:
                                inputLayerActiveNodes = ind.nnLayers[l].inSize * [True]
                            else:
                                inputLayerActiveNodes = ind.activeNodeCoordinates[l-1]
                            w.setRandomWeights(layerId=l, inputLayerActiveNodes=inputLayerActiveNodes)


    def chooseLayerProportionallyNodeUniformly(self, ind: SubTopology):
        """
        Chooses randomly one layer proportionally to its depth and
        chooses one of its active nodes randomly uniformly.
        :param ind:
        :return:
        """
        layerId = 0
        x = 1.0 / ((1 + len(self.nnLayers)) * len(self.nnLayers) / 2.0)
        while True:
            r = SRConfig.r.random()
            n = len(self.nnLayers)
            acc = n * x
            while r > acc and n > 0:
                n -= 1
                acc += n * x
                layerId = len(self.nnLayers) - n
            if np.sum(ind.activeNodeCoordinates[layerId]) > 0:
                break
        while True:
            nodeId = SRConfig.r.integers(low=0, high=len(self.nnLayers[layerId].units))
            if ind.activeNodeCoordinates[layerId][nodeId]:
                break
        return layerId, nodeId


    def chooseLayerUniformlyNodeUniformly(self, ind: SubTopology):
        """
        Chooses randomly one layer uniformly to its depth and
        chooses one of its active nodes randomly uniformly.
        :param ind:
        :return:
        """
        while True:
            layerId = SRConfig.r.integers(low=0, high=len(self.nnLayers))
            if np.sum(ind.activeNodeCoordinates[layerId]) > 0:
                break
        while True:
            nodeId = SRConfig.r.integers(low=0, high=len(self.nnLayers[layerId].units))
            if ind.activeNodeCoordinates[layerId][nodeId]:
                break
        return layerId, nodeId


    def networkConfigMutation_B(self, ind: SubTopology):
        """
        Mutates only ACTIVE NODEs weights.
        Firstly, one layer is randomly chosen using a distribution that linearly increases from the output
        layer to the first one. For L layers, a probability of choosing the last layer is x %, while the probability
        of choosing the first layer if L*x %.
        :param ind:
        :return:
        """
        changed: bool = True    # --- TODO: check if the chosen config differs from the node's current weights
        # -------------------------------------------------------------
        # --- Choose layer and one of its nodes to be selected
        # -------------------------------------------------------------
        layerId, nodeId = self.chooseLayerProportionallyNodeUniformly(ind)
        # layerId, nodeId = self.chooseLayerUniformlyNodeUniformly(ind)
        # -------------------------------------------------------------
        # --- Apply mutation to the selected node
        # -------------------------------------------------------------
        # --- Choose randomly one config, if there is one
        config: InputConfiguration
        if SRConfig.r.random() < SRConfig.nodeConfigMutationRate:
            config = None
        elif SRConfig.historyType == 0:
            config = self.networkHistory[layerId][nodeId].getRandomConfig()
        else:
            if SRConfig.historyConfigPreferences == 0:
                config = MasterTopology.networkHistoryPerSize[layerId][nodeId].getConfigRandom()
            elif SRConfig.historyConfigPreferences == -1:
                config = MasterTopology.networkHistoryPerSize[layerId][nodeId].getConfigRouletteWheel(preferSmall=True)
            else:
                config = MasterTopology.networkHistoryPerSize[layerId][nodeId].getConfigRouletteWheel(preferSmall=False)
        # ---
        if config and checkConfigShape(nodeWeights=ind.nnLayers[layerId].units[nodeId].weights, configWeights=config.nodeWeights):
            for w, c in zip(ind.nnLayers[layerId].units[nodeId].weights, config.nodeWeights):
                w.set_value_mask(c)
        # --- Or generate a new random config
        else:
            if 'UnitIdent1' in ind.nnLayers[layerId].units[nodeId].name:
                # --------------------------------------------
                # --- UnitIdent1 node: set its default
                # --------------------------------------------
                ind.nnLayers[layerId].units[nodeId].reset_weights_and_bias()
            else:
                # --------------------------------------------
                # --- Ordinary unit: generate random values
                # --------------------------------------------
                for w in ind.nnLayers[layerId].units[nodeId].weights:
                    inputLayerActiveNodes: list[bool]
                    if ' b' in w.v.name:    # --- bias
                        inputLayerActiveNodes = [True]
                    else:
                        if layerId == 0:
                            inputLayerActiveNodes = [SRConfig.r.random() < 0.5 for _ in range(ind.nnLayers[layerId].inSize)]
                            if sum(inputLayerActiveNodes) == 0:
                                nonzero: int = SRConfig.r.integers(low=0, high=ind.nnLayers[layerId].inSize)
                                inputLayerActiveNodes[nonzero] = True
                        else:
                            inputLayerActiveNodes = ind.activeNodeCoordinates[layerId-1]
                    w.setRandomWeights(layerId=layerId, inputLayerActiveNodes=inputLayerActiveNodes)
        # ---
        return changed


    def activateNode(self, subTopology:SubTopology, layerId: int, nodeId: int):
        """
        Activates a node at the given position and makes sure that its weights and mask are initialized.
        :param layerId:
        :param nodeId:
        :return:
        """
        subTopology.activeNodeCoordinates[layerId][nodeId] = True
        if 'UnitIdent1' in subTopology.nnLayers[layerId].units[nodeId].name:
            subTopology.nnLayers[layerId].units[nodeId].reset_weights_and_bias()
        else:
            config: InputConfiguration
            if SRConfig.r.random() > SRConfig.historyUsageProb:       # --- ORIGINAL: commented; TODO: uncomment
                config = None
            elif SRConfig.historyType == 0:
                config = self.networkHistory[layerId][nodeId].getRandomConfig()
            else:
                if SRConfig.historyConfigPreferences == 0:
                    config = MasterTopology.networkHistoryPerSize[layerId][nodeId].getConfigRandom()
                elif SRConfig.historyConfigPreferences == -1:
                    config = MasterTopology.networkHistoryPerSize[layerId][nodeId].getConfigRouletteWheel(preferSmall=True)
                else:
                    config = MasterTopology.networkHistoryPerSize[layerId][nodeId].getConfigRouletteWheel(preferSmall=False)
            # ---
            if config:
                # --- Already has some config stored in its history
                for w, c in zip(subTopology.nnLayers[layerId].units[nodeId].weights, config.nodeWeights):
                    w.set_value_mask(c)
            else:
                # --- generate random values
                for w in subTopology.nnLayers[layerId].units[nodeId].weights:
                    inputLayerActiveNodes: list[bool]
                    if ' b' in w.v.name:    # --- bias
                        inputLayerActiveNodes = [True]
                    elif layerId == 0:
                        inputLayerActiveNodes = subTopology.nnLayers[0].inSize * [True]
                    else:
                        inputLayerActiveNodes = subTopology.activeNodeCoordinates[layerId-1]
                    print(f'inputLayerActiveNodes: {inputLayerActiveNodes}')
                    w.setRandomWeights(layerId=layerId, inputLayerActiveNodes=inputLayerActiveNodes)


    def useActiveNode(self, subTopology:SubTopology, layerId: int, nodeId: int):
        """
        Link the output of the given active node 'nnLayers[layerId].units[nodeId]' with a randomly chosen active node in the subsequent layer 'layerId+1'.
        :param layerId:
        :param nodeId:
        :return:
        """
        # --- Collect ordinary active nodes in the subsequent layer 'l+1'
        activeOrdinaryOutputNodes = [i for i in range(len(subTopology.activeNodeCoordinates[layerId+1])) if (subTopology.activeNodeCoordinates[layerId+1][i] and ('UnitIdent1' not in subTopology.nnLayers[layerId+1].units[i].name))]
        # --------------------------------------------------------------
        # --- Ordinary function unit is to be used in the layer l+1
        # --------------------------------------------------------------
        if 'UnitIdent1' not in subTopology.nnLayers[layerId].units[nodeId].name:
            # --- Ordinary unit in layer 'l' can lead only to an ordinary unit in layer 'l+1'
            # --- TODO: ordinary unit can lead to unitIdent1 as well
            if activeOrdinaryOutputNodes:
                res: int = SRConfig.r.choice(activeOrdinaryOutputNodes)      # --- choose one unit in the layer 'l+1' that will use the unit nnLayers[layerId].units[nodeId]
                for w in subTopology.nnLayers[layerId+1].units[res].weights:
                    if ' b' not in w.v.name:    # --- weights, NOT bias
                        values = w.v.numpy()
                        # values[nodeId] = SRConfig.r.uniform(low=-SRConfig.weight_sigma, high=SRConfig.weight_sigma, size=1)     # --- ORIGINAL: uniform
                        if SRConfig.r.random() < 0.5:
                            if (layerId == 0) and (SRConfig.r.random() < SRConfig.weight_init_one):
                                values[nodeId] = 1.0     # --- TODO   +1
                            else:
                                values[nodeId] = SRConfig.weight_min_init + abs(SRConfig.r.normal(loc=0.0, scale=SRConfig.weight_sigma, size=1)[0])     # --- TODO: normal with SRConfig.weight_min_init
                        else:
                            if (layerId == 0) and (SRConfig.r.random() < SRConfig.weight_init_one):
                                values[nodeId] = -1.0     # --- TODO   -1
                            else:
                                values[nodeId] = -SRConfig.weight_min_init - abs(SRConfig.r.normal(loc=0.0, scale=SRConfig.weight_sigma, size=1)[0])    # --- TODO: normal with SRConfig.weight_min_init
                        w.set_value_mask(values)
                        break       # --- TODO: only one variable is updated
        # --------------------------------------------------------------------------------------
        # --- UnitIdent1 is the unit to be used as an iput to some unit in the next layer l+1
        # --------------------------------------------------------------------------------------
        else:
            # --- Find a corresponding UnitIdent1 id in the subsequent layer layerId+1
            identId: int = 0
            for i, node in enumerate(subTopology.nnLayers[layerId+1].units):
                node: UnitIdent1
                if ('UnitIdent1' in node.name) and (node.inId == nodeId):
                    identId = i
                    activeOrdinaryOutputNodes.append(i)     # --- this UnitIdent1 node, nnLayers[layerId+1].units[identId], is added to the set of possible consumers of the current node's output
                    break
            # --- Choose randomly one of the output nodes.
            # --- It can be either an ordinary function unit or the particular UnitIdent1.
            if not activeOrdinaryOutputNodes:
                return
            res: int = SRConfig.r.choice(activeOrdinaryOutputNodes)
            if res == identId:
                # --- Output node is UnitIdent1
                subTopology.nnLayers[layerId+1].units[res].reset_weights_and_bias()
                subTopology.activeNodeCoordinates[layerId+1][res] = True
            else:
                # --- Output node is an ordinary unit
                for w in subTopology.nnLayers[layerId+1].units[res].weights:
                    if ' b' not in w.v.name:    # --- weights, NOT bias
                        values = w.v.numpy()
                        if SRConfig.r.random() < 0.5:
                            # if ((layerId == 0) or ('x[' in subTopology.nnLayers[layerId].units[nodeId].proxyName)) and (SRConfig.r.random() < SRConfig.weight_init_one):
                            if (layerId == 0) and (SRConfig.r.random() < SRConfig.weight_init_one):
                                values[nodeId] = 1.0     # --- TODO   +1
                            else:
                                values[nodeId] = SRConfig.weight_min_init + abs(SRConfig.r.normal(loc=0.0, scale=SRConfig.weight_sigma, size=1)[0])     # --- TODO: normal with SRConfig.weight_min_init
                        else:
                            # if ((layerId == 0) or ('x[' in subTopology.nnLayers[layerId].units[nodeId].proxyName)) and (SRConfig.r.random() < SRConfig.weight_init_one):
                            if (layerId == 0) and (SRConfig.r.random() < SRConfig.weight_init_one):
                                values[nodeId] = -1.0     # --- TODO   -1
                            else:
                                values[nodeId] = -SRConfig.weight_min_init - abs(SRConfig.r.normal(loc=0.0, scale=SRConfig.weight_sigma, size=1)[0])    # --- TODO: normal with SRConfig.weight_min_init
                        w.set_value_mask(values)


    def rectifyActiveNodesConnectivity(self, subTopology:SubTopology):
        """
        Forward pass only.
        For each layer, from the output one forward, apply the following actions to all its active nodes:
           1) Make sure that the node weights has some weight config chosen.
                    If the active node is the UnitIdent1 then set its mask to mask_orig and set its weights accordingly.
           2) Make sure that all its active weights point to active nodes. If some active weight links to an inactive node then
                  - the weight is inactivated,
                  - the mask is adjusted accordingly.
               If the weight pointing to an inactive node is the last active one then
                  - another inactive weight pointing to an active node is activated,
                  - the mask is adjusted accordingly.
           3) Make sure that the node's output is passed to at least one node in the subsequent layer.
              The output node is chosen only among the set of active nodes of the subsequent layer.
              If not applicable, do nothing.
        :return:
        """
        if SRConfig.r.random() > SRConfig.rectify:     # --- do not perform rectification
            return
        # ---
        for l in range(len(subTopology.nnLayers)):     # --- l ... current layer
            # ----------------------------------------
            # --- All layers, but the first one.
            # ----------------------------------------
            if l > 0:
                # --- input-nodes activity accumulation
                for n, currentNode in enumerate(subTopology.nnLayers[l].units):
                    for tfvar in currentNode.weights:
                        if ' b' not in tfvar.v.name:    # --- weights, NOT bias
                            if ('UnitIdent1' in currentNode.name):
                                # ----------------------------------------------------------------------------
                                # --- 1) If the active node is the UnitIdent1
                                # ---    then set its mask to mask_orig and set its weights accordingly.
                                # ----------------------------------------------------------------------------
                                currentNode: UnitIdent1
                                if not subTopology.activeNodeCoordinates[l-1][currentNode.inId]:
                                    # --- Unsatisfiable: deactivate UnitIdent1
                                    subTopology.activeNodeCoordinates[l][n] = False
                                    tfvar.set_allzero_value_mask()
                            else:
                                # ----------------------------------------------------------------------------
                                # --- 2) Make sure that all its active weights point to active nodes.
                                # ----------------------------------------------------------------------------
                                unsatisfiedActivity = [req and (not act) for req, act in zip (tfvar.mask.flatten(), subTopology.activeNodeCoordinates[l-1])]
                                if sum(unsatisfiedActivity) > 0:
                                    weights = tfvar.v.numpy()
                                    for i, u in enumerate(unsatisfiedActivity):
                                        if u:
                                            weights[i] = 0
                                    tfvar.set_value_mask(weights)
                                    # ---
                                    if sum(weights) == 0:
                                        origMask = np.array(subTopology.activeNodeCoordinates[l-1]).reshape((len(subTopology.activeNodeCoordinates[l-1]), 1))
                                        origMaskNonZero = origMask.nonzero()
                                        if len(origMaskNonZero[0]) > 0:
                                            toActivate = SRConfig.r.choice(origMaskNonZero[0], 1)[0]
                                            # weights[toActivate] = SRConfig.r.uniform(low=-SRConfig.weight_sigma, high=SRConfig.weight_sigma, size=1)    # --- ORIGINAL: uniform
                                            if SRConfig.r.random() < 0.5:
                                                weights[toActivate] = SRConfig.weight_min_init + abs(SRConfig.r.normal(loc=0.0, scale=SRConfig.weight_sigma, size=1)[0])  # --- TODO: normal with weight_min_init
                                            else:
                                                weights[toActivate] = -SRConfig.weight_min_init - abs(SRConfig.r.normal(loc=0.0, scale=SRConfig.weight_sigma, size=1)[0])  # --- TODO: normal with weight_min_init
                                            tfvar.set_value_mask(weights)
                                        else:
                                            # --- Unsatisfiable: deactivate node[l][n]
                                            subTopology.activeNodeCoordinates[l][n] = False
            # ----------------------------------------
            # --- All layers, but the last one.
            # ----------------------------------------
            if l < len(subTopology.nnLayers)-1:
                # --- 3) Make sure that the node's output is passed to at least one node in the subsequent layer.
                outputNodesInputActivity = np.zeros(shape=[len(subTopology.nnLayers[l].units),1])   # --- aggregated input activity over all active nodes in the next layer 'l+1'
                # --- input-nodes activity aggregation
                for outputNode in subTopology.nnLayers[l+1].units:
                    for tfvar in outputNode.weights:
                        if ' b' not in tfvar.v.name:    # --- weights, NOT bias
                            outputNodesInputActivity = np.logical_or(outputNodesInputActivity, tfvar.mask)
                # --- output layer activity check
                for i, u in enumerate(subTopology.activeNodeCoordinates[l]):
                    if u and not outputNodesInputActivity[i]:   # --- the active node 'nnLayers[l].units[i]' IS NOT used by any active node in the subsequent layer 'l+1'
                        self.useActiveNode(subTopology, l, i)


    def initializeNodeHistory(self, layerId:int, nodeId:int, subTopology:SubTopology):
        """
        :param layerId:
        :param nodeId:
        :param subTopology:
        :return:
        """
        weights = copy.deepcopy(subTopology.nnLayers[layerId].units[nodeId].weights)
        newConfig = InputConfiguration(nodeVars=weights, losses=copy.deepcopy(subTopology.losses), masterTopologyId=subTopology.masterTopologyId)
        self.networkHistory[layerId][nodeId].configs.append(newConfig)
        if len(self.networkHistory[layerId][nodeId].configs) > SRConfig.historySize:
            self.networkHistory[layerId][nodeId].configs.pop(0)


    def initializeNetworkHistory(self):
        # --- For each node in each layer
        self.networkHistory = []
        MasterTopology.networkHistoryPerSize = []
        MasterTopology.lastSubTopologiesAddedToHistory = [1000] * SRConfig.historySize
        for i, layer in enumerate(self.nnLayers):
            newHistoryLayer = list(NodeHistory([]) for _ in layer.units)
            self.networkHistory.append(newHistoryLayer)
            newHistoryLayer = list(NodeHistoryPerSize({}) for _ in layer.units)
            MasterTopology.networkHistoryPerSize.append(newHistoryLayer)


    def removeConfigs(self, layerId:int, nodeId:int, toRemove: list[int]):
        for i in range(len(toRemove)-1, -1, -1):
            self.networkHistory[layerId][nodeId].configs.pop(toRemove[i])


    def isDominatedByHistoryConfig(self, layerId:int, nodeId:int, subTopology:SubTopology, byTopologyId:int=-1):
        """
        Checks whether the active node at position [layerId, nodeId] is dominated by some configuration stored
        in the corresponding node's history list.
        Comparisons are made only to the configs with the same masterTopologyId as the one of the 'subTopology'.
        :param layerId:
        :param nodeId:
        :param subTopology:
        :return: dominated, dominates
                 dominated ... True, if the subTopology is dominated. False, otherwise.
                 dominates ... a list of dominated config ids
        """
        dominates: list[int] = []
        for i, c in enumerate(self.networkHistory[layerId][nodeId].configs):
            if (byTopologyId > -1) and (c.masterTopologyId != subTopology.masterTopologyId):      # --- do not compare irrelevant config
                continue
            isWorse = False
            isBetter = False
            for key in SubTopology.keysAll:
                if subTopology.losses[key] < c.losses[key]:
                    isBetter = True    # --- the subTopology is strictly better in some loss term, so it cannot be dominated
                if subTopology.losses[key] > c.losses[key]:
                    isWorse = True     # --- the subTopology is strictly worse in some loss term, so it might be dominated
            if (not isBetter) and isWorse:
                return True, None       # --- subTopology is dominated
            if (not isWorse):
                dominates.append(i)     # --- i-th config is dominated by the subTopology
        return False, dominates


    def updateNodeHistory(self, layerId:int, nodeId:int, subTopology:SubTopology, partitionSize: int, byTopologyId:int=-1):
        """
        Updates history of the unit [layerId, nodeId] according to the corresponding active node of the 'subTopology'.
        :param layerId:
        :param nodeId:
        :param subTopology:
        :return:
        """
        # --------------------------------------------------------------------------------------
        # --- Check if the new config is dominated by or dominates configs assigned
        # --- to the given node.
        # --- Consider masterTopologyId.
        # --------------------------------------------------------------------------------------
        isDominated, dominates = self.isDominatedByHistoryConfig(layerId, nodeId, subTopology, byTopologyId=byTopologyId)
        if not isDominated:
            # --------------------------------------------------------------------------------------
            # --- Use the new config only iff it is not dominated by any other config of the node
            # --------------------------------------------------------------------------------------
            if dominates:
                # --------------------------------------------------------------
                # --- Remove all configs that are dominated by the new config
                # --------------------------------------------------------------
                self.removeConfigs(layerId, nodeId, dominates)
            nodeVars = copy.deepcopy(subTopology.nnLayers[layerId].units[nodeId].weights)
            newConfig = InputConfiguration(nodeVars=nodeVars, losses=subTopology.losses, masterTopologyId=subTopology.masterTopologyId)
            # -----------------------------------------------
            # --- Find the best position for the newConfig
            # -----------------------------------------------
            # position = self.chooseBestPositionForConfig(newConfig, layerId, nodeId)
            # if position == -1:
            #     # --- extend the history
            #     self.networkHistory[layerId][nodeId].configs.append(newConfig)
            # elif position >= 0:
            #     self.networkHistory[layerId][nodeId].configs[position] = newConfig
            # -----------------------------------------------
            # FIFO w.r.t. masterTopologyId
            # -----------------------------------------------
            # --- Find config to be removed
            relevantConfigs: list = []     # --- list of positions of configs with the right masterTopologyId
            for i, c in enumerate(self.networkHistory[layerId][nodeId].configs):
                if c.masterTopologyId == newConfig.masterTopologyId:
                    relevantConfigs.append(i)
            # ---
            self.networkHistory[layerId][nodeId].configs.append(newConfig)
            # --- Update size of the History
            if len(relevantConfigs) == partitionSize:
                # --- Remove the oldest relevant config
                self.networkHistory[layerId][nodeId].configs.pop(relevantConfigs[0])
            elif len(self.networkHistory[layerId][nodeId].configs) > SRConfig.historySize:
                # --- Remove the oldest config
                self.networkHistory[layerId][nodeId].configs.pop(0)
            # ---
            return True     # --- new config was added to the history
        else:
            return False    # --- new config was not added


    def updateNetworkHistory(self, subTopology:SubTopology, partitionSize: int, byTopologyId:int=-1):
        """
        Updates history records of each unit in the master topology according to active nodes of the subTopology.
        :param subTopology:
        :return:
        """
        # --- Clear history
        self.filterOutAllConfigsDominatedBySubTopology(subTopology, byTopologyId=byTopologyId)
        # --- For each active node in each layer
        subTopologyWasUsed = False
        for l, layer in enumerate(self.nnLayers):
            for n in range(len(layer.units)):
                if subTopology.activeNodeCoordinates[l][n]:
                    currAdded = self.updateNodeHistory(l, n, subTopology, partitionSize=partitionSize, byTopologyId=byTopologyId)
                    subTopologyWasUsed = subTopologyWasUsed and currAdded


    def filterOutAllConfigsDominatedBySubTopology(self, subTopology:SubTopology, byTopologyId:int=-1):
        """
        Removes all configs dominated by 'subTopology' from the network history.
        :param subTopology:
        """
        # --------------------------------------------------------------------------------------
        # --- Check if the new config is dominated by or dominates configs assigned
        # --- to the given node.
        # --- Consider masterTopologyId.
        # --------------------------------------------------------------------------------------
        for l, layer in enumerate(self.nnLayers):
            for n in range(len(layer.units)):
                isDominated, dominates = self.isDominatedByHistoryConfig(l, n, subTopology, byTopologyId=byTopologyId)
                if dominates:
                    # --------------------------------------------------------------
                    # --- Remove all configs that are dominated by the new config
                    # --------------------------------------------------------------
                    self.removeConfigs(l, n, dominates)


    def isDominatedByHistoryPerSizeConfig(self, layerId:int, nodeId:int, subTopology:SubTopology):
        """
        Checks whether the active node at position [layerId, nodeId] is dominated by some configuration stored
        in the corresponding node's history list.
        :param layerId:
        :param nodeId:
        :param subTopology:
        :return: dominated, dominates
                 dominated == True, if the subTopology is dominated.
                 dominates ... a list of dominated config tuples (nbOfActiveNodes, id)
        """
        dominates: list[tuple[int, int]] = []
        if not MasterTopology.networkHistoryPerSize[layerId][nodeId].configs:
            return False, dominates    # --- configs is empty, it cannot dominate anything
        # ---
        complexities: list[int] = list(self.networkHistory[layerId][nodeId].configs.keys())    # --- all keys in configs
        for complexity in complexities:
            for i, c in enumerate(self.networkHistory[layerId][nodeId].configs[complexity]):
                isWorse = False
                isBetter = False
                for key in SubTopology.keysAll:
                    if subTopology.losses[key] < c.losses[key]:
                        isBetter = True    # --- the subTopology is strictly better in some loss term, so it cannot be dominated
                    if subTopology.losses[key] > c.losses[key]:
                        isWorse = True     # --- the subTopology is strictly worse in some loss term, so it might be dominated
                if (not isBetter) and isWorse:
                    return True, None       # --- subTopology is dominated
                if (not isWorse):
                    dominatedConfig: tuple[int, int] = (complexity, i)   # --- i-th config is dominated by the subTopology
                    dominates.append(dominatedConfig)
        return False, dominates


    def updateNodeHistoryPerSize(self, layerId:int, nodeId:int, subTopology:SubTopology):
        """
        Updates history of the particular unit [layerId, nodeId] according to the corresponding active node of the subTopology.
        :param layerId:
        :param nodeId:
        :param subTopology:
        :return:
        """
        # isDominated, dominates = self.isDominatedByHistoryPerSizeConfig(layerId, nodeId, subTopology)
        # if not isDominated:
        #     if dominates:
        #         for el in dominates:
        #             el: tuple[int, int]
        #             self.networkHistoryPerSize[layerId][nodeId].removeConfig(el[0], el[1])
        # ---
        complexity = subTopology.nbOfActiveNodes
        nodeVars = copy.deepcopy(subTopology.nnLayers[layerId].units[nodeId].weights)
        newConfig = InputConfiguration(nodeVars=nodeVars, losses=subTopology.losses, masterTopologyId=subTopology.masterTopologyId)
        # -----------------------------------------------
        # FIFO
        # -----------------------------------------------
        MasterTopology.networkHistoryPerSize[layerId][nodeId].addConfig(complexity, newConfig)


    def checkNetworkHistoryPerSize(self, subTopology:SubTopology):
        """
        Checks if the new subTopology can be added to the network history.
        First, it checks if the new subTopology is dominated by any other subTopology stored in the network history.
        If yes, then it will not be added to the history.
        If not, then it find all dominated configs and removes them from the network history.
        :param subTopology:
        :return:
        """
        allDominatedConfigs: list[tuple[int, int, int, list[int]]] = []  # --- list[tuple(layerId, nodeId, complexity, list[configIds])]
        maxSize = np.max(self.lastSubTopologiesAddedToHistory)
        for l, historyLayer in enumerate(MasterTopology.networkHistoryPerSize):
            for n, historyNode in enumerate(historyLayer):
                historyNode: NodeHistoryPerSize
                complexities = list(historyNode.configs.keys())
                for complexity in complexities:
                    newDominatedConfigs = []
                    for i, config in enumerate(historyNode.configs[complexity]):
                        # --- config comes from too large subTopology
                        if SRConfig.historyCheckConfigSize and complexity > maxSize:
                            newDominatedConfigs.append(i)
                            continue
                        # --- check the config's dominance
                        isWorse = False
                        isBetter = False
                        for k in SubTopology.keysAll:
                            if subTopology.losses[k] < config.losses[k]:
                                isBetter = True  # --- newConfig is strictly better in some loss term, so it cannot be dominated
                            if subTopology.losses[k] > config.losses[k]:
                                isWorse = True  # --- newConfig is strictly worse in some loss term, so it might be dominated
                        if (not isBetter) and isWorse:
                            return True, None  # --- newConfig is dominated hence it cannot be added to the network history
                        if (not isWorse):
                            newDominatedConfigs.append(i)
                    # --- this complexity dominated configs
                    if newDominatedConfigs:
                        newDominatedConfigs.sort(reverse=True)
                        allDominatedConfigs.append((l, n, complexity, newDominatedConfigs))
        return False, allDominatedConfigs


    def purgeNetworkHistoryPerSize(self, dominatedConfigs: list[tuple[int, int, int, list[int]]]):  # --- list[tuple(layerId, nodeId, complexity, list[configIds])]
        """
        Removes dominated configs from the network history.
        :param dominatedConfigs:
        :return:
        """
        for el in dominatedConfigs:
            for configId in el[3]:
                MasterTopology.networkHistoryPerSize[el[0]][el[1]].removeConfig(complexity=el[2], configId=configId)


    def updateNetworkHistoryPerSize(self, subTopology:SubTopology):
        """
        Updates history records of each unit in the master topology according to active nodes of the subTopology.
        :param subTopology:
        :return:
        """
        # -----------------------------------------------------------------------------
        # --- Check whether the subTopology can be used for history update
        # -----------------------------------------------------------------------------
        if SRConfig.historyCheckConfigSize and (subTopology.nbOfActiveNodes > np.max(self.lastSubTopologiesAddedToHistory)):
            return
        # ---
        stIsDominated, stDominates = self.checkNetworkHistoryPerSize(subTopology)
        if stIsDominated:
            return
        # ---
        if stDominates:
            self.purgeNetworkHistoryPerSize(stDominates)
        # -----------------------------------------------------------------------------
        # --- Update history
        # -----------------------------------------------------------------------------
        if subTopology.nbOfActiveNodes not in self.lastSubTopologiesAddedToHistory:
            self.lastSubTopologiesAddedToHistory.append(subTopology.nbOfActiveNodes)
            self.lastSubTopologiesAddedToHistory.sort()
            self.lastSubTopologiesAddedToHistory.pop(-1)
        # ---
        for l, layer in enumerate(self.nnLayers):
            for n in range(len(layer.units)):
                if subTopology.activeNodeCoordinates[l][n]:
                    self.updateNodeHistoryPerSize(l, n, subTopology)


    def printNetworkHistory(self):
        print(f'\nHistory')
        for l, layer in enumerate(self.networkHistory):
            if l < len(self.networkHistory)-1:  # --- print only the last layer
                continue
            print(f'{l}. layer: ')
            for n, node in enumerate(layer):
                node: NodeHistory
                print(f'{n}. node:')
                for c in node.configs:
                    print(f"\t\tcomplexity={c.losses['complexity']}: rmse_valid={c.losses['rmse_valid']:2.5f}, rmse_constr={c.losses['rmse_constr']:2.5f}, topologyId={c.masterTopologyId}")


    def printNetworkHistoryPerSize(self):
        print('\n--------------------------------')
        print('--- Network history per size:')
        for l, layer in enumerate(MasterTopology.networkHistoryPerSize):
            print(f'{l}. layer: ')
            for n, node in enumerate(layer):
                node: NodeHistoryPerSize
                print(f'\t{n}. node:')
                for complexity in node.configs.keys():
                    for c in node.configs[complexity]:
                        print(f"\t\tnodes={complexity}: rmse_valid={c.losses['rmse_valid']:2.5f}, rmse_constr={c.losses['rmse_constr']:2.5f}")


