import numpy as np
import copy

import SRConfig
from SRUnits import MyTFVariable

class InputConfiguration():

    def __init__(self, nodeVars:list[MyTFVariable]=None, losses=None, masterTopologyId:int=0):
        self.varNames = [tfvar.v.name for tfvar in nodeVars]
        self.nodeMasks: list[np.ndarray] = []
        self.nodeWeights: list[np.ndarray] = []  # --- A particular set of node’s weights.
                                                # --- Only tunable weights are non-zero.
        self.masterTopologyId: int = masterTopologyId          # --- masterTopologyId of the source subTopology the config is derived from
        weights = [tfvar.v.numpy() for tfvar in nodeVars]
        for i in range(len(weights)):
            condlist = [weights[i] < -SRConfig.activeWeightThreshold, weights[i] > SRConfig.activeWeightThreshold]
            choicelist = [weights[i], weights[i]]
            values = np.select(condlist, choicelist, 0)
            mask = values != 0
            self.nodeWeights.append(values)
            self.nodeMasks.append(mask)
        self.losses: dict[str,float] = copy.deepcopy(losses)  # --- loss values of the model that the configuration comes from

    def __str__(self):
        res = ''
        for key in self.losses.keys():
            res += f'\t{key}={self.losses[key]:2.4f}'
        # res += '\n'
        for w, weights in enumerate(self.nodeWeights):
            res += f'\n\t {self.varNames[w]}:'
            for el in weights:
                res += f' {el}'
        return res

    # def setData(self, nodeWeights=None, losses=None):
    #     if nodeWeights:
    #         self.nodeWeights = nodeWeights
    #     if losses:
    #         self.losses = losses

    # def getData(self):
    #     return self.nodeWeights, self.losses



