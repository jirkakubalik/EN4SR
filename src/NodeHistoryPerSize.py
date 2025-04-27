import numpy as np
from InputConfiguration import InputConfiguration
from SRUnits import MyTFVariable
from SubTopology import SubTopology
import SRConfig


class NodeHistoryPerSize():

    def __init__(self, configs: dict[int, list[InputConfiguration]]=None):
        self.configs: dict[int, list[InputConfiguration]] = configs     # --- dict[nbOfActiveNodes, list[extreme InputConfigurations]]


    def getBestPerforming(self):
        """
        Chooses the best configuration.
        :return:
        """
        pass


    def getConfigRandom(self):
        """
        Chooses randomly one of the node's configurations.
        :return:
        """
        if not self.configs:
            return None
        # --- 1. Choose the subTopology complexity
        complexities = list(self.configs.keys())
        cplx = SRConfig.r.choice(complexities)
        # --- 2. Choose one config
        currConfigs = self.configs[cplx]
        selectedConfig = SRConfig.r.choice(currConfigs)
        return selectedConfig


    def getConfigRouletteWheel(self, preferSmall: bool = True):
        """
        Chooses one node's configuration according to the Roulette Wheel.
        :return:
        """
        if not self.configs:
            return None
        # --------------------------------------------------
        # --- Small configs are preferred
        # --------------------------------------------------
        if preferSmall:
            maxCplx = np.max(list(self.configs.keys()))
            wheel = np.sum([maxCplx-k+1 for k in self.configs.keys()])
            p = wheel * SRConfig.r.random()
            sum = 0.0
            for k in self.configs.keys():
                sum += maxCplx-k+1
                if sum >= p:
                    currConfigs = self.configs[k]
                    selectedConfig = SRConfig.r.choice(currConfigs)
                    return selectedConfig
        # --------------------------------------------------
        # --- Large configs are preferred
        # --------------------------------------------------
        else:
            minCplx = np.min(list(self.configs.keys()))
            wheel = np.sum([k-minCplx+1 for k in self.configs.keys()])
            p = wheel * SRConfig.r.random()
            sum = 0.0
            for k in self.configs.keys():
                sum += k-minCplx+1
                if sum >= p:
                    currConfigs = self.configs[k]
                    selectedConfig = SRConfig.r.choice(currConfigs)
                    return selectedConfig


    def addConfig(self, newComplexity: int, newConfig: InputConfiguration):
        # ----------------------------------------------------------------------------
        # --- 2. Add newConfig into the node's configs
        # ----------------------------------------------------------------------------
        complexities = list(self.configs.keys())
        if newComplexity in complexities:
            # --- There exist some configs of the given complexity
            if len(self.configs[newComplexity]) < len(SubTopology.keysPerformance):
                # --- There are not all extreme configs in the node's history for this complexity
                self.configs[newComplexity].append(newConfig)
            else:
                # --- a) Find the one to be replaced
                replace: int = -1   # --- default is not to replace anyone
                for i, config in enumerate(self.configs[newComplexity]):
                    for l in SubTopology.keysPerformance:
                        if newConfig.losses[l] < config.losses[l]:
                            replace = i
                            break
                # --- b) Replace an existing config, if applicable
                if replace > -1:
                    self.configs[newComplexity][replace] = newConfig
        else:
            # --- There are no configs of the given complexity
            if len(complexities) < SRConfig.historySize:
                # --- a) The history is not full to its capacity
                self.configs[newComplexity] = [newConfig]
            else:
                # --- b) The history is full, replace the largest complexity list
                maxComplexity = np.max(complexities)
                if newComplexity < maxComplexity:
                    self.configs.pop(maxComplexity)
                    self.configs[newComplexity] = [newConfig]


    def removeConfig(self, complexity: int, configId: int):
        """
        Remove a particular config from node's configurations.
        :return:
        """
        if self.configs[complexity]:
            self.configs[complexity].pop(configId)
            if not self.configs[complexity]:
                self.configs.pop(complexity)    # --- it was the last config of this complexity,
                                                # --- remove given complexity from self.configs
