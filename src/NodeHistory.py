import numpy as np
from InputConfiguration import InputConfiguration
from SRUnits import MyTFVariable
import SRConfig


class NodeHistory:

    def __init__(self, configs=None):
        self.configs = (
            configs  # --- # ---  a set of recent best input configurations of the node
        )

    def getBestPerforming(self):
        """
        Chooses the best configuration.
        :return:
        """
        pass

    def getRandomConfig(self):
        """
        Chooses randomly one of the node's configurations.
        :return:
        """
        if self.configs:
            return SRConfig.r.choice(self.configs)
        else:
            return None


