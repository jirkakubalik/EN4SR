import warnings

import SRData

warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import copy

import SRConfig
from NodeHistory import NodeHistory
from InputConfiguration import InputConfiguration
from SRUnits import NNUnit, UnitMultiply, UnitDivide, \
    UnitSin, UnitCos, UnitTanh, UnitArcTan, \
    UnitSquare, UnitSqrt, UnitCube, \
    UnitIdent, UnitIdent1


class NNLayer():

    def __init__(self, inSize, id):
        self.inSize: int = inSize    #--- nb. of inputs coming from previous layer
        self.id = id
        self.units: list[NNUnit] = []    #--- list of node objects
        self.learnableParams = []


    def add_unit(self, unit):
        learn_params, nolearn_params = unit.define_weight_and_bias_vars(layerId=-1, inSize=self.inSize)
        if learn_params is not None:
            self.learnableParams.extend(learn_params)
        self.units.append(unit)


    def clone_layer(self, layerId:int=None, activeNodes:list[list[bool]]=None):
        """
        Clones the layerId-th layer.
        For active nodes:
           - sets the input weights to some values.
           - adds its parameters to newLayer.learnableParams
        For inactive nodes:
           - sets the input weights to zero.
        :param layerId:
        :param activeNodes:
        :param networkHistory:
        :return:
        """
        newLayer = NNLayer(self.inSize, self.id)
        inputLayerActiveNodes: np.ndarray
        if layerId == 0:
            inputLayerActiveNodes = np.ones(shape=[self.inSize, 1])
        else:
            l = activeNodes[layerId-1]
            inputLayerActiveNodes = np.array(l).reshape(len(l), 1)
        # --- Units
        for i, u in enumerate(self.units):
            newUnit = copy.deepcopy(u)
            newUnit.a = None
            learn_params, nolearn_params = newUnit.define_weight_and_bias_vars(layerId=layerId, inSize=newLayer.inSize)
            # --- Set input weights and mask of an inactive node to zero
            for w_new, w_old in zip(newUnit.weights, u.weights):
                w_new.set_value_mask(w_old.v.numpy())
            if learn_params:
                newLayer.learnableParams.extend(learn_params)
            newLayer.units.append(newUnit)
        # ---
        return newLayer


    def randomly_generate_layer(self, layerId:int=None, layerActiveNodes:list[bool]=None, inputLayer=None):
        """
        Randomly generates the layerId-th layer given the active nodes.
           - Chooses weight configs for the given active nodes.
        :param layerId:
        :param layerActiveNodes: list[bool] ... active nodes of this layer
        :param inputLayer: previous MasterTopology layer of this layer
        :return: newly created newLayer, inputLayerActiveNodes: list[bool]
        """
        # ------------------------------------------------
        # --- Choose weight configs for active nodes
        # ------------------------------------------------
        actualInputLayerActiveNodes: np.array = np.zeros(shape=(self.inSize, 1))
        inputLayerActiveNodes: list[bool] = self.inSize * [True]
        if sum(inputLayerActiveNodes) == 0:
            inputLayerActiveNodes[SRConfig.r.integers(self.inSize)] = True
        inputLayerActivity = np.array(inputLayerActiveNodes).reshape(len(inputLayerActiveNodes), 1)
        newLayer = NNLayer(self.inSize, self.id)
        for i, u in enumerate(self.units):
            newUnit = copy.deepcopy(u)
            learn_params, _ = newUnit.define_weight_and_bias_vars(layerId=layerId, inSize=newLayer.inSize)
            if layerActiveNodes[i]:
                # --------------------------------------------------
                # --- Process an active ordinary function newUnit
                # --------------------------------------------------
                if ('UnitIdent1' not in newUnit.name) and ('UnitMultiplySimplified' not in newUnit.name):
                    for w in newUnit.weights:
                        if ' W' in w.v.name:
                            finalValues = w.v.numpy()
                            # ----------------------------
                            # --- Choose active weights
                            # ----------------------------
                            nonzero = SRConfig.r.choice(inputLayerActivity.nonzero()[0], 1)[0]
                            if layerId > 0:
                                # --- TODO weights to UnitIdent1 representing variables are set to +/-1 with p==weight_init_one
                                finalMask = np.array([int(SRConfig.r.random() <= SRConfig.initActiveUnitsProp) if el else 0 for el in inputLayerActiveNodes]).reshape(inputLayerActivity.shape)
                                # -----------------------------------------------------------------------
                                # --- Adjust weight leading to UnitIdent1 nodes representing variables
                                # -----------------------------------------------------------------------
                                for m in range(len(finalMask)):
                                    inputUnit: NNUnit = inputLayer.units[m]
                                    if (finalMask[m] == 1) and (inputUnit.isVariable) and (SRConfig.r.random() <= SRConfig.weight_init_one):
                                        if SRConfig.r.random() < 0.5:
                                            newValue = 1.0   # --- TODO   +1
                                        else:
                                            newValue = -1.0  # --- TODO   -1
                                        # print(f'layerId={layerId}, unitId={i}, unitName={newUnit.name}, weightId={m}: {finalValues[m]} --> {newValue}')
                                        finalValues[m] = newValue
                            else:
                                # -----------------------------------------------------------------------
                                # --- layerId==0 function nodes choose their input variables with p==0.5
                                # -----------------------------------------------------------------------
                                finalMask = np.array([int(SRConfig.r.random() < 0.5) if el else 0 for el in inputLayerActiveNodes]).reshape(inputLayerActivity.shape)
                            if sum(finalMask) == 0:
                                finalMask[nonzero] = 1
                            finalMask = w.adjustArity(finalMask)
                            actualInputLayerActiveNodes = np.add(actualInputLayerActiveNodes, finalMask)
                            # -----------------------------------
                            # --- Set inactive weights to zero
                            # -----------------------------------
                            finalValues = np.multiply(finalValues, finalMask)
                        else:
                            finalValues = w.v.numpy()
                        # ---------------------------
                        # --- Set finalValues to w
                        # ---------------------------
                        w.set_value_mask(finalValues)
                # -------------------------------------------------
                # --- Process an active UnitIdent1 in layerId==0
                # -------------------------------------------------
                elif ('UnitIdent1' in newUnit.name) and (layerId == 0):
                    newUnit: UnitIdent1
                    if newUnit.inId < SRData.SRData.x_data.shape[1]:
                        newUnit.isVariable = True
                # ---
                if learn_params:
                    newLayer.learnableParams.extend(learn_params)
            else:
                # -------------------------------------------------------------
                # --- Inactive node
                # --- Set input weights and mask of an inactive node to zero.
                # -------------------------------------------------------------
                for w in newUnit.weights:
                    w.set_allzero_value_mask()
            newLayer.units.append(newUnit)
        # ---
        inputLayerActiveNodes = [el[0] > 0 for el in actualInputLayerActiveNodes]
        return newLayer, inputLayerActiveNodes


    def forward_pass(self, x_data):
        a = []
        z = []
        for u in self.units:
            a_i, z_i = u.forward_pass(x_data)
            a.append(a_i)
            if z_i is not None:
                z.append(z_i)
        a = tf.concat(a, 1)
        return a, z
