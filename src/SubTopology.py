import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
import statistics
import copy
import pickle

# from bokeh.core.property.validation import validate

import SRConfig
from SRData import SRData
from SRConstraints import SRConstraints
from NNLayer import NNLayer
from SRUnits import MyTFVariable, NNUnit, UnitIdent1
from InputConfiguration import InputConfiguration


class bsfSubTopology:
    def __init__(self, iter, losses, weights):
        self.iter = iter
        self.losses = losses
        self.weights = weights


class L05Regularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, strength, a):
        self.strength = strength
        self.a = a

    def __call__(self, x):
        x = tf.concat(x, 0)
        # --- |x| >= a
        a = tf.constant(self.a * np.ones(x.shape))
        x1 = x[tf.greater_equal(tf.abs(x), a)]
        r1 = tf.reduce_sum(tf.sqrt(tf.abs(x1)))
        # --- |x| < a
        x2 = x[tf.less(tf.abs(x), a)]
        c3 = tf.constant(3.0 * np.ones(x2.shape))
        c4 = tf.constant(4.0 * np.ones(x2.shape))
        c8 = tf.constant(8.0 * np.ones(x2.shape))
        a1 = tf.constant(self.a * np.ones(x2.shape))
        a3 = tf.constant(tf.pow(self.a * np.ones(x2.shape), c3))
        t1 = -tf.divide(tf.pow(x2, c4), tf.multiply(c8, a3))
        t2 = tf.divide(tf.multiply(c3, tf.square(x2)), tf.multiply(c4, a1))
        t3 = tf.divide(tf.multiply(c3, a1), c8)
        r2 = tf.reduce_sum(tf.sqrt(tf.add_n([t1, t2, t3])))
        # ---
        s = tf.constant(self.strength * np.ones(1))
        res = tf.reduce_sum(tf.multiply(s, tf.add(r1, r2)))
        return res


def getMutualDominance(
    first: dict[str, float], second: dict[str, float], criteria: list[str]
):
    """
    Check mutual dominance of the two subTopologies w.r.t. the given criteria set.
    :return:
    """
    firstIsBetter = False
    secondIsBetter = False
    firstDominates = False
    secondDominates = False
    firstEqualsSecond = False
    if len(criteria) == 1:
        if first[criteria[0]] < second[criteria[0]]:
            firstDominates = True
        elif first[criteria[0]] > second[criteria[0]]:
            secondDominates = True
        else:
            firstEqualsSecond = True
    else:
        for key in criteria:
            if first[key] < second[key]:
                firstIsBetter = True
            if first[key] > second[key]:
                secondIsBetter = True
        if firstIsBetter and (not secondIsBetter):
            firstDominates = True
        elif secondIsBetter and (not firstIsBetter):
            secondDominates = True
        elif (not firstIsBetter) and (not secondIsBetter):
            firstDominates = True
            firstEqualsSecond = True
    # ---
    return firstDominates, secondDominates, firstEqualsSecond


class SubTopology:
    keysAll: list[str] = ["rmse_valid", "rmse_constr", "complexity"]
    keysPerformance: list[str] = ["rmse_valid", "rmse_constr"]
    keysPerfNodes: list[str] = ["rmse_valid", "rmse_constr", "nbOfActiveNodes"]
    backpropCount: int = 0

    def __init__(self):
        self.masterTopologyId: int = -1
        self.nnLayers: list[NNLayer] = []
        self.activeNodeCoordinates: list[list[bool]] = (
            []
        )  # --- list of node coordinates [layer, row] that will undergo the weights update
        self.nbOfActiveNodes: float = 0
        self.nbOfActiveWeights: int = 0
        self.nbOfActiveNodeOutputs: int = (
            0  # --- number of active nodes' output links (i.e., links that are on some path from some input to the output)
        )
        self.learnableParams: list[MyTFVariable] = []  # --- parameters optimized by SGD
        self.regularizableParams: list = []
        self.params_2_learn: list[
            tf.Variable
        ]  # --- a list of tf.Variables that are tuned
        self.div_vars: list = []
        self.forwardFullMask = None
        self.front = SRConfig.maxFrontNb
        self.y_hat: tf.Tensor = None
        # --- topology complexity in terms of the number of active weights
        self.complexity: float = 0.0
        # --- Training RMSE
        self.last_rmse_terms: list[float] = []
        self.last_rmse_terms_mean = 0.0
        # --- Regularization
        self.last_regularization_terms = SRConfig.stack_size * [0.0]
        self.reg_weight = SRConfig.reg2rmse
        # --- Singularity nodes
        self.last_raw_sng_terms = []
        self.last_sng_terms = []
        self.sng_weight = SRConfig.sng2rmse
        self.static_sng_weight = None
        # --- Constraints
        self.constraint_weight = SRConfig.constr2rmse
        self.constraint_terms: dict = {}
        self.raw_constraint_terms: dict[str, float] = {}
        self.last_raw_constraints: dict[str, list[float]] = {}
        self.last_constraints_terms: dict[str, list[float]] = {}
        # --- partial loss terms and an overall loss used in the weight update step
        # ---   rmse_train: loss observed on a training data set
        # ---   rmse_valid: loss observed on a validation data set
        # ---   rmse_constr: sum of constraint violation losses
        self.losses: dict[str, float] = {
            "rmse_valid": -1.0,
            "rmse_train": -1.0,
            "rmse_constr": -1.0,
            "nbOfActiveNodes": 0.0,
            "complexity": 0.0,
        }
        for constr in SRConstraints.constraintModules:
            self.last_raw_constraints[constr.name] = SRConfig.stack_size * [0.0]
            self.last_constraints_terms[constr.name] = SRConfig.stack_size * [0.0]
        self.nonDominatedImprovement = False

    def __eq__(self, other):
        for k in SubTopology.keysAll:
            if self.losses[k] != other.losses[k]:
                return False
        return True

    def __hash__(self):
        hashValues = tuple([self.losses[k] for k in SubTopology.keysAll])
        return hash(hashValues)

    def cloneSubTopology(self):
        """
        Clones the subTopology.
        """
        subTopology = SubTopology()
        subTopology.activeNodeCoordinates = copy.deepcopy(self.activeNodeCoordinates)
        subTopology.masterTopologyId = self.masterTopologyId

        for i in range(len(self.nnLayers)):
            newLayer = self.nnLayers[i].clone_layer(
                layerId=i, activeNodes=self.activeNodeCoordinates
            )
            subTopology.learnableParams.extend(newLayer.learnableParams)
            subTopology.nnLayers.append(newLayer)

        subTopology.losses = {
            "rmse_valid": -1.0,
            "rmse_train": -1.0,
            "rmse_constr": -1.0,
            "nbOfActiveNodes": 0,
        }
        subTopology.nbOfActiveNodes = self.nbOfActiveNodes
        subTopology.nbOfActiveWeights = self.nbOfActiveWeights
        subTopology.front = copy.deepcopy(self.front)
        # subTopology.div_vars = copy.deepcopy(self.div_vars)   # --- ORIGINAL
        subTopology.y_hat = None  # --- TODO: remove
        # ---
        return subTopology

    def resetSubTopology(self):
        """
        Resets the whole subTopology so that all units are inactive.
        """
        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                for w in node.weights:
                    w.set_allzero_value_mask()
                self.activeNodeCoordinates[l][n] = False
        self.nbOfActiveNodes = 0
        self.nbOfActiveWeights = 0
        self.nbOfActiveNodeOutputs = 0
        self.losses = {
            "rmse_valid": -1.0,
            "rmse_train": -1.0,
            "rmse_constr": -1.0,
            "nbOfActiveNodes": 0,
        }

    def setProxyNames(self):
        for l, layer in enumerate(self.nnLayers):
            # --- skip the output layer
            if l == len(self.nnLayers) - 1:
                return
            # ---
            for n, node in enumerate(layer.units):
                if "UnitIdent1" in self.nnLayers[l].units[n].name:
                    proxyNode: UnitIdent1 = self.nnLayers[l].units[n]
                    proxyId = proxyNode.inId
                    if l == 0:
                        proxyNode.proxyName = f"x[{proxyId}]"
                    elif "UnitIdent1" not in self.nnLayers[l - 1].units[proxyId].name:
                        proxyNode.proxyName = (
                            self.nnLayers[l - 1].units[proxyId].name
                        )  # --- node[l][n] points to an ordinary function unit
                    else:
                        proxyNode.proxyName = (
                            self.nnLayers[l - 1].units[proxyId].proxyName
                        )  # --- node[l][n] points to the UnitIdent1

    def resetFront(self):
        self.front = SRConfig.maxFrontNb

    def separate_learnable_nonlearnable_params(self):
        self.learnableParams = []
        self.regularizableParams = []
        for l, layer in enumerate(self.nnLayers):
            for u, unit in enumerate(layer.units):
                if (not self.activeNodeCoordinates[l][u]) or (not unit.learnable):
                    continue
                for w in unit.weights:
                    if np.sum(w.mask) > 0.0:
                        self.learnableParams.append(w)
                        # if ' W' in w.v.name:      # --- original
                        if True:
                            self.regularizableParams.append(w)

    def get_params_2_learn(self):
        self.params_2_learn = [p.v for p in self.learnableParams]
        self.params_2_regularize = [p.v for p in self.regularizableParams]
        self.params_2_regularize_shapes = self.get_shapes(self.params_2_regularize)

    def set_forwardpass_masks(self):
        self.forwardFullMask = np.zeros(shape=(SRData.forwardpass_data.shape[0], 1))
        for k in SRData.forwardpass_data_boundaries.keys():
            m = copy.deepcopy(self.forwardFullMask)
            m[
                SRData.forwardpass_data_boundaries[k][
                    0
                ] : SRData.forwardpass_data_boundaries[k][1]
                + 1
            ] = 1.0
            SRData.forwardpass_masks[k] = m

    def layer_feed_forward(self, layer, x_input):
        a, z = layer.forward_pass(x_input)
        if z:
            self.div_vars.extend(z)
        return a, z

    def model_feed_forward(self, x_input):
        self.div_vars = []
        for l in self.nnLayers:
            x_input, z = self.layer_feed_forward(l, x_input)
        return x_input  # --- final layer output

    def get_RMSE(self, y_hat, y_output, counts):
        loss = SRConfig.rmseScaleCoeff * tf.sqrt(
            tf.divide(tf.reduce_sum(tf.square(tf.subtract(y_output, y_hat))), counts)
        )
        return loss

    def getAnalyticFormula(self, simplify: bool = False):
        """
        Constructs analytic expressions of all units.
        :return: Final analytic expression of the whole SubTopology.
        """
        for i, l in enumerate(self.nnLayers):
            for u in l.units:
                u.get_analytic_expression_forwardpass(
                    currLayer=i, layers=self.nnLayers, simplify=simplify
                )
        return self.nnLayers[-1].units[0].string_analytic_expression

    def printAnalyticFormula(self, printSimplified=False):
        outputNode: NNUnit = self.nnLayers[-1].units[0]
        print(f"\tAnalytic formula: {outputNode.string_analytic_expression}")
        if printSimplified:
            print(
                f"\tSimplified analytic formula: {outputNode.symbolic_expression.simpleExpr}"
            )

    def getFormula(self):
        """
        Just returns the analytic formula represented by the SubTopology.
        :return: The analytic expression of the output node.
        """
        outputNode: NNUnit = self.nnLayers[-1].units[0]
        return outputNode.string_analytic_expression

    def printCheckPoints(self):
        print(
            f"!!! CHECK (printBestTopologies): RMSE_valid: {self.losses['rmse_valid']}"
        )
        x_in, y_gt, y_hat, y_formula = self.calculateCheckPoints()
        print("Check points:")
        for x, y, z, u in zip(x_in, y_gt, y_hat, y_formula):
            print(f"\t{x[0]},{x[1]}: {y[0]} / {z[0]:2.3} / {u[0]:2.3}")

    def evaluateAnalyticFormula(self, x0, x1):
        outputNode: NNUnit = self.nnLayers[-1].units[0]
        if len(outputNode.string_analytic_expression) > 0:
            # print(f'\tAnalytic formula: {outputNode.string_analytic_expression}')
            return eval(outputNode.string_analytic_expression)
        else:
            return np.zeros(shape=(x0.shape))

    def get_shapes(self, reg_vars):
        shapes = [(v.shape.dims[0].value, v.shape.dims[1].value) for v in reg_vars]
        list_set = set(shapes)
        shapes = list(list_set)
        return shapes

    def get_regularization_term(self, reg_vars=None, reg_lambda=0.1, shapes=None):
        if reg_lambda == 0.0:
            return tf.constant(0.0, dtype=np.float64)
        # ---
        if shapes is None:
            shapes = self.get_shapes(reg_vars)
        # ---
        regularizer_L05 = L05Regularizer(strength=reg_lambda, a=0.01)
        # ---
        reg_terms = []
        for s in shapes:
            new_vars = [v for v in reg_vars if v.shape == s]
            reg_term = regularizer_L05(new_vars)
            reg_terms.append(reg_term)
        if reg_terms:
            return tf.math.add_n(reg_terms)
        else:
            return tf.constant(0.0, dtype=np.float64)

    def get_division_penalty_term(self, div_vars, theta=1e-3):
        if (not div_vars) or (tf.math.count_nonzero(div_vars) == 0):
            return tf.constant(0.0, dtype=np.float64)
        # --- Consider all data points stored in SRData.forwardpass_data
        p = [tf.reduce_sum(tf.maximum(tf.subtract(theta, dv), 0.0)) for dv in div_vars]
        p = tf.reduce_sum(p)
        return p

    def calculate_constraints_terms_weight_new(self):
        sum = 0.0
        for constr in SRConstraints.constraintModules:
            sum += np.nanmean(self.last_constraints_terms[constr.name])
        if sum == 0.0:
            self.constraint_weight = 1.0
        else:
            self.constraint_weight = SRConfig.constr2rmse * (
                self.last_rmse_terms_mean / sum
            )

    def add_constraint_terms(self, c_terms, raw_constraint_terms, updateWeights=False):
        sum = 0.0  # --- sum of normalized constraint terms
        for constr in SRConstraints.constraintModules:
            new_constraint_term = constr.module.get_constraint_term(
                constr=constr, data=SRData, y_hat=self.y_hat, weight=1.0
            )
            raw_constraint_terms[constr.name] = new_constraint_term
            self.last_raw_constraints[constr.name].append(new_constraint_term.numpy())
            self.last_raw_constraints[constr.name] = self.last_raw_constraints[
                constr.name
            ][1:]
            if SRConfig.adaptive:
                m = np.mean(
                    self.last_raw_constraints[constr.name]
                )  # --- mean of the last RAW constraint term values
                if m > 0.0:
                    new_constraint_term /= m
            else:
                # new_constraint_term = tf.divide(new_constraint_term, SRConfig.constr2rmse)
                new_constraint_term = SRConfig.constr2rmse * new_constraint_term
            self.last_constraints_terms[constr.name].append(new_constraint_term.numpy())
            self.last_constraints_terms[constr.name] = self.last_constraints_terms[
                constr.name
            ][1:]
            sum += new_constraint_term
            # ---
            c_terms[constr.name] = new_constraint_term

        if SRConfig.adaptive:
            # -----------------------------------
            # ------- Update constraint_weight
            # -----------------------------------
            if updateWeights:
                self.calculate_constraints_terms_weight_new()

            # ---------------------------------
            # ------- Weight c_terms and sum
            # ---------------------------------
            sum *= self.constraint_weight
            for constr in SRConstraints.constraintModules:
                c_terms[constr.name] *= self.constraint_weight

            # ---------------------------------
            # ------- Adjust c_terms
            # ---------------------------------
            if sum > SRConfig.constr2rmse * self.last_rmse_terms_mean:
                r = sum / (SRConfig.constr2rmse * self.last_rmse_terms_mean)
                for constr in SRConstraints.constraintModules:
                    c_terms[constr.name] = tf.divide(c_terms[constr.name], r)

        return c_terms, raw_constraint_terms

    def updateLearnableParamsMaskAndValues(self):
        """
        Updates p.mask and p.v based on current p.v
        """
        # --- create curr_params_mask based on actual p.v values
        curr_params_mask = [
            np.array(np.abs(p.v) >= SRConfig.activeWeightThreshold, dtype="float64")
            for p in self.learnableParams
        ]
        # [p.set_mask(np.multiply(m, p.mask)) for m, p in zip(curr_params_mask, self.learnableParams)]
        for m, p in zip(curr_params_mask, self.learnableParams):
            p.set_mask(np.multiply(m, p.mask))
        # --- Mask p.v according to the new p.mask
        # [p.v.assign(tf.multiply(p.v, p.mask), read_value=True) for p in self.learnableParams]   # --- ORIGINAL
        for p in self.learnableParams:
            p.v.assign(np.multiply(p.v, p.mask), read_value=True)

    def calculateValidationRMSE(self):
        """
        Calculates RMSE on 'valid' data
        :return:
        """
        # self.set_forwardpass_masks()
        x_input = SRData.forwardpass_data
        x_input[
            SRData.forwardpass_data_boundaries["valid"][
                0
            ] : SRData.forwardpass_data_boundaries["valid"][1]
            + 1,
            :,
        ] = SRData.x_val
        self.y_hat = self.model_feed_forward(x_input)
        # --- Just for debugging
        vvv = self.y_hat.numpy()
        # if np.all(vvv == vvv[0]):
        #     print('constant expression')
        # ---
        y_output = copy.copy(self.forwardFullMask)
        y_output[
            SRData.forwardpass_data_boundaries["valid"][
                0
            ] : SRData.forwardpass_data_boundaries["valid"][1]
            + 1,
            :,
        ] = SRData.y_val
        rmse_valid = self.get_RMSE(
            tf.multiply(self.y_hat, SRData.forwardpass_masks["valid"]),
            y_output,
            SRData.forwardpass_counts["valid"],
        )
        # self.losses['rmse_valid'] = round(rmse_valid.numpy(), SRConfig.precisionDigits)
        res = round(rmse_valid.numpy(), SRConfig.precisionDigits)
        return res

    def calculateRMSE(self, x: np.array, y: np.array):
        """
        Calculates RMSE on a given data set.
        :return:
        """
        # y_hat = self.model_feed_forward(x)
        # mask = np.zeros((y.shape[0], 1))
        # mask[0:y.shape[0]] = 1
        # rmse_valid = self.get_RMSE(tf.multiply(y_hat, mask), y, counts=y.shape[0])
        # ---
        y_hat = self.model_feed_forward(x)
        rmse_valid = self.get_RMSE(y_hat, y, counts=y.shape[0])
        return rmse_valid, y_hat[: y.shape[0], 0]

    def calculateConstrainViolationsRMSE(self, step: int = 1):
        self.constraint_terms = {}
        self.raw_constraint_terms = {}
        self.constraint_terms, self.raw_constraint_terms = self.add_constraint_terms(
            self.constraint_terms,
            self.raw_constraint_terms,
            updateWeights=(step % SRConfig.update_step == 0),
        )
        c_raw = float(np.sum([el.numpy() for el in self.raw_constraint_terms.values()]))
        c_weighted = float(
            np.sum([el.numpy() for el in self.constraint_terms.values()])
        )
        return round(c_raw, SRConfig.precisionDigits), round(
            c_weighted, SRConfig.precisionDigits
        )

    # --- TODO: implement
    def acceptanceCriterionRMSEValidConstraints(
        self, bsf_rmse_valid: float, bsf_constraint_terms: dict
    ):
        """
        Acceptance criterion based on Constraints Violation Error.
        Accepts this solution if none of the constraint values worsens: sym, half, leR, grR
        """
        if bsf_constraint_terms is None:
            return True
        # ---
        for k in self.raw_constraint_terms:
            if self.raw_constraint_terms[k] > bsf_constraint_terms[k]:
                return False
        if self.losses["rmse_valid"] > bsf_rmse_valid:
            return False
        # --- If everything is OK
        return True

    def checkUnitIdent1Nodes(self):
        """
        Checks if all UnitIdent1 nodes have at most one active input weight.
        :return:
        """
        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                if (self.activeNodeCoordinates[l][n]) and ("UnitIdent1" in node.name):
                    for w in node.weights:
                        if " b" not in w.v.name:
                            nonZeroValues = w.v.numpy().nonzero()
                            if len(nonZeroValues[0]) > 1:
                                return False  # --- more than one active weights
        return True  # --- OK

    def findBelowThresholdActiveWeights(self, learnableParams: list[MyTFVariable]):
        """
        Find all active weights within the params_2_learn that fall below threshold.
        :return: list of tuples where each tuple is (varId, weightId)
        """
        result = []
        for i, p in enumerate(learnableParams):
            belowThresholdValues = np.array(
                np.abs(p.v.numpy()) < SRConfig.activeWeightThreshold, dtype=int
            ).reshape(p.v.shape[0], 1)
            newBelowThresholdValues = np.logical_and(belowThresholdValues, p.mask != 0)
            if sum(newBelowThresholdValues) > 0:
                # print(f'\n>>> newBelowThresholdValues: {i} / {p.v.name} / {hash(self)}')
                # print('\tweights:', end='')
                for j in range(p.v.shape[0]):
                    if newBelowThresholdValues[j]:
                        # print(f' {j}', end='')
                        result.append((i, j))
                # print('')
        return result

    def findUnitsWithBelowThresholdOutput(self):
        """
        Finds all active units that have the absolute value of its output less than the SRConfig.activeWeightThreshold
        for all input vectors.
        :return: list of tuples where each tuple is (l, n), i.e., coordinates of the below-threshold output unit
        """
        result = []
        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:  # --- it is an active unit
                    aboveThresholdOutputValues = np.array(
                        np.abs(node.a.numpy()) > SRConfig.pruningOutputThreshold,
                        dtype=int,
                    )
                    if sum(aboveThresholdOutputValues) == 0:
                        result.append((l, n))
        return result

    def pruneUnitsWithBelowThresholdOutput(self):
        """
        Prunes units that have below-threshold output
        """
        if self.y_hat is None:
            self.forwardPass(data=SRData.forwardpass_data)
        belowThresholdOutputNodes = self.findUnitsWithBelowThresholdOutput()
        # --- Inactivate belowThresholdOutputNodes
        if belowThresholdOutputNodes:
            print("\n >>>>>>> Below-threshold output nodes: ", end="")
            for el in belowThresholdOutputNodes:
                print(f" ({el[0]}, {el[1]})", end="")
                self.activeNodeCoordinates[el[0]][el[1]] = False
                self.nbOfActiveNodes -= 1
                for w in self.nnLayers[el[0]].units[el[1]].weights:
                    w.set_allzero_value_mask()
                # --- TODO: inactivate all weights taking the inactivated unit as its input
            print("")
            return True  # --- at least one un it has been pruned
        return False  # --- no pruning carried out

    def pruneWeights(self):
        """
        Prunes weights of each active unit.
        Weights with below-threshold value of abs(input_i * w_i) are set to zero and the variable's mask is updated accordingly.
        """
        if self.y_hat is None:
            self.forwardPass(data=SRData.forwardpass_data)
        # -----------------------------------------------
        # --- Forward pass for all layers
        # -----------------------------------------------
        result = False  # --- flag if any weight was pruned
        for l, layer in enumerate(self.nnLayers):
            # --- create input to the l-th layer, i.e., output matrix composed of the 'l-1' layer's nodes output
            if l == 0:
                inputX = SRData.forwardpass_data
            else:
                inputX = np.empty(
                    (SRData.forwardpass_data.shape[0], len(self.nnLayers[l - 1].units))
                )
                for u, unit in enumerate(self.nnLayers[l - 1].units):
                    inputX[:, u] = unit.a.numpy()[:, 0]
            # -----------------------------------------------
            # --- Find weights to be pruned in this layer
            # -----------------------------------------------
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:
                    currResult = False
                    for w in node.weights:
                        if " b" not in w.v.name:
                            nbOfActiveWeights_old = int(sum(w.mask)[0])
                            inputTimesW = np.abs(
                                inputX * w.v.numpy().T
                            )  # --- calculate values passed via all input edges to the variable 'w'
                            # --------------------------------------------------------------------------------------
                            # --- Variant A: Works with the largest absolute value of inputTimesW for each weight
                            # --------------------------------------------------------------------------------------
                            maxAbsValuePerWeight = np.max(inputTimesW, axis=0)
                            maxAbsValuePerWeight = maxAbsValuePerWeight.reshape(
                                (maxAbsValuePerWeight.shape[0], 1)
                            )
                            maxAbsWeight = np.max(maxAbsValuePerWeight)
                            minAcceptableWeightAbsValue = (
                                SRConfig.pruningMinWeightRatio * maxAbsWeight
                            )  # --- minimum acceptable absolute value for the weight to stay active
                            # ---
                            okWeights = np.array(
                                maxAbsValuePerWeight >= minAcceptableWeightAbsValue,
                                dtype=int,
                            ).reshape((maxAbsValuePerWeight.shape[0], 1))
                            newWeights_A = np.multiply(w.v.numpy(), okWeights)
                            nbOfActiveWeights_new_A = np.sum(newWeights_A != 0.0)
                            # ---------------------------------------------------------------------------------------
                            # --- Variant A2: Works with the largest quantile value of inputTimesW for each weight
                            # ---------------------------------------------------------------------------------------
                            # maxAbsValuePerWeight = np.quantile(a=inputTimesW, q=0.9, axis=0)
                            # maxAbsValuePerWeight = maxAbsValuePerWeight.reshape((maxAbsValuePerWeight.shape[0], 1))
                            # maxAbsWeight = np.max(maxAbsValuePerWeight)
                            # minAcceptableWeightAbsValue = SRConfig.pruningMinWeightRatio * maxAbsWeight   # --- minimum acceptable absolute value for the weight to stay active
                            # # ---
                            # okWeights = np.array(maxAbsValuePerWeight >= minAcceptableWeightAbsValue, dtype=int).reshape((maxAbsValuePerWeight.shape[0], 1))
                            # newWeights_A2 = np.multiply(w.v.numpy(), okWeights)
                            # nbOfActiveWeights_new_A2 = np.sum(newWeights_A2 != 0.0)
                            # ------------------------------------------------------------------------------------------
                            # --- Variant B:
                            # ---     1) Calculate a median inputTimesW value for each weight
                            # ---     2) Find the most significant weight with the maximum median inputTimesW value
                            # ---     3) Keep only the weights that have the median ratio of its inputTimesW to the inputTimesW
                            # ---        of the most significant weight larger than or equal to the SRConfig.pruningMinWeightRatio
                            # ------------------------------------------------------------------------------------------
                            # medianAbsValuePerWeight = np.median(inputTimesW, axis=0)
                            # medianAbsValuePerWeight = medianAbsValuePerWeight.reshape((medianAbsValuePerWeight.shape[0], 1))
                            # mostSignificantWeightId = np.argmax(medianAbsValuePerWeight, axis=0)   # --- index of the most significant weight
                            # mostSignificantWeightValues = inputTimesW[:, mostSignificantWeightId].reshape((inputTimesW.shape[0], 1))
                            # ratioWeightToMostSignificantWeight = np.divide(inputTimesW, mostSignificantWeightValues)
                            # medianRatioPerWeight = np.median(ratioWeightToMostSignificantWeight, axis=0)
                            # # ---
                            # okWeights = np.array(medianRatioPerWeight >= SRConfig.pruningMinWeightRatio, dtype=int).reshape((inputX.shape[1], 1))
                            # newWeights_B = np.multiply(w.v.numpy(), okWeights)
                            # nbOfActiveWeights_new_B = np.sum(newWeights_B != 0.0)
                            # ----------------------------------------------------------------------------
                            # --- Choose the variant
                            # ----------------------------------------------------------------------------
                            w.set_value_mask(newValues=newWeights_A)
                            if nbOfActiveWeights_old > nbOfActiveWeights_new_A:
                                print(
                                    f"\n\t >>>>>>>  A) Below-threshold weight [{l}][{n}]: origActiveWeights={nbOfActiveWeights_old}, newActiveWeights={nbOfActiveWeights_new_A} "
                                )
                                # print(f'\t >>>>>>> A2) Below-threshold weight [{l}][{n}]: origActiveWeights={nbOfActiveWeights_old}, newActiveWeights={nbOfActiveWeights_new_A2} ')
                                # print(f'\t >>>>>>>  B) Below-threshold weight [{l}][{n}]: origActiveWeights={nbOfActiveWeights_old}, newActiveWeights={nbOfActiveWeights_new_B} ')
                                currResult = True
                            if nbOfActiveWeights_new_A == 0:
                                self.activeNodeCoordinates[l][n] = False
                                self.nbOfActiveNodes -= 1
                                print(
                                    f"\t    >>>>>>> pruneWeights completely eliminated unit[{l}][{n}]"
                                )
                    # if currResult:
                    #     self.forwardPass(data=SRData.forwardpass_data)  # --- Something changed in the subTopology, so update the self.y_hat
                    result = result or currResult
        return result

    def pruneWeights_new(self):
        """
        Prunes weights of each active unit.
        Weights with below-threshold value of abs(input_i * w_i) are set to zero and the variable's mask is updated accordingly.
        """
        if self.y_hat is None:
            self.forwardPass(data=SRData.forwardpass_data)
        # -----------------------------------------------
        # --- Forward pass over all layers
        # -----------------------------------------------
        result = False  # --- flag if any weight was pruned
        for l, layer in enumerate(self.nnLayers):
            # ---------------------------------------------------------------------
            # --- create input to the l-th layer,
            # --- i.e., output matrix composed of the 'l-1' layer's nodes output
            # ---------------------------------------------------------------------
            if l == 0:
                inputX = SRData.forwardpass_data
            else:
                inputX = np.empty(
                    (SRData.forwardpass_data.shape[0], len(self.nnLayers[l - 1].units))
                )
                for u, unit in enumerate(self.nnLayers[l - 1].units):
                    inputX[:, u] = unit.a.numpy()[:, 0]
            # -----------------------------------------------
            # --- Find weights to be pruned in this layer
            # -----------------------------------------------
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:
                    currResult = False
                    for w in node.weights:
                        if " b" not in w.v.name:
                            # ----------------
                            # --- Weights W
                            # ----------------
                            nbOfActiveWeights_old = int(sum(w.mask)[0])
                            inputTimesW = np.abs(
                                inputX * w.v.numpy().T
                            )  # --- calculate values passed via all input edges to the variable 'w'
                            # --------------------------------------------------------------------------------------
                            # --- Works with the largest absolute value of inputTimesW for each weight
                            # --------------------------------------------------------------------------------------
                            maxAbsValuePerWeight = np.max(inputTimesW, axis=0)
                            maxAbsValuePerWeight = maxAbsValuePerWeight.reshape(
                                (maxAbsValuePerWeight.shape[0], 1)
                            )
                            maxAbsWeight = np.max(maxAbsValuePerWeight)
                            # --- Set the minimum acceptable value for the weight to stay active
                            # minAcceptableWeightAbsValue = SRConfig.pruningMinWeightRatio   # --- Absolute
                            minAcceptableWeightAbsValue = (
                                SRConfig.pruningMinWeightRatio * maxAbsWeight
                            )  # --- Relative
                            # ---
                            okWeights = np.array(
                                maxAbsValuePerWeight >= minAcceptableWeightAbsValue,
                                dtype=int,
                            ).reshape((maxAbsValuePerWeight.shape[0], 1))
                            newWeights = np.multiply(w.v.numpy(), okWeights)
                            nbOfActiveWeights_new = np.sum(newWeights != 0.0)
                            # ----------------------------------------------------------------------------
                            # --- Choose the variant
                            # ----------------------------------------------------------------------------
                            w.set_value_mask(newValues=newWeights)
                            if nbOfActiveWeights_old > nbOfActiveWeights_new:
                                print(
                                    f"\n\t >>>>>>>  A) Below-threshold weight [{l}][{n}]: origActiveWeights={nbOfActiveWeights_old}, newActiveWeights={nbOfActiveWeights_new} "
                                )
                                currResult = True
                            if nbOfActiveWeights_new == 0:
                                self.activeNodeCoordinates[l][n] = False
                                self.nbOfActiveNodes -= 1
                                print(
                                    f"\t    >>>>>>> pruneWeights completely eliminated unit[{l}][{n}]"
                                )
                        else:
                            # ----------------
                            # --- Bias b
                            # ----------------
                            if w.v.numpy() < SRConfig.pruningMinWeightRatio:
                                w.set_allzero_value_mask()
                                currResult = True
                    result = result or currResult
        return result

    def forwardPass(self, data):
        # x_input = tf.Variable(data)
        x_input = data
        self.y_hat = self.model_feed_forward(x_input)

    def train_nn(
        self,
        epochLength: int = SRConfig.epochLength,
        learningSteps: int = 0,
        learning_subset_size=1.0,
        learningRate: float = 0.010,
        reduceLearningRate=False,
        finalTuning: bool = False,
        regularize=False,
        clipWeights: bool = True,
        deactivateBelowThresholdUnits: bool = False,
        constrain=True,
        alternatePrefs=False,
        printStep=0,
    ):
        """
        :param learningSteps:
        :param learning_subset_size: a proportion of variables to update in each learning iteration
        :param learningRate:
        :param finalTuning: True ... rmse_valid and rmse_constr are recalculated at the beginning ot the tuning procedure
        :param regularize:
        :param clipWeights: True ... weights are clipped to 0, if they have a below-threshold value
        :param reduceLearningRate:
        :param deactivateBelowThresholdUnits:
        :param constrain:   True ... constraints are considered in the learning process.
        :param alternatePrefs: True, if goal objectives are randomly chosen. False, use self.keysPerformance.
        :param printStep:
        :return:
        """

        tf_seed = SRConfig.r.integers(1, 1e6)
        tf.random.set_seed(tf_seed)
        eps = 1e-05
        nbOfNondominatedImprovements = 0

        # ------------------------------------
        # --- Learning rate decay schedule
        # ------------------------------------
        if learningSteps < 10:
            lr_boundaries = [int(learningSteps / 2)]
            lr_values = [learningRate, learningRate / 2.0]
        else:
            lr_boundaries = [int(learningSteps * (i / 10.0)) for i in range(1, 10, 1)]
            lr_values = [
                learningRate * (i / 10.0) for i in range(10, 0, -1)
            ]  # --- i: 10, 9, ..., 1
        learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            lr_boundaries, lr_values
        )
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_fn, epsilon=eps
        )

        # ------------------------------------------------
        # --- Prepare learnable variables
        # ------------------------------------------------
        self.separate_learnable_nonlearnable_params()
        self.get_params_2_learn()
        self.set_forwardpass_masks()

        # -------------------------------------------------------
        # --- Recalculate rmse_valid and rmse_constr
        # -------------------------------------------------------
        self.losses["nbOfActiveNodes"] = self.nbOfActiveNodes
        self.losses["complexity"] = self.complexity
        rmse_valid = self.calculateValidationRMSE()
        if (self.losses["rmse_valid"] > 0.0) and (
            self.losses["rmse_valid"] != rmse_valid
        ):
            print(
                f"\n !!!!!!! Different validation RMSE: {self.losses['rmse_valid']} vs. {rmse_valid}"
            )
        self.losses["rmse_valid"] = rmse_valid
        if constrain:
            c_raw, c_weighted = self.calculateConstrainViolationsRMSE(step=0)
            self.losses["rmse_constr"] = c_raw
        else:
            self.losses["rmse_constr"] = 0

        # ------------------------------------------------
        bsfLosses: dict[str, float] = copy.deepcopy(self.losses)
        bsfParams2Learn = None
        bsfNondominatedLosses: dict[str, float] = copy.deepcopy(self.losses)
        bsfNondominatedParams2Learn = None
        currSubTopology: bsfSubTopology = bsfSubTopology(
            0, self.losses, self.params_2_learn
        )
        nonDominatedImprovements: list[bsfSubTopology] = [currSubTopology]
        # ---
        self.last_rmse_terms = []
        self.last_regularization_terms = []
        # ---
        if alternatePrefs:
            currPerformanceKeys = [
                k for k in self.keysPerformance if SRConfig.r.random() < 0.5
            ]
            if not currPerformanceKeys:
                currPerformanceKeys = self.keysPerformance
        else:
            currPerformanceKeys = self.keysPerformance
        # ---
        x_input: tf.Variable
        y_output: tf.Variable
        step = 0
        mini_batch_x = None
        mini_batch_y = None
        batchId = SRConfig.r.choice(range(len(SRData.miniBatches)), 1)[0]
        # ==============================================================================================================
        # ======= Main loop: learningSteps iterations
        # ==============================================================================================================
        while step < learningSteps:
            if step % epochLength == 0:
                mini_batch_x = SRData.miniBatches[batchId][0]
                mini_batch_y = SRData.miniBatches[batchId][1]
                batchId = SRConfig.r.choice(range(len(SRData.miniBatches)), 1)[0]

            # --------------------------------------------------------------------------------------
            # --- Calculation of loss, regularization, division penalty, and constraints terms
            # --------------------------------------------------------------------------------------
            loss_terms = []
            x_input_data = SRData.forwardpass_data
            x_input_data[0 : SRConfig.minibatchSize, :] = mini_batch_x
            y_output_data = copy.copy(self.forwardFullMask)
            y_output_data[0 : SRConfig.minibatchSize, :] = mini_batch_y
            # ---
            with tf.GradientTape(persistent=True) as tape:
                x_input = tf.Variable(x_input_data)
                y_output = tf.Variable(y_output_data)
                self.y_hat = self.model_feed_forward(x_input)
                self.updateLossTermMeans: bool
                if (
                    step >= SRConfig.update_step and step % SRConfig.update_step == 0
                ) or (step < SRConfig.update_step):
                    self.updateLossTermMeans = True
                else:
                    self.updateLossTermMeans = False
                # ---------------------------------------------
                # --- Training RMSE
                # ---------------------------------------------
                self.rmse_train = self.get_RMSE(
                    tf.multiply(self.y_hat, SRData.forwardpass_masks["train"]),
                    y_output,
                    SRData.forwardpass_counts["train"],
                )
                # if 'rmse_valid' in currPerformanceKeys:
                loss_terms.append(self.rmse_train)
                self.last_rmse_terms.append(self.rmse_train.numpy())
                if len(self.last_rmse_terms) > SRConfig.stack_size:
                    self.last_rmse_terms = self.last_rmse_terms[1:]
                if step % SRConfig.update_step == 0:
                    self.last_rmse_terms_mean = np.mean(self.last_rmse_terms)
                v = round(
                    self.rmse_train.numpy(), SRConfig.precisionDigits
                )  # --- TODO: v = round(float(self.rmse_train.numpy(), SRConfig.precisionDigits)
                self.losses["rmse_train"] = v
                # ---------------------------------------------
                # --- Validation RMSE
                # ---------------------------------------------
                rmse_valid = self.calculateValidationRMSE()
                self.losses["rmse_valid"] = rmse_valid
                # ---------------------------------------------
                # --- Regularization
                # ---------------------------------------------
                if regularize:
                    reg_term = self.get_regularization_term(
                        reg_vars=self.params_2_regularize,
                        reg_lambda=1.0,
                        shapes=self.params_2_regularize_shapes,
                    )
                    self.last_regularization_terms.append(reg_term.numpy())
                    if len(self.last_regularization_terms) > SRConfig.stack_size:
                        self.last_regularization_terms = self.last_regularization_terms[
                            1:
                        ]
                    if SRConfig.adaptive:
                        b = np.mean(self.last_regularization_terms)
                        if b != 0:
                            self.reg_weight = (
                                SRConfig.reg2rmse * self.last_rmse_terms_mean / b
                            )
                            reg_term *= self.reg_weight
                        else:
                            reg_term *= SRConfig.reg2rmse
                    else:
                        reg_term *= SRConfig.reg2rmse
                else:
                    reg_term = tf.constant(0.0, dtype=np.float64)
                loss_terms.append(reg_term)
                # ---------------------------------------------
                # --- Incorrect denominator's values
                # ---------------------------------------------
                if constrain:
                    theta_div = 1e-4
                    if not SRConfig.adaptive:
                        # # --- static: static S3
                        division_term = self.get_division_penalty_term(
                            div_vars=self.div_vars, theta=theta_div
                        )
                        if self.static_sng_weight is None:
                            self.static_sng_weight = (
                                SRConfig.sng2rmse
                                * self.rmse_train.numpy()
                                / division_term.numpy()
                            )
                        division_term *= self.static_sng_weight
                    else:
                        # --- adaptive
                        division_term = self.get_division_penalty_term(
                            div_vars=self.div_vars, theta=theta_div
                        )
                        self.last_raw_sng_terms.append(division_term.numpy())
                        if len(self.last_raw_sng_terms) > SRConfig.stack_size:
                            self.last_raw_sng_terms = self.last_raw_sng_terms[1:]
                        b = np.mean(self.last_raw_sng_terms)
                        # --- Variant A: current paper experiments
                        if b > 0:
                            norm_div_term = division_term.numpy() / b
                            self.last_sng_terms.append(norm_div_term)
                            if len(self.last_sng_terms) > SRConfig.stack_size:
                                self.last_sng_terms = self.last_sng_terms[1:]
                            self.sng_weight = (
                                SRConfig.sng2rmse
                                * self.last_rmse_terms_mean
                                / np.mean(self.last_sng_terms)
                            )
                        else:
                            self.sng_weight = 1.0
                        division_term = tf.multiply(self.sng_weight, division_term)
                        loss_terms.extend([division_term])
                # self.constraint_terms['div'] = division_term
                # ---------------------------------------------
                # --- Constraint violations
                # ---------------------------------------------
                if constrain:
                    c_raw, c_weighted = self.calculateConstrainViolationsRMSE(step=step)
                    self.losses["rmse_constr"] = c_raw
                    # if 'rmse_constr' in currPerformanceKeys:
                    loss_terms.extend(self.constraint_terms.values())
                else:
                    c_raw = 0.0
                    c_weighted = 0.0
                    self.losses["rmse_constr"] = 0
                # ---
                loss = tf.math.add_n(loss_terms)

            # ---------------------------------------------
            # --- Stats
            # ---------------------------------------------
            if printStep > 0 and step % printStep == 0:
                print(
                    f"\t\t {step}.\trmse_train={self.rmse_train:2.8f} / rmse_valid={rmse_valid:2.8f} / constr_raw={c_raw:2.8f} / constr_term={c_weighted:2.8f} / reg_term={reg_term.numpy():2.8f}"
                )

            # ---------------------------------------------
            # --- Update best-so-far self.params_2_learn
            # ---------------------------------------------
            firstDominates, secondDominates, firstEqualsSecond = getMutualDominance(
                first=self.losses,
                second=bsfNondominatedLosses,
                criteria=SubTopology.keysPerformance,
            )
            if firstDominates and not firstEqualsSecond:
                bsfNondominatedLosses = copy.deepcopy(self.losses)
                bsfNondominatedParams2Learn = copy.deepcopy(self.params_2_learn)
                nonDominatedImprovements.append(
                    bsfSubTopology(step, self.losses, self.params_2_learn)
                )
                nbOfNondominatedImprovements += 1
            elif alternatePrefs:
                firstDominates, secondDominates, firstEqualsSecond = getMutualDominance(
                    first=self.losses, second=bsfLosses, criteria=currPerformanceKeys
                )
                if firstDominates and not firstEqualsSecond:
                    bsfLosses = copy.deepcopy(self.losses)
                    bsfParams2Learn = copy.deepcopy(self.params_2_learn)

            # ---------------------------------------------
            # --- Update weights using the gradients
            # ---------------------------------------------
            try:
                if learning_subset_size == 1.0:
                    # --- update all variables
                    gradients = tape.gradient(loss, self.params_2_learn)
                    # V = [np.max(np.abs(tfvar.numpy())) for tfvar in gradients]
                    # I = list(range(len(V)))
                    # # K = [f'{index}-{tfvar.name}' for index,tfvar in zip(I, self.params_2_learn)]
                    # # maxGrads = dict(zip(K,V))
                    # # maxGrads = dict(sorted(maxGrads.items(), key=lambda item: item[1], reverse=True))
                    # maxGradPossible = 0.5
                    # grads = [tfvar.numpy() for tfvar in gradients]
                    # clippedGrads = [np.clip(g, -maxGradPossible, maxGradPossible) for g in grads]
                    optimizer.apply_gradients(zip(gradients, self.params_2_learn))
                else:
                    # --- choose only some variables to update; one layer at a time?
                    n = int(len(self.params_2_learn) * learning_subset_size)
                    idx = list(range(len(self.params_2_learn)))
                    SRConfig.r.shuffle(idx)
                    idx = idx[:n]
                    idx.sort()
                    selected_params = [self.params_2_learn[i] for i in idx]
                    gradients = tape.gradient(loss, selected_params)
                    optimizer.apply_gradients(zip(gradients, selected_params))
            except Exception as e:
                print(
                    f"\n\nError in optimizer.apply_gradients(zip(gradients, self.params_2_learn)): {e}\n\n"
                )

            # -----------------------------------------------
            # --- Update p.mask based on current p.v
            # -----------------------------------------------
            if clipWeights:
                self.updateLearnableParamsMaskAndValues()  # --- ORIGINAL: uncomment

            SubTopology.backpropCount += 1
            step = step + 1
            if step % 1000 == 0:
                print(f"\tstep: {step}")
        # ==============================================================================================================
        # ======= Main loop: End
        # ==============================================================================================================

        # ---------------------------------------------------
        # --- Set p.v according to bsfParams2Learn
        # ---------------------------------------------------
        if (
            bsfNondominatedParams2Learn is None
        ) and learningSteps > SRConfig.totalItersNewbornNoPerturbations:
            if bsfParams2Learn is not None:
                print(
                    f"!!! NO NON-DOMINATED IMPROVEMENT !!! ({bsfLosses['rmse_valid']} / {bsfLosses['rmse_constr']})  "
                )
            else:
                print(
                    f"!!! NO NON-DOMINATED IMPROVEMENT !!! ({self.losses['rmse_valid']} / {self.losses['rmse_constr']})  "
                )
        # ---
        if bsfNondominatedParams2Learn is not None:
            originNonDominatedImprovement = True
            self.nonDominatedImprovement = True
            self.losses["rmse_train"] = bsfNondominatedLosses["rmse_train"]
            for oldV, newV in zip(self.params_2_learn, bsfNondominatedParams2Learn):
                oldV.assign(newV.value())
            # for i, newV in enumerate(bsfNondominatedParams2Learn):
            #     self.learnableParams[i].set_value_mask(newV.value())

        else:
            originNonDominatedImprovement = False
            self.nonDominatedImprovement = False
            if bsfParams2Learn is not None:
                self.losses["rmse_train"] = bsfLosses["rmse_train"]
                for oldV, newV in zip(self.params_2_learn, bsfParams2Learn):
                    oldV.assign(newV.value())
                # for i, newV in enumerate(bsfParams2Learn):
                #     self.learnableParams[i].set_value_mask(newV.value())

        # ---------------------------------------------------
        # --- Set p.mask according to p.v
        # --- for all MyTFVariables in self.learnableParams
        # ---------------------------------------------------
        # --- If clipWeights==True then useless
        # for myTFVar in self.learnableParams:
        #     myTFVar.infer_mask()

        # ---------------------------------------------------
        # --- Update p.mask and p.v based on current p.v
        # ---------------------------------------------------
        if clipWeights:
            self.updateLearnableParamsMaskAndValues()

        somethingWasPruned = False
        # ---------------------------------------------------
        # --- Prune units that have below-threshold output
        # ---------------------------------------------------
        if deactivateBelowThresholdUnits and SRConfig.pruning:
            somethingWasPruned = (
                somethingWasPruned or self.pruneUnitsWithBelowThresholdOutput()
            )

        # ---------------------------------------------------
        # --- Prune weights that have below-threshold
        # --- ratio 'abs(w)/max(abs(W))'
        # ---------------------------------------------------
        if deactivateBelowThresholdUnits and SRConfig.pruning:
            somethingWasPruned = somethingWasPruned or self.pruneWeights_new()

        # -----------------------------------------------
        # --- Update the number of active nodes
        # -----------------------------------------------
        self.updateActiveNodes()

        # -----------------------------------------------
        # --- RMSE on validation data
        # -----------------------------------------------
        self.losses["rmse_valid"] = self.calculateValidationRMSE()
        if not somethingWasPruned:
            if bsfNondominatedParams2Learn is not None:
                if self.losses["rmse_valid"] != bsfNondominatedLosses["rmse_valid"]:
                    # print(
                    #     f"!!! END Unequal rmse_valid: {self.losses['rmse_valid']} vs. {bsfNondominatedLosses['rmse_valid']}"
                    # )
                    self.losses["rmse_valid"] = bsfNondominatedLosses["rmse_valid"]
            elif (bsfParams2Learn is not None) and (
                self.losses["rmse_valid"] != bsfLosses["rmse_valid"]
            ):
                # print(
                #     f"!!! END Unequal rmse_valid: {self.losses['rmse_valid']} vs. {bsfLosses['rmse_valid']}"
                # )
                self.losses["rmse_valid"] = bsfLosses["rmse_valid"]

        # -------------------------------
        # --- Constraint violations
        # -------------------------------
        if constrain:
            c_raw, c_weighted = self.calculateConstrainViolationsRMSE(step=step)
            self.losses["rmse_constr"] = c_raw
        if not somethingWasPruned:
            if bsfNondominatedParams2Learn is not None:
                if (
                    0.0
                    < bsfNondominatedLosses["rmse_constr"]
                    != self.losses["rmse_constr"]
                ):
                    # print(
                    #     f"!!! a) END Unequal rmse_constr: {self.losses['rmse_constr']} vs. {bsfNondominatedLosses['rmse_constr']}"
                    # )
                    self.losses["rmse_constr"] = bsfNondominatedLosses["rmse_constr"]
            elif (
                (bsfParams2Learn is not None)
                and (bsfLosses["rmse_constr"] > 0.0)
                and (self.losses["rmse_constr"] != bsfLosses["rmse_constr"])
            ):
                # print(
                #     f"!!! b) END Unequal rmse_constr: {self.losses['rmse_constr']} vs. {bsfLosses['rmse_constr']}"
                # )
                self.losses["rmse_constr"] = bsfLosses["rmse_constr"]
        # ---
        return originNonDominatedImprovement, nonDominatedImprovements

    def calculateCheckPoints(self):
        """
        Calculates output values on a given set of check points.
        :return:
        """
        self.set_forwardpass_masks()
        data = [
            [1.0, 1.0, 0.5],
            [1.0, 5.0, 0.833],
            [1.0, 9.0, 0.9],
            [5.0, 1.0, 0.833],
            [5.0, 5.0, 2.5],
            [5.0, 9.0, 3.214],
            [9.0, 1.0, 0.9],
            [9.0, 5.0, 3.214],
            [9.0, 9.0, 4.5],
        ]
        x_input = np.array([[el[0], el[1]] for el in data]).reshape((len(data), 2))
        y_output = np.array([el[2] for el in data]).reshape((len(data), 1))
        y_hat = self.model_feed_forward(x_input)
        x0 = np.array([el[0] for el in data]).reshape((len(data), 1))
        x1 = np.array([el[1] for el in data]).reshape((len(data), 1))
        y_formula = self.evaluateAnalyticFormula(x0, x1)
        return x_input, y_output, y_hat, y_formula

    def updateActiveNodes(self):
        # ----------------------------------------------------------------------------------------------
        # --- Backward pass
        # --- If an active node does not serve as an input to any active node in the subsequent layer then
        # ---   1. the node is inactivated
        # ---   2. all its input weights and masks are set to 0
        # ----------------------------------------------------------------------------------------------
        for l in reversed(list(range(1, len(self.nnLayers)))):  # --- output layer
            currentNodes = np.zeros(
                shape=[len(self.nnLayers[l - 1].units), 1]
            )  # --- current layer
            for outputNode in self.nnLayers[l].units:  # --- nodes in output layer
                outputNodesInputActivity = np.array(self.activeNodeCoordinates[l - 1])
                outputNodesInputActivity = outputNodesInputActivity.reshape(
                    currentNodes.shape
                )
                for tfvar in outputNode.weights:  # --- for all nodes in output layer
                    if " b" not in tfvar.v.name:  # --- weights, NOT bias
                        # --- output layer input-nodes activity check
                        newVals = np.multiply(outputNodesInputActivity, tfvar.v.numpy())
                        tfvar.set_value_mask(newVals)
                        # ---
                        currentNodes = np.add(currentNodes, tfvar.mask)
            # --- current layer activity check
            self.activeNodeCoordinates[l - 1] = list(
                np.logical_and(
                    self.activeNodeCoordinates[l - 1], currentNodes.flatten()
                )
            )
            for n, node in enumerate(self.nnLayers[l - 1].units):
                if not self.activeNodeCoordinates[l - 1][n]:
                    for tfvar in node.weights:
                        tfvar.set_allzero_value_mask()
        # ----------------------------------------------------------------------------------------------
        # --- Forward pass
        # --- If an active node has no active input then
        # ---   1. the node is inactivated
        # ---   2. all its input weights and masks are set to 0
        # ----------------------------------------------------------------------------------------------
        inputNodes = np.ones(shape=[self.nnLayers[0].inSize, 1])
        for l, layer in enumerate(self.nnLayers):
            node: NNUnit
            if l > 0:
                inputNodes = self.activeNodeCoordinates[l - 1]
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:
                    hasActiveWeights = False
                    for tfvar in node.weights:
                        if " b" not in tfvar.v.name:  # --- weights, NOT bias
                            if tfvar.hasAnyActiveInput(
                                inputLayerActiveNodes=inputNodes
                            ):  # --- some triple of [weight, mask, activeNode] must be True
                                hasActiveWeights = True
                            else:
                                hasActiveWeights = False
                    if not hasActiveWeights:
                        self.activeNodeCoordinates[l][
                            n
                        ] = False  # --- all weights of this active node are inactive
                        for tfvar in node.weights:
                            tfvar.set_allzero_value_mask()
        # ----------------------------------------------------------------------------------------------
        # --- Calculate the number of active nodes and active weights
        # ----------------------------------------------------------------------------------------------
        self.nbOfActiveNodes = 0
        self.nbOfActiveWeights = 0
        # ---
        for l, layer in enumerate(self.nnLayers):
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n] and not ("UnitIdent1" in node.name):
                    self.nbOfActiveNodes += 1.0
                    # ---
                    for tfvar in node.weights:
                        # if ' b' not in tfvar.v.name:  # --- Original: weights, NOT bias
                        if True:
                            nbWeights = np.sum(tfvar.mask)
                            self.nbOfActiveWeights += nbWeights
        # ---
        if self.nbOfActiveNodes < SRConfig.minNbOfActiveNodes:
            self.nbOfActiveNodes = SRConfig.maxFrontNb
        self.complexity = (
            self.nbOfActiveNodes + self.nbOfActiveWeights / 10000.0
        )  # --- v03
        # self.complexity = self.nbOfActiveWeights          # --- v04
        # ---
        # self.losses['nbOfActiveNodes'] = self.nbOfActiveNodes     # --- Original
        # self.losses['nbOfActiveWeights'] = self.nbOfActiveWeights
        # self.nbOfActiveNodes = self.nbOfActiveNodes + self.nbOfActiveWeights/10000.0
        self.losses["nbOfActiveNodes"] = self.nbOfActiveNodes
        self.losses["complexity"] = self.complexity

    def activateAllInactiveNodes(self):
        # ----------------------------------------------------------------------------------------------
        # --- Forward pass through all hidden layers; output layer node is always active:
        # ---   1. activates all inactive nodes in the current layer
        # ---   2. randomly initializes all of its input weights
        # ----------------------------------------------------------------------------------------------
        for l in range(len(self.nnLayers) - 1):
            for n, node in enumerate(self.nnLayers[l].units):
                if (not self.activeNodeCoordinates[l][n]) and (
                    SRConfig.r.random() < SRConfig.perturbation
                ):
                    # --- activate the node
                    self.activeNodeCoordinates[l][n] = True
                    # --- and set randomly all of its weights
                    if "UnitIdent1" in node.name:
                        node.reset_weights_and_bias()
                    else:
                        self.nbOfActiveNodes += 1
                        for w in node.weights:
                            inputLayerActiveNodes: list[bool]
                            if " b" in w.v.name:  # --- bias
                                inputLayerActiveNodes = [True]
                            else:
                                inputLayerActiveNodes = self.nnLayers[l].inSize * [True]
                            w.setRandomWeights(
                                layerId=l,
                                inputLayerActiveNodes=inputLayerActiveNodes,
                                perturbation=False,
                            )  # --- ORIGINAL
                            # w.set_zerovalues_onemask()                      # --- TODO: inactive weights are set to zero

    def printSubtopology(self, printSimplified=False, printCheckPoints=False):
        self.printActiveNodes()
        self.printPerformanceMetrics()
        self.getAnalyticFormula(
            simplify=self.nbOfActiveNodes <= SRConfig.maxNbOfActiveNodesToSympyfy
        )
        self.printAnalyticFormula(printSimplified=printSimplified)
        if printCheckPoints:
            self.printCheckPoints()

    def writeSubtopologyToFile(self, fileName: str, savePickle=False):
        f = open(SRConfig.outputNamePrefix + "_" + fileName + ".txt", "w")
        s = self.printPerformanceMetrics()
        f.write(s)
        # ---
        f.write(f"\n\nBackprop iterations: {self.backpropCount}")
        # --- Write expression in Python
        self.getAnalyticFormula(
            simplify=self.nbOfActiveNodes <= SRConfig.maxNbOfActiveNodesToSympyfy
        )
        outputNode: NNUnit = self.nnLayers[-1].units[0]
        s = f"\n\nAnalytic formula in Python: {outputNode.string_analytic_expression}"
        if outputNode.symbolic_expression:
            s += f"\n\nSimplified analytic formula: {outputNode.symbolic_expression.simpleExpr}"
        # --- Write expression in Matlab
        exprMatlab = outputNode.string_analytic_expression
        exprMatlab = exprMatlab.replace("*", ".*")
        exprMatlab = exprMatlab.replace("/", "./")
        exprMatlab = exprMatlab.replace("np.", "")
        s += f"\n\nAnalytic formula in Matlab: {exprMatlab}"
        if outputNode.symbolic_expression:
            exprMatlab = str(outputNode.symbolic_expression.simpleExpr)
            exprMatlab = exprMatlab.replace("*", ".*")
            exprMatlab = exprMatlab.replace("/", "./")
            exprMatlab = exprMatlab.replace("np.", "")
            s += f"\n\nSimplified analytic formula in Matlab: {exprMatlab}"
        # ---
        f.write(s)
        f.close()
        # --- Save to pickle
        if savePickle:
            with open(
                SRConfig.outputNamePrefix + "_" + fileName + ".pickle", "wb"
            ) as outfile:
                pickle.dump(self, outfile)

    def printActiveNodes(self):
        print(f"\nActive nodes: {self.complexity}")
        for l, layer in enumerate(self.nnLayers):
            print(
                f"{l}. layer: {len(layer.units)} nodes / {np.sum(self.activeNodeCoordinates[l])} active nodes"
            )
            for n, node in enumerate(layer.units):
                if self.activeNodeCoordinates[l][n]:
                    print(f"\tACTIVE node[{l}][{n}]: {node.name}", end="")
                    for tfvar in node.weights:
                        print(f"\n\t\t{str(tfvar.v.numpy().flatten())}", end="")
                    print("")
                else:
                    print(f"\tINACTIVE node[{l}][{n}]: {node.name}", end="")
                    for tfvar in node.weights:
                        print(f"\n\t\t{str(tfvar.v.numpy().flatten())}", end="")
                    print("")

    def printPerformanceMetrics(self):
        s = f"Performance metrics:"
        # print(f'\nPerformance metrics:')
        s += f"\n\tnb. of active nodes: {int(self.nbOfActiveNodes)}"
        s += f"\n\tnb. of active weights: {int(self.nbOfActiveWeights)}"
        s += f"\n\tcomplexity: {self.complexity}"
        # print(f'\tnb. of active nodes: {self.nbOfActiveNodes}')
        for key in self.losses:
            if "rmse" in key:
                s += f"\n\t{key}: {self.losses[key]:2.15f}"
        s += f"\n\traw constraint terms:"
        for key in self.raw_constraint_terms:
            s += f"\n\t\t{key}: {self.raw_constraint_terms[key]:2.15f}"
        s += f"\n\tweighted constraint terms:"
        for key in self.constraint_terms:
            s += f"\n\t\t{key}: {self.constraint_terms[key]:2.15f}"
        print("\n" + s)
        return s
