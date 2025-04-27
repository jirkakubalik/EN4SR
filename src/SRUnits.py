import copy

import numpy as np
import warnings

warnings.filterwarnings("ignore")
import tensorflow as tf
import functools
import sympy as sp
import abc
from abc import abstractmethod
from call_function_with_timeout import SetTimeout

import SRConfig
from SRData import SRData

x0, x1, x2, x3, x4 = sp.symbols("x0 x1 x2 x3 x4")


def init_np_seed(seed):
    np.random.seed(seed)


class MyTFVariable:
    def __init__(self, maxArity, weights, mask: np.array, name):
        self.v = tf.Variable(weights, name=name)
        self.mask = mask
        self.mask_orig = copy.deepcopy(mask)
        self.maxArity = maxArity
        self.a: tf.Tensor
        return

    def set_allzero_value_mask(self):
        self.v.assign(np.zeros(tf.shape(self.v)))
        self.mask = np.zeros(self.mask.shape)

    def set_allzero_value(self):
        self.v.assign(np.zeros(tf.shape(self.v)))

    def set_zerovalues_onemask(self):
        self.v.assign(np.zeros(tf.shape(self.v)))
        self.mask = np.ones(self.mask.shape)

    def set_value(self, newValues):
        self.v.assign(newValues)

    def set_value_mask(self, newValues):
        # newValues = self.adjustArity(newValues)
        self.v.assign(newValues)
        m = np.array(
            np.abs(newValues) >= SRConfig.activeWeightThreshold, dtype=int
        ).reshape(len(newValues), 1)
        self.mask = m

    def set_allzero_mask(self):
        self.mask = np.zeros(self.mask.shape)

    def infer_mask(self):
        m = np.array(
            np.abs(self.v.numpy()) >= SRConfig.activeWeightThreshold, dtype=int
        ).reshape(self.mask.shape)
        self.mask = m

    def set_mask(self, mask):
        self.mask = np.array(mask)

    # def set_mask_value(self, mask):
    #     self.mask = np.array(mask)
    #     newValues = np.multiply(self.v.numpy(), self.mask)
    #     self.v.assign(newValues)

    def hasAnyNonzeroWeight(self):
        res = np.any(self.v.numpy())
        return res

    def hasAnyNonzeroMask(self):
        res = np.any(self.mask)
        return res

    def hasAnyActiveInput(self, inputLayerActiveNodes: list[bool]):
        """
        Checks if the variable has any non-zero pair of weight-mask
        :return: True, if the node has some active input. False, otherwise.
        """
        res = np.multiply(self.v.numpy(), self.mask)
        res = np.multiply(res, inputLayerActiveNodes)
        res = np.any(res)
        return res

    def setRandomWeights(
        self,
        layerId: int,
        inputLayerActiveNodes: list[bool],
        perturbation: bool = False,
    ):
        """
        Generates an array of weights using only a random subset of inputLayerActiveNodes.
        Assigns the new values and sets the mask accordingly.
        """
        origMask = np.array(inputLayerActiveNodes).reshape(
            (len(inputLayerActiveNodes), 1)
        )
        origMaskNonZero = origMask.nonzero()
        if len(origMaskNonZero[0]) == 0:
            self.set_allzero_value_mask()
            return
        # ---
        tmp = SRConfig.r.choice(origMaskNonZero[0], 1)
        if len(tmp) == 0:
            self.set_allzero_value_mask()
            print(f"\n\n!!! empty origMaskNonZero[0]: {origMaskNonZero} !!!\n")
            return
        # ---
        nonzero = tmp[0]
        finalMask = np.array(
            [int(SRConfig.r.random() < 1.0) if el == 1 else 0 for el in origMask]
        ).reshape(
            origMask.shape
        )  # --- ORIGINAL: 0.5. TODO: 1.0
        if sum(finalMask) == 0:
            finalMask[nonzero] = 1
        # values = SRConfig.r.uniform(low=-SRConfig.weight_sigma, high=SRConfig.weight_sigma, size=self.v.shape)    # --- ORIGINAL: uniform distribution
        # ---
        if perturbation:
            values = NNUnit.generate_normal_distribution(
                layerId=layerId,
                mu=0.0,
                sigma=SRConfig.weight_sigma / 10.0,
                size=self.v.shape,
                biasWeight=(" b" in self.v.name),
            )  # --- TODO: normal distribution respecting the minimum init vale
        else:
            values = NNUnit.generate_normal_distribution(
                layerId=layerId,
                mu=0.0,
                sigma=SRConfig.weight_sigma,
                size=self.v.shape,
                biasWeight=(" b" in self.v.name),
            )  # --- TODO: normal distribution respecting the minimum init vale
        # ---
        values = np.multiply(values, finalMask)
        self.set_value_mask(values)

    def adjustArity(self, mask):
        nonZeroMask = mask.nonzero()
        while len(nonZeroMask[0]) > self.maxArity:
            toZero = SRConfig.r.choice(nonZeroMask[0], 1)[0]
            mask[toZero] = 0
            nonZeroMask = mask.nonzero()
        return mask


class SymbolicExpression:

    def __init__(self, strExpr, simplify: bool = False):
        self.strExpr = strExpr.replace("np.", "")
        self.symExpr = "0"
        self.simpleExpr = "0"
        func_with_timeout = SetTimeout(
            self.simplifyFormula, timeout=SRConfig.simplificationTimeout
        )
        if self.strExpr != "":
            # self.symExpr = sp.sympify(self.strExpr)
            # self.simpleExpr = sp.expand(self.symExpr)
            # self.simpleExpr = sp.simplify(self.symExpr)
            is_done: bool = False
            if simplify:
                is_done, is_timeout, erro_message, results = func_with_timeout(
                    self.strExpr
                )
            if is_done:
                self.simpleExpr = results
            else:
                self.simpleExpr = "NA"
        else:
            self.symExpr = sp.sympify("0")
            is_done: bool = False
            if simplify:
                is_done, is_timeout, erro_message, results = func_with_timeout(
                    self.strExpr
                )
            if is_done:
                self.simpleExpr = results
            else:
                self.simpleExpr = "NA"

    def simplifyFormula(self, formula):
        self.symExpr = sp.sympify(formula)
        simpleFormula = sp.simplify(self.symExpr)
        return simpleFormula

    def printExpr(self):
        print(f"{self.symExpr}")

    def printSimpleExpr(self):
        print(f"{self.simpleExpr}")


class NNUnit(abc.ABC):
    """
    Abstract class for NNUnits with different activation functions.
    """

    # name: str
    # weights: list[MyTFVariable]  # --- list of MyTFVariables
    # a: tf.Tensor                 # --- node's output
    # string_analytic_expression: str
    # symbolic_expression = None
    mu = 0.0
    # weight_sigma = SRConfig.weight_sigma
    # weight_max_init = SRConfig.weight_max_init
    # grad_threshold = 0.1

    @abstractmethod
    def __init__(self):
        self.name: str = ""
        self.proxyName: str = ""
        self.isVariable: bool = (
            False  # --- True only for UnitIdent1 units representing variables
        )
        self.learnable: bool = (
            True  # --- Default is that the unit's weight are learnable
        )
        self.weights: list[MyTFVariable] = []  # --- list of MyTFVariables
        self.maxArity: int = 3  # --- Maximum number of active input weights
        self.a: tf.Tensor = None  # --- node's output
        self.string_analytic_expression: str = ""
        self.symbolic_expression: SymbolicExpression = None

    @abstractmethod
    def define_weight_and_bias_vars(self, layerId: int, inSize: int):
        pass

    def reset_weights_and_bias(self):
        pass

    @abstractmethod
    def forward_pass(self, x_input):
        pass

    # @abstractmethod
    # def get_analytic_expression(self, layers):
    #     '''
    #     Generates string representation of the node.
    #     '''
    #     pass

    @abstractmethod
    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        """
        Generates string representation of the node using a single forward pass procedure.
        """
        pass

    def get_name(self):
        """ """
        print(f"Name: {self.name}")
        return self.name

    def node_info(self):
        print(f"Name: {self.name}")

    @staticmethod
    def generate_normal_distribution(
        layerId: int, mu: float, sigma: float, size, biasWeight: bool = False
    ):
        if biasWeight:
            v = SRConfig.r.normal(loc=mu, scale=sigma, size=size)
            v[0] = 0.0
        elif (
            (layerId == 0)
            and (SRConfig.r.random() < SRConfig.weight_init_one)
            and (not biasWeight)
        ):
            # -----------------------------------------------
            # --- +/- 1
            # -----------------------------------------------
            v = SRConfig.r.normal(loc=mu, scale=sigma, size=size)
            # --- negative values
            b = np.where(v < 0.0)[0]
            v[b] = -1.0
            # --- positive values
            b = np.where(v > 0.0)[0]
            v[b] = 1.0
        else:
            # -----------------------------------------------
            # --- Normal distribution random values
            # -----------------------------------------------
            v = SRConfig.r.normal(loc=mu, scale=sigma, size=size)
            # --- negative values
            b = np.where(v < 0.0)[0]
            v[b] = -SRConfig.weight_min_init + v[b]
            # --- positive values
            b = np.where(v > 0.0)[0]
            v[b] = SRConfig.weight_min_init + v[b]
        return v

    @staticmethod
    def create_tf_variable(
        shape, name, layerId: int, maxArity=1000, weights=None, mask=None, offset=0.0
    ):
        if weights is None:
            # --- ORIGINAL
            # if SRConfig.r.random() < 0.5:
            # weights = offset + np.random.uniform(low=-SRConfig.weight_sigma, high=SRConfig.weight_sigma, size=shape)
            # else:
            # weights = -offset + np.random.uniform(low=-SRConfig.weight_sigma, high=SRConfig.weight_sigma, size=shape)
            # --- TODO: normal distribution respecting the minimum value
            weights = NNUnit.generate_normal_distribution(
                layerId=layerId,
                mu=0.0,
                sigma=SRConfig.weight_sigma,
                size=shape,
                biasWeight=(" b" in name),
            )
        # ---
        if mask is None:
            mask = np.ones(shape=shape)
        weights = np.multiply(weights, mask)
        tf_v = MyTFVariable(maxArity=maxArity, weights=weights, mask=mask, name=name)
        return tf_v


class UnitMultiply(NNUnit):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2  # --- 4
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, layerId: int, inSize):
        """
        Called in NNLayer.add_unit()
        :param inSize:
        :return:
        """
        self.W1: MyTFVariable = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.W2: MyTFVariable = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W2",
            layerId=layerId,
        )
        self.weights = [self.W1, self.W2]
        return [self.W1, self.W2], None

    def forward_pass(self, x_input):
        if not (np.any(self.W1.v.numpy())):
            self.W2.v.assign(np.zeros(shape=self.W2.v.shape.as_list()))
        elif not (np.any(self.W2.v.numpy())):
            self.W1.v.assign(np.zeros(shape=self.W1.v.shape.as_list()))
        # ---
        z1 = tf.matmul(x_input, self.W1.v)
        z2 = tf.matmul(x_input, self.W2.v)
        self.a = tf.multiply(z1, z2)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W1.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        op2 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W2.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            op1 = f"({op1})"
        if op2:
            op2 = functools.reduce(lambda a, b: f"{a}+{b}", op2)
            op2 = f"({op2})"
        # ---
        if op1 and op2:
            self.string_analytic_expression = f"({op1} * {op2})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitDivide(NNUnit):

    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2  # --- 4
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # # --- W and b for two inner potentials
        # self.W1 = NNUnit.create_tf_variable(shape=(inSize, 1), name=self.name+' '+'W1')
        # self.b1 = NNUnit.create_tf_variable(shape=(1, 1), name=self.name+' '+'b1')
        # self.W2 = NNUnit.create_tf_variable(shape=(inSize, 1), name=self.name+' '+'W2')
        # self.b2 = NNUnit.create_tf_variable(shape=(1, 1), name=self.name+' '+'b2')
        # self.weights = [self.W1, self.b1, self.W2, self.b2]
        # return [self.W1, self.b1, self.W2, self.b2], None
        # --- W and b for two inner potentials
        self.W1 = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W1",
            layerId=layerId,
        )
        self.W2 = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W2",
            layerId=layerId,
        )
        self.weights = [self.W1, self.W2]
        return [self.W1, self.W2], None

    def forward_pass(self, x_input):
        if not (np.any(self.W1.v.numpy())):
            self.W2.v.assign(np.zeros(shape=self.W2.v.shape.as_list()))
        elif not (np.any(self.W2.v.numpy())):
            self.W1.v.assign(np.zeros(shape=self.W1.v.shape.as_list()))
        # ---
        div_thresh = 1.0e-4
        z1 = tf.matmul(x_input, self.W1.v)
        z2 = tf.matmul(x_input, self.W2.v)
        mask = tf.cast(z2 > div_thresh, dtype=tf.float64)
        # mask = tf.cast(tf.where(z2 > div_thresh, 1.0, -1.0), dtype=tf.float64)
        div = tf.math.reciprocal(tf.abs(z2) + 1e-15)
        self.a = tf.multiply(z1, div)
        self.a = tf.multiply(self.a, mask)
        return self.a, z2

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        # if(currLayer == 0):
        #     in_op = [f'x{i}' for i in range(SRData.x_data.shape[1])]
        # else:
        #     in_op = [u.string_analytic_expression for u in layers[currLayer-1].units]
        # op1 = ['{:2.15}*{}'.format(w, i) for w,i in zip(self.W1.v.numpy().flatten().tolist(), in_op) if (np.abs(w) > SRConfig.weight_threshold) and (i != '')]
        # op2 = ['{:2.15}*{}'.format(w, i) for w,i in zip(self.W2.v.numpy().flatten().tolist(), in_op) if (np.abs(w) > SRConfig.weight_threshold) and (i != '')]
        # if np.abs(self.b1.v.numpy().flatten()[0]) > SRConfig.weight_threshold:
        #     op1.append('{:2.15}'.format(self.b1.v.numpy().flatten()[0]))
        # if np.abs(self.b2.v.numpy().flatten()[0]) > SRConfig.weight_threshold:
        #     op2.append('{:2.15}'.format(self.b2.v.numpy().flatten()[0]))
        # # ---
        # if op1:
        #     op1 = functools.reduce(lambda a, b: f'{a}+{b}', op1)
        #     op1 = f'({op1})'
        # if op2:
        #     op2 = functools.reduce(lambda a, b: f'{a}+{b}', op2)
        #     op2 = f'({op2})'
        # # ---
        # if op1 and op2:
        #     self.string_analytic_expression = f'({op1} / {op2})'
        # else:
        #     self.string_analytic_expression = ''
        # # ---
        # self.symbolic_expression = SymbolicExpression(self.string_analytic_expression)
        # return self.string_analytic_expression
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W1.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        op2 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W2.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            op1 = f"({op1})"
        if op2:
            op2 = functools.reduce(lambda a, b: f"{a}+{b}", op2)
            op2 = f"({op2})"
        # ---
        if op1 and op2:
            self.string_analytic_expression = f"({op1} / {op2})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitSin(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1  # ---
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # self.W = NNUnit.create_tf_variable(shape=(inSize, 1), maxArity=self.maxArity, name=self.name+' '+'W')
        # self.b = NNUnit.create_tf_variable(shape=(1, 1), maxArity=1, name=self.name+' '+'b')
        # self.weights = [self.W, self.b]
        # return [self.W, self.b], None
        # ---
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        # if not(np.any(self.W.v)):
        #     self.b.v.assign(np.zeros(self.b.v.shape))
        # # ---
        # z = tf.add(tf.matmul(x_input, self.W.v), self.b.v)
        # a = tf.sin(z)
        # return a, None
        # ---
        # if np.sum(self.W.mask) == 0.0:
        #     self.b.set_allzero_mask()
        #     a = tf.zeros([x_input.shape[0], 1], tf.float64)
        # else:
        #     z = tf.add(tf.matmul(x_input, self.W.v), self.b.v)
        #     a = tf.sin(z)
        # ---
        if np.sum(self.W.mask) == 0.0:
            self.a = tf.zeros([x_input.shape[0], 1], tf.float64)
        else:
            z = tf.matmul(x_input, self.W.v)
            self.a = tf.sin(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        # if np.abs(self.b.v.numpy().flatten()[0]) > SRConfig.activeWeightThreshold:
        #     op1.append('{:2.15}'.format(round(self.b.v.numpy().flatten()[0], SRConfig.precisionDigits)))
        # if op1:
        #     op1 = functools.reduce(lambda a, b: f'{a}+{b}', op1)
        # ---
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
            self.string_analytic_expression = f"np.sin({op1})"
        else:
            self.string_analytic_expression = ""
        # ---
        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitCos(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2  # --- TODO: remove 'b'
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        # if not(np.any(self.W.v)):
        #     self.b.v.assign(np.zeros(self.b.v.shape))
        if (np.sum(self.W.mask) == 0.0) and (not (np.any(np.abs(self.W.v) > 0.0))):
            self.b.set_allzero_mask()
            self.a = tf.zeros([x_input.shape[0], 1], tf.float64)
        else:
            z = tf.add(tf.matmul(x_input, self.W.v), self.b.v)
            self.a = tf.cos(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if self.b.mask:
            op1.append(
                "{:2.15}".format(
                    round(self.b.v.numpy().flatten()[0], SRConfig.precisionDigits)
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        else:
            op1 = "0"
        # ---
        if op1:
            self.string_analytic_expression = f"np.cos({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitTanh(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1  # --- TODO: remove 'b'
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        # if not(np.any(self.W.v)):
        #     self.b.v.assign(np.zeros(self.b.v.shape))
        # ---
        z = tf.matmul(x_input, self.W.v)
        self.a = tf.math.tanh(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"np.tanh({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitArcTan(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1  # --- TODO: remove 'b'
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        z = tf.matmul(x_input, self.W.v)
        self.a = tf.math.atan(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"np.arctan({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitSquare(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2  # --- TODO: remove 'b'
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        if not (np.any(self.W.v)):
            self.b.v.assign(np.zeros(self.b.v.shape))
        # ---
        z = tf.add(tf.matmul(x_input, self.W.v), self.b.v)
        self.a = tf.square(z)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if np.abs(self.b.v.numpy().flatten()[0]) > SRConfig.activeWeightThreshold:
            op1.append(
                "{:2.15}".format(
                    round(self.b.v.numpy().flatten()[0], SRConfig.precisionDigits)
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**2"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitSqrt(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1  # --- TODO: remove 'b'
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        # self.b = NNUnit.create_tf_variable(shape=(1, 1), maxArity=1, name=self.name+' '+'b')
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        # if not(np.any(self.W.v)):
        #     self.b.v.assign(np.zeros(self.b.v.shape))
        # ---
        # z = tf.add(tf.matmul(x_input, self.W.v), self.b.v)
        # a = tf.sqrt(z)
        # ---
        sqrt_thresh = 1.0e-6
        z = tf.matmul(x_input, self.W.v)
        mask = tf.cast(z > sqrt_thresh, dtype=tf.float64)
        self.a = tf.sqrt(tf.abs(z))
        self.a = tf.multiply(self.a, mask)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**0.5"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitCube(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        self.weights = [self.W]
        return [self.W], None

    def forward_pass(self, x_input):
        # if not(np.any(self.W.v)):
        #     self.b.v.assign(np.zeros(self.b.v.shape))
        # ---
        z = tf.matmul(x_input, self.W.v)
        self.a = tf.pow(z, 3.0)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})**3"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitIdent(NNUnit):
    def __init__(self):
        super().__init__()
        self.name = self.__class__.__name__
        self.nbOfVars = 2
        self.maxArity: int = 100  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            layerId=layerId,
        )
        self.b = NNUnit.create_tf_variable(
            shape=(1, 1), maxArity=1, name=self.name + " " + "b", layerId=-1
        )
        self.weights = [self.W, self.b]
        return [self.W, self.b], None

    def forward_pass(self, x_input):
        self.a = tf.add(tf.matmul(x_input, self.W.v), self.b.v)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{:2.15}*{}".format(round(w, SRConfig.precisionDigits), i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if np.abs(self.b.v.numpy().flatten()[0]) > SRConfig.activeWeightThreshold:
            op1.append(
                "{:2.15}".format(
                    round(self.b.v.numpy().flatten()[0], SRConfig.precisionDigits)
                )
            )
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"({op1})"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


class UnitIdent1(NNUnit):
    def __init__(self, inId=0):
        super().__init__()
        self.inId = inId
        # if self.inId < SRData.x_data.shape[1]:  # --- points directly to a variable
        #     self.isVariable = True
        self.learnable = False
        self.name = self.__class__.__name__
        self.nbOfVars = 1
        self.maxArity: int = 1  # --- Maximum number of active input weights

    def define_weight_and_bias_vars(self, inSize, layerId: int):
        # --- weights
        w = np.zeros(shape=(inSize, 1))
        w[self.inId] = 1.0
        # --- mask
        m = np.zeros(shape=(inSize, 1))
        m[self.inId] = 1.0
        self.W = NNUnit.create_tf_variable(
            shape=(inSize, 1),
            maxArity=self.maxArity,
            name=self.name + " " + "W",
            weights=w,
            mask=m,
            layerId=layerId,
        )
        self.weights = [self.W]
        return None, [self.W]  # --- W and b are not being learned

    def reset_weights_and_bias(self):
        # --- weights
        w = np.zeros(shape=self.W.v.shape)
        w[self.inId] = 1.0
        self.W.set_value_mask(w)

    def forward_pass(self, x_input):
        self.a = tf.matmul(x_input, self.W.v)
        return self.a, None

    def get_analytic_expression_forwardpass(
        self, currLayer, layers, simplify: bool = False
    ):
        if currLayer == 0:
            in_op = [f"x{i}" for i in range(SRData.x_data.shape[1])]
        else:
            in_op = [u.string_analytic_expression for u in layers[currLayer - 1].units]
        op1 = [
            "{}".format(i)
            for w, i in zip(self.W.v.numpy().flatten().tolist(), in_op)
            if (np.abs(w) > SRConfig.activeWeightThreshold) and (i != "")
        ]
        if op1:
            op1 = functools.reduce(lambda a, b: f"{a}+{b}", op1)
        # ---
        if op1:
            self.string_analytic_expression = f"{op1}"
        else:
            self.string_analytic_expression = ""

        self.symbolic_expression = SymbolicExpression(
            self.string_analytic_expression, simplify=simplify
        )
        return self.string_analytic_expression


# ================= EXTRAS =============================================================================================
