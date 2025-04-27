import sys
import SRConfig
import pickle
import numpy as np
from SubTopology import SubTopology

from SRUnits import UnitMultiply, UnitDivide, \
    UnitSin, UnitCos, UnitTanh, UnitArcTan, \
    UnitSquare, UnitSqrt, UnitCube, \
    UnitIdent, UnitIdent1


def testPickle(population):
    # #--- Pickle to serialize and deserialize
    # pickled_model = pickle.dumps(self.population[0])
    # reconstructed: SubTopology = pickle.loads(pickled_model)
    # print(f'Unpickled {reconstructed.losses}')

    fileName = SRConfig.outputNamePrefix + '_' + f'individual_{1}.pickle'
    with open(fileName, 'wb') as outfile:
        pickle.dump(population[0], outfile)

    with open(fileName, "rb") as infile:
        reconstructed: SubTopology = pickle.load(infile)
        print(f'Unpickled (original) rmse_valid={reconstructed.losses["rmse_valid"]}')
        rmse_valid = reconstructed.calculateValidationRMSE()
        print(f'Unpickled (new) rmse_valid={rmse_valid}')

    n = 20
    x0 = np.linspace(0.01, 10, n)
    x1 = np.linspace(0.01, 10, n)
    x0, x1 = np.meshgrid(x0, x1)
    x0 = x0.reshape((n*n,1))
    x1 = x1.reshape((n*n,1))
    test_data = np.zeros((n*n,2))
    test_data[:,0] = x0[:,0]
    test_data[:,1] = x1[:,0]
    y_gt = x0*x1 / (x0+x1)
    rmse_test, y_hat = reconstructed.calculateRMSE(test_data, y_gt)
    print(f'rmse_test = {rmse_test}')


class NNTopology():
    unit_types = {'UnitMultiply': UnitMultiply,
                  'UnitDivide': UnitDivide,
                  'UnitSin': UnitSin, 'UnitCos': UnitCos, 'UnitTanh': UnitTanh, 'UnitArcTan': UnitArcTan,
                  'UnitSquare': UnitSquare, 'UnitSqrt': UnitSqrt, 'UnitCube': UnitCube,
                  'UnitIdent': UnitIdent, 'UnitIdent1': UnitIdent1}

    def __init__(self):
        pass


    def get_topology_definitions(topology_file: str=''):
        """
        Reads possibly multiple topology definitions from a file.
        :param topology_file:
        :return: lists of layer_defs and identities
        """
        all_layer_defs = []
        all_identities = []
        nn_layer_defs = []
        nn_identities = []
        print(f'\nTopology: {topology_file}')
        if topology_file is not None:
            with open('topologies/' + topology_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if 'topology' in line:
                        if nn_layer_defs:
                            all_layer_defs.append(nn_layer_defs)
                            all_identities.append(nn_identities)
                        # ---
                        nn_layer_defs = []
                        nn_identities = []
                        continue
                    print(f'\t{line}')
                    if line.startswith('Unit'):
                        l = {}  # --- new layer
                        line = line.split(',')
                        for u in line:
                            u = u.split(':')
                            l[NNTopology.unit_types[u[0].strip()]] = int(u[1])
                        nn_layer_defs.append(l)
                    elif line.startswith('identities'):
                        line = line.split(':')
                        if line[1].strip().lower() == 'true':
                            nn_identities.append(True)
                        elif line[1].strip().lower() == 'false':
                            nn_identities.append(False)
                        else:
                            print('! Wrong identities value.')
                            sys.exit(3)
                    else:
                        print('! Wrong topology file content.')
                        sys.exit(3)
                if nn_layer_defs:
                    all_layer_defs.append(nn_layer_defs)
                    all_identities.append(nn_identities)
        return all_layer_defs, all_identities


# --------------------------------------------------------------------------
# nnTopology = NNTopology()