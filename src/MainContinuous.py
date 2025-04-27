import sys
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np

import SRConfig
from SRConstraints import SRConstraints
import SRUnits
from SRData import SRData
from ReadParams import read_params
from EN4SRSolverContinuous import EN4SRSolverContinuous


#----------------------------------------------------------------------
if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    argv_str = read_params(sys.argv[1:])
    SRConfig.r = np.random.default_rng(seed=1)
    np.random.seed(SRConfig.seed)
    SRUnits.init_np_seed(SRConfig.seed)
    SRConfig.saveConfigurationToFile()

    # --- Init data
    SRData.collect_dataset_sequence(baseName=SRConfig.train_data)
    if not SRData.datasetNames:
        sys.exit('Error: No data set available.')
    SRData.init_data_continuous(test_file=SRConfig.test_data)

    # --- Init constraints
    SRConstraints.load_constraints(path=SRConfig.constraints_file)

    # --- Init solver
    solver = EN4SRSolverContinuous()

    # --- Run the solver
    solver.run()
