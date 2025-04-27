import sys, getopt
import SRConfig

def read_params(argv):
    argv_str = 'arguments: '
    try:
        opts, args = getopt.getopt(argv,"r:s:o:t:c:l:w:",['seed=','outfolder=','savestep=',
                                                          'constraints=','topology=',
                                                          'maxbackprops=', 'initpopsize=', 'popsize=', 'genpopsize=', 'eagens=',
                                                          'pruning=', 'perturb=', 'perturbstep=',
                                                          'histusage=', 'inheritfirst=',
                                                          'train_data=','valid_data=','test_data=','train_val=',
                                                          'scale=','rmsescale=','minibatch=',
                                                          'max_weights=','w_thr=', 'active_units=',
                                                          'reg_lambda=','learning_rate=','learning_rate_noreg=','adaptive=','best_ext=',
                                                          'stack_size=', 'update_step=', 'sng2rmse=', 'reg2rmse=', 'constr2rmse=',
                                                          'iters=','iters_A=','reg_iters=','noreg_iters=','noconstr_iters='])
        for opt, arg in opts:
            argv_str = argv_str + '\n' + opt.replace('-', '') + '=' + arg
            if opt == '-h':
                print('SRMain.py -s <seed> -o <outfolder> -r <reg_lambda> -t <topology> -c <constraints> -l <learning_rate>'
                      '-w <w_thr>'
                      '--learning_rate_noreg <learning_rate_noreg>'
                      '--stack_size <stack_size>', '--update_step <update_step>', '--sng2rmse <sng2rmse>', '--reg2rmse <reg2rmse>', '--constr2rmse <constr2rmse>',
                      '--adaptive <adaptive>', '--best_ext <best_ext>',
                      '--max_weights <max_weights> --scale <scaleCoeff>'
                      '--iters <total_iters> --iters_A <iters_A> --reg_iters <reg_iters> --noreg_iters <noreg_iters> --noconstr_iters <noconstr_iters>'
                      '--train_val --train_data --valid_data --test_data --minibatch')
                sys.exit(1)
            elif opt in ("-s", "--seed"):
                SRConfig.seed = int(arg)
            elif opt in ("-o", "--outfolder"):
                outfolder = arg
            elif opt in ("--savestep"):
                SRConfig.saveBestTopologiesStep = int(arg)
            elif opt in ("-t", "--topology"):
                SRConfig.topology_file = arg
            elif opt in ("-c", "--constraints"):
                SRConfig.constraints_file = arg
            elif opt in ("--maxbackprops"):
                SRConfig.maxBackpropIters = int(arg)
            elif opt in ("--initpopsize"):
                SRConfig.initialPopsize = int(arg)
            elif opt in ("--popsize"):
                SRConfig.popSize = int(arg)
            elif opt in ("--genpopsize"):
                SRConfig.generatedPopSize = int(arg)
            elif opt in ("--eagens"):
                SRConfig.eaGenerations = int(arg)
            elif opt in ("--histusage"):
                SRConfig.historyUsageProb = float(arg)
            elif opt in ("--inheritfirst"):
                SRConfig.inheritFromFirstParent = float(arg)
            elif opt in ("--pruning"):
                if 'true' in arg.lower():
                    SRConfig.pruning = True
                elif 'false' in arg.lower():
                    SRConfig.pruning = False
                else:
                    print("Error: wrong value for 'pruning'")
                    sys.exit(2)
            elif opt in ("--perturb"):
                SRConfig.perturbation = float(arg)
            elif opt in ("--perturbstep"):
                SRConfig.perturbationStep = int(arg)
            elif opt in ("--train_val"):
                SRConfig.train_val_split = float(arg)
            elif opt in ("--train_data"):
                SRConfig.train_data = 'data/'+arg
            elif opt in ("--valid_data"):
                SRConfig.valid_data = 'data/'+arg
            elif opt in ("--test_data"):
                SRConfig.test_data = 'data/'+arg
            elif opt in ("--scale"):
                SRConfig.scaleCoeff = float(arg)
            elif opt in ("--rmsescale"):
                SRConfig.rmseScaleCoeff = float(arg)
            elif opt in ("--stack_size"):
                SRConfig.stack_size = int(arg)
            elif opt in ("--update_step"):
                SRConfig.update_step = int(arg)
            elif opt in ("--sng2rmse"):
                SRConfig.sng2rmse = float(arg)
            elif opt in ("--reg2rmse"):
                SRConfig.reg2rmseMainRun = float(arg)
            elif opt in ("--constr2rmse"):
                SRConfig.constr2rmse = float(arg)
            elif opt in ('--adaptive'):
                if 'true' in arg.lower():
                    SRConfig.adaptive = True
                elif 'false' in arg.lower():
                    SRConfig.adaptive = False
                else:
                    print("Error: wrong value for 'adaptive'")
                    sys.exit(2)
            elif opt in ('--best_ext'):
                if 'true' in arg.lower():
                    SRConfig.best_ext = True
                elif 'false' in arg.lower():
                    SRConfig.best_ext = False
                else:
                    print("Error: wrong value for 'best_ext'")
                    sys.exit(2)
            elif opt in ("-r", "--reg_lambda"):
                SRConfig.reg_lambda = float(arg)
            elif opt in ("-l", "--learning_rate"):
                SRConfig.learning_rate_reg = float(arg)
            elif opt in ("--learning_rate_noreg"):
                SRConfig.learning_rate_noreg = float(arg)
            elif opt in ("-w", "--w_thr"):
                SRConfig.activeWeightThreshold = float(arg)
            elif opt in ('--active_units'):
                if 'true' in arg.lower():
                    SRConfig.active_units = True
                elif 'false' in arg.lower():
                    SRConfig.active_units = False
                else:
                    print("Error: wrong value for 'active_units'")
                    sys.exit(2)
            elif opt in ("--minibatch"):
                SRConfig.minibatchSize = int(arg)
            elif opt in ("--max_weights"):
                SRConfig.maxExpressionWeights = int(arg)
            elif opt in ("--iters"):
                SRConfig.noreg_phase_C = int(arg) - 1000
                SRConfig.totalIters = int(arg)
            elif opt in ("--iters_A"):
                SRConfig.noreg_phase_A = int(arg)
            elif opt in ("--noconstr_iters"):
                SRConfig.noconstr_iters = int(arg)
            elif opt in ("--noreg_iters"):
                SRConfig.max_noreg_iters = int(arg)
            elif opt in ("--reg_iters"):
                SRConfig.max_reg_iters = int(arg)
    except getopt.GetoptError:
        print('SRMain.py -s <seed> -o <outfolder> -r <reg_lambda> -t <topology> -c <constraints> -l <learning_rate>'
              '-w <w_thr>'
              '--learning_rate_noreg <learning_rate_noreg>'
              '--stack_size <stack_size>', '--update_step <update_step>', '--sng2rmse <sng2rmse>', '--reg2rmse <reg2rmse>', '--constr2rmse <constr2rmse>',
              '--adaptive <adaptive>', '--best_ext <best_ext>'
                                       '--max_weights <max_weights> --scale <scaleCoeff>'
                                       '--iters <total_iters> --iters_A <iters_A> --reg_iters <reg_iters> --noreg_iters <noreg_iters> --noconstr_iters <noconstr_iters>'
                                       '--train_val --train_data --valid_data --test_data --minibatch')
        sys.exit(2)
    # ---
    SRConfig.totalItersNewbornNoPerturbations = min(SRConfig.fineTuningIters - 1, SRConfig.totalItersNewbornNoPerturbations)
    # ---
    argv_str = argv_str + '\n\n'
    SRConfig.outputNamePrefix = f'{outfolder}/seed={SRConfig.seed}'
    SRConfig.outputPath = outfolder
    SRConfig.progress_file = SRConfig.outputNamePrefix + '_progress' + SRConfig.fileSuffix + '.txt'
    return argv_str