import random

seed: int = 19
r: random = None
maxBackpropIters: int = 90000
initActiveUnitsProp = 1.0
initialPopsize: int = 10  # --- 150, 40
popSize: int = 10  # --- 50, 30; TODO 50
generatedPopSize: int = 50  # --- 100, 30
eaGenerations: int = 60
nsgaAcceptanceScalingFactor: float = 1.5
nsgaGenerationsPerturbed: int = 1
nsgaGenerationsNoPerturbations: int = 2  # --- ORIGINAL: 3
nichingInterval: int = (
    10  # --- nichingInterval <= 0 ... no masterTopologyId assignment is carried out.
)
# --- nichingInterval > 0 ... masterTopologyId assignment is done every nichingInterval eaGeneration

historySize: int = 5  # --- ORIGINAL: 5; TODO 10
historyType: int = 0  # --- 0 ... original vanilla version, 1 ... advanced version
historySizeQuantile: int = 2  # --- 4 ... take it all
historyQualityQuantile: int = 1  # --- 0     TODO 1
historyScalingFactor: float = 1.5
historyUsageProb: float = (
    1.0  # --- probability of using history configs, otherwise a new config is randomly generated
)
historySortedDescending: bool = (
    True  # --- True ... configs sorted in an descending order, i.e., reverse=True. False, reverse=False.
)
historyConfigPreferences: int = (
    0  # --- -1 ... uses roulette wheel to prefer small configs
)
# ---  1 ... uses roulette wheel to prefer small configs
# ---  0 ... random selection
historyCheckConfigSize: bool = (
    False  # --- True ... config larger than the largest one in the history will not be added to the history
)
# ---          and all configs larger than the largest one in the history will be removed from the history.
# --- False ... config size is not checked.

totalIters: int
totalItersInit: int = 100  # --- 100
totalItersNewbornPerturbed: int = 100  # --- ORIGINAL: 2, TODO 150
totalItersNewbornNoPerturbations: int = 10  # --- ORIGINAL: 2, TODO 10
totalItersRun: int = 100  # --- 100
fineTuningIters: int = 50  # --- 50
finalIters: int = 200  # --- 500
alternatePrefs: bool = True

activeWeightThreshold: float = 0.0001  # --- 0.0001
weight_min_init: float = 0.05  # --- 0.05
weight_max_init: float = 0.5  # --- 0.5
weight_sigma: float = 0.2  # --- 0.2
weight_init_one: float = 1.0  # --- probability that the weight is initialized to +/-1

reg_type: str = "L05"
minibatchSize: int = 50
epochLength: int = 10
train_val_split: float = 0.67
learning_rate_newborn: float = 0.01  # --- old: 0.01, new: 0.005
learning_rate_finetune: float = 0.001  # --- old: 0.001, new: 0.0025
learning_rate_finaltune: float = 0.0001  # --- old: 0.001, new: 0.0001
learning_rate_reg: float = 0.01
learning_subset_size: float = (
    1.0  # --- x == 1.0 ... all subTopology parameters are updated in each backprop iteration.
)
# --- 0 < x < 1 ... only a subset of parameters is updated.
perturbation: float = 1.0
perturbationStep: int = 20  # --- perturb every perturbationFreq generation
perturbationResetHistory: bool = (
    True  # --- True ... history is reset on perturbation. False, history remains unchanged on perturbation.
)
perturbationUseArchive: bool = (
    False  # --- True ... perturbationUseArchive solutions are merged with a current population before perturbation is carried out
)

pruning: bool = True
pruningStep: int = 1000  # --- number of generations between perturbations
pruningIters: int = 50  # --- number of iterations applied to the pruned subTopology
pruningOutputThreshold: float = (
    0.01  # --- units that have the maximum absolute output value less than the threshold are inactivated
)
pruningMinWeightRatio: float = (
    0.001  # --- if the absolute value of the weight is less than 'pruningMinWeightRatio * maximumWeight', then the weight is pruned
)

tournamentSize1: int = 2
tournamentSize2: int = 2
rectify: float = 0.5  # --- probability of using the subTopology rectification operator
enhancedDominated: bool = (
    False  # --- True ... unused non-dominated solutions are added to dominated set in splitPopulationByDomination
)
crossoverRate: float = 0.80  # --- 0.8
inheritFromFirstParent: float = (
    0.9  # --- probability of inheriting genome of the first parent
)
inheritWeights: float = 0.80  # --- 0.8
applyActivityOrConfigMutation: float = 0.50  # --- 0.50 / 0.25
nodeActivityMutationRate: float = 0.10  # --- 0.10 / 0.75
nodeConfigMutationRate: float = 0.10  # --- 0.10 / 0.75
nodeUndersampledMutationRate: float = 0.1
nodeOversampledMutationRate: float = 0.1
utilLow: float = 0.5
utilHigh: float = 0.5

adaptive: bool = True
stack_size: int = 10
update_step: int = 1
scaleCoeff: float = 1.0
rmseScaleCoeff: float = 1.0

reg2rmse: float = 0.0
reg2rmseMainRun: float = 0.5  # --- 0.1           TODO 0.5
regFreeInitialPhase: int = (
    3  # --- number of the first EA generations without regularization
)
constr2rmse: float = 0.5
sng2rmse: float = 0.5

minNbOfActiveNodes: int = 2  # --- 2
minNbOfActiveNodesToTune: int = 7
maxNbOfActiveNodesInHistory: int = 100
maxNbOfTunedSolutions: int = 10
maxNbOfActiveNodesToSympyfy: int = 7
maxFrontNb: int = 1000

precisionDigits: int = 15
simplificationTimeout: int = 15

continuousStart: int = 1
continuousEnd: int = 22
continuousStep: int = 5
continuousBatch: int = 150  # --- maximum number of samples in the training data set

saveBestTopologiesStep: int = 2
savePickleFinal: bool = False
savePickleIntermediate: bool = False
fileSuffix: str = ""

# -----------------------------------------------

topology_file: str
constraints_file: str = ""
outputNamePrefix: str
outputPath: str
progress_file: str

train_data: str
valid_data: str
test_data: str

# -------------------


def saveConfigurationToFile():
    f = open(outputPath + "/" + f"seed={seed}_configuration" + fileSuffix + ".txt", "w")
    f.write(f"seed: {seed}\n")
    f.write(f"maxBackpropIters: {maxBackpropIters}\n")
    f.write(f"initActiveUnitsProp: {initActiveUnitsProp}\n")
    f.write(f"initialPopsize: {initialPopsize}\n")
    f.write(f"popSize: {popSize}\n")
    f.write(f"generatedPopSize: {generatedPopSize}\n")
    f.write(f"eaGenerations: {eaGenerations}\n")
    f.write(f"nsgaAcceptanceScalingFactor: {nsgaAcceptanceScalingFactor}\n")
    f.write(f"nsgaGenerationsPerturbed: {nsgaGenerationsPerturbed}\n")
    f.write(f"nsgaGenerationsNoPerturbations: {nsgaGenerationsNoPerturbations}\n")
    f.write(f"nichingInterval: {nichingInterval}\n")
    f.write(f"\n")
    f.write(f"historySize: {historySize}\n")
    f.write(f"historyType: {historyType}\n")
    f.write(f"historySizeQuantile: {historySizeQuantile}\n")
    f.write(f"historyQualityQuantile: {historyQualityQuantile}\n")
    f.write(f"historyScalingFactor: {historyScalingFactor}\n")
    f.write(f"historyCheckConfigSize: {historyCheckConfigSize}\n")
    f.write(f"historyConfigPreferences: {historyConfigPreferences}\n")
    f.write(f"historyUsageProb: {historyUsageProb}\n")
    f.write(f"historySortedDescending: {historySortedDescending}\n")
    f.write(f"\n")
    f.write(f"totalItersInit: {totalItersInit}\n")
    f.write(f"totalItersNewbornPerturbed: {totalItersNewbornPerturbed}\n")
    f.write(f"totalItersNewbornNoPerturbations: {totalItersNewbornNoPerturbations}\n")
    f.write(f"totalItersRun: {totalItersRun}\n")
    f.write(f"fineTuningIters: {fineTuningIters}\n")
    f.write(f"finalIters: {finalIters}\n")
    f.write(f"alternatePrefs: {alternatePrefs}\n")
    f.write(f"\n")
    f.write(f"activeWeightThreshold: {activeWeightThreshold}\n")
    f.write(f"weight_min_init: {weight_min_init}\n")
    f.write(f"weight_max_init: {weight_max_init}\n")
    f.write(f"weight_sigma: {weight_sigma}\n")
    f.write(f"weight_init_one: {weight_init_one}\n")
    f.write(f"\n")
    f.write(f"reg_type: {reg_type}\n")
    f.write(f"minibatchSize: {minibatchSize}\n")
    f.write(f"epochLength: {epochLength}\n")
    f.write(f"train_val_split: {train_val_split}\n")
    f.write(f"learning_rate_newborn: {learning_rate_newborn}\n")
    f.write(f"learning_rate_finetune: {learning_rate_finetune}\n")
    f.write(f"learning_rate_finaltune: {learning_rate_finaltune}\n")
    f.write(f"learning_rate_reg: {learning_rate_reg}\n")
    f.write(f"learning_subset_size: {learning_subset_size}\n")
    f.write(f"\n")
    f.write(f"perturbation: {perturbation}\n")
    f.write(f"perturbationStep: {perturbationStep}\n")
    f.write(f"perturbationResetHistory: {perturbationResetHistory}\n")
    f.write(f"perturbationUseArchive: {perturbationUseArchive}\n")
    f.write(f"\n")
    f.write(f"pruning: {pruning}\n")
    f.write(f"pruningStep: {pruningStep}\n")
    f.write(f"pruningIters: {pruningIters}\n")
    f.write(f"pruningOutputThreshold: {pruningOutputThreshold}\n")
    f.write(f"pruningMinWeightRatio: {pruningMinWeightRatio}\n")
    f.write(f"\n")
    f.write(f"tournamentSize1: {tournamentSize1}\n")
    f.write(f"tournamentSize2: {tournamentSize2}\n")
    f.write(f"rectify: {rectify}\n")
    f.write(f"enhancedDominated: {enhancedDominated}\n")
    f.write(f"crossoverRate: {crossoverRate}\n")
    f.write(f"inheritFromFirstParent: {inheritFromFirstParent}\n")
    f.write(f"inheritWeights: {inheritWeights}\n")
    f.write(f"applyActivityOrConfigMutation: {applyActivityOrConfigMutation}\n")
    f.write(f"nodeActivityMutationRate: {nodeActivityMutationRate}\n")
    f.write(f"nodeConfigMutationRate: {nodeConfigMutationRate}\n")
    f.write(f"nodeUndersampledMutationRate: {nodeUndersampledMutationRate}\n")
    f.write(f"nodeOversampledMutationRate: {nodeOversampledMutationRate}\n")
    f.write(f"utilLow: {utilLow}\n")
    f.write(f"utilHigh: {utilHigh}\n")
    f.write(f"\n")
    f.write(f"adaptive: {adaptive}\n")
    f.write(f"stack_size: {stack_size}\n")
    f.write(f"update_step: {update_step}\n")
    f.write(f"scaleCoeff: {scaleCoeff}\n")
    f.write(f"rmseScaleCoeff: {rmseScaleCoeff}\n")
    f.write(f"\n")
    f.write(f"reg2rmseMainRun: {reg2rmseMainRun}\n")
    f.write(f"regFreeInitialPhase: {regFreeInitialPhase}\n")
    f.write(f"constr2rmse: {constr2rmse}\n")
    f.write(f"sng2rmse: {sng2rmse}\n")
    f.write(f"\n")
    f.write(f"minNbOfActiveNodes: {minNbOfActiveNodes}\n")
    f.write(f"maxNbOfActiveNodes: {minNbOfActiveNodesToTune}\n")
    f.write(f"maxNbOfTunedSolutions: {maxNbOfTunedSolutions}\n")
    f.write(f"\n")
    f.write(f"topology_file: {topology_file}\n")
    f.write(f"constraints_file: {constraints_file}\n")
    f.write(f"train_data: {train_data}\n")
    f.write(f"test_data: {test_data}\n")
    f.write(f"\n")
    f.write(f"continuousStart: {continuousStart}\n")
    f.write(f"continuousEnd: {continuousEnd}\n")
    f.write(f"continuousStep: {continuousStep}\n")
    f.write(f"continuousBatch: {continuousBatch}\n")
    f.write(f"\n")
    f.write(f"outputNamePrefix: {outputNamePrefix}\n")
    f.write(f"outputPath: {outputPath}\n")
    f.write(f"saveBestTopologiesStep: {saveBestTopologiesStep}\n")
    f.write(f"savePickleFinal: {savePickleFinal}\n")
    f.write(f"savePickleIntermediate: {savePickleIntermediate}\n")
    f.write(f"progress_file: {progress_file}\n")
    f.write(f"fileSuffix: {fileSuffix}\n")

    f.close()
