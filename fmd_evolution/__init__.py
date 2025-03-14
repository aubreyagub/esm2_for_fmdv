import os
import random
import numpy as np
import torch

SEED = 100 # set seed for reproducibility
os.environ["PYTHONHASHSEED"] = str(SEED) 
random.seed(SEED)
rng = np.random.RandomState(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .model_singleton import ModelSingleton 
from .protein_sequence import ProteinSequence
from .mutation_strategy import MutationStrategy, MinLogitPosSub, BlosumWeightedSub, MetropolisHastings
from .evolution import Evolution
from .evaluation_strategy import EvaluationStrategy
from .ranked_evaluation_strategy import RankedEvaluationStrategy
from .model_singleton import ModelSingleton
from .evaluation import Evaluation
from .protein_language_model import ProteinLanguageModel