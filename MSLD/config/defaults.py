from yacs.config import CfgNode as CN
from .model_path import MODEL_PATH

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.MODEL = CN()
_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = "linear_norm"
_C.MODEL.HEAD.DIM =  512 #28224
_C.MODEL.HEAD.KMEAN = 100
_C.MODEL.WEIGHT = ""

# Data option
_C.DATA = CN()
#_C.DATA.TRAIN_IMG_ORI = '/home/lab/data/dataset/CARS196_2/'
_C.DATA.TRAIN_IMG_ORI = '/home/lab/data/dataset/CUB_200_2011_2/CUB_200_2011/'
_C.DATA.TRAIN_IMG_SOURCE = _C.DATA.TRAIN_IMG_ORI+'/train.txt'
_C.DATA.TEST_IMG_SOURCE = _C.DATA.TRAIN_IMG_ORI+'/test.txt'

_C.DATA.TRAIN_BATCHSIZE = 36 #256 #160
_C.DATA.TEST_BATCHSIZE = 128
_C.DATA.NUM_WORKERS = 0 #4
_C.DATA.NUM_INSTANCES = 3 #8

# Input option
_C.INPUT = CN()
_C.INPUT.MODE = 'BGR'

# SOLVER
_C.SOLVER = CN()
_C.SOLVER.IS_FINETURN = False
_C.SOLVER.FINETURN_MODE_PATH = ''
_C.SOLVER.MAX_ITERS = 4000
_C.Epoch = 5
_C.SOLVER.STEPS = [1000, 2000, 3000]


# Logger
_C.LOGGER = CN()
_C.LOGGER.LEVEL = 20
_C.LOGGER.STREAM = 'stdout'


_C.dataset = 'CUB'
_C.data_name='CUB'
_C.mode = 'test' #train | test
_C.outf = './output/'
_C.iteration = 20000
_C.resume = '' 
_C.augment = True   
_C.neighbor_k = 1 #1
_C.lr = 0.0005 #0.0005
_C.theta1 = 1.6
_C.theta2 = 1.7  #1.8
_C.P = 5  # 40 | 5
_C.Q = 100 
_C.seed = 1000

_C.t = 0.1  
_C.neg_m = 1 
_C.alpha = 1.0
