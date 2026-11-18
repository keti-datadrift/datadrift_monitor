from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np

from easydict import EasyDict as edict

__C = edict()


cfg = __C




__C.TRAIN = edict()


__C.TRAIN.LEARNING_RATE = 0.001


__C.TRAIN.MOMENTUM = 0.9


__C.TRAIN.WEIGHT_DECAY = 0.0005


__C.TRAIN.GAMMA = 0.1


__C.TRAIN.STEPSIZE = [30000]


__C.TRAIN.DISPLAY = 10


__C.TRAIN.DOUBLE_BIAS = True


__C.TRAIN.TRUNCATED = False


__C.TRAIN.BIAS_DECAY = False


__C.TRAIN.USE_GT = False



__C.TRAIN.ASPECT_GROUPING = False


__C.TRAIN.SNAPSHOT_KEPT = 3


__C.TRAIN.SUMMARY_INTERVAL = 180



__C.TRAIN.SCALES = (600,)


__C.TRAIN.MAX_SIZE = 1000


__C.TRAIN.TRIM_HEIGHT = 600
__C.TRAIN.TRIM_WIDTH = 600


__C.TRAIN.IMS_PER_BATCH = 1


__C.TRAIN.BATCH_SIZE = 128


__C.TRAIN.FG_FRACTION = 0.25


__C.TRAIN.FG_THRESH = 0.5



__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.1


__C.TRAIN.USE_FLIPPED = True


__C.TRAIN.BBOX_REG = True



__C.TRAIN.BBOX_THRESH = 0.5


__C.TRAIN.SNAPSHOT_ITERS = 5000



__C.TRAIN.SNAPSHOT_PREFIX = 'res101_faster_rcnn'







__C.TRAIN.BBOX_NORMALIZE_TARGETS = True

__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = True
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)


__C.TRAIN.PROPOSAL_METHOD = 'gt'






__C.TRAIN.HAS_RPN = True

__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

__C.TRAIN.RPN_CLOBBER_POSITIVES = False

__C.TRAIN.RPN_FG_FRACTION = 0.5

__C.TRAIN.RPN_BATCHSIZE = 256

__C.TRAIN.RPN_NMS_THRESH = 0.7

__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000

__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

__C.TRAIN.RPN_MIN_SIZE = 8

__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)



__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0


__C.TRAIN.USE_ALL_GT = True


__C.TRAIN.BN_TRAIN = False




__C.TEST = edict()



__C.TEST.SCALES = (600,)


__C.TEST.MAX_SIZE = 1000



__C.TEST.NMS = 0.3



__C.TEST.SVM = False


__C.TEST.BBOX_REG = True


__C.TEST.HAS_RPN = False


__C.TEST.PROPOSAL_METHOD = 'gt'


__C.TEST.RPN_NMS_THRESH = 0.7

__C.TEST.RPN_PRE_NMS_TOP_N = 6000


__C.TEST.RPN_POST_NMS_TOP_N = 300


__C.TEST.RPN_MIN_SIZE = 16



__C.TEST.MODE = 'nms'


__C.TEST.RPN_TOP_N = 5000





__C.RESNET = edict()





__C.RESNET.MAX_POOL = False



__C.RESNET.FIXED_BLOCKS = 1





__C.MOBILENET = edict()


__C.MOBILENET.REGU_DEPTH = False



__C.MOBILENET.FIXED_LAYERS = 5


__C.MOBILENET.WEIGHT_DECAY = 0.00004


__C.MOBILENET.DEPTH_MULTIPLIER = 1.










__C.DEDUP_BOXES = 1. / 16.




__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])


__C.RNG_SEED = 3


__C.EPS = 1e-14


__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))


__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))


__C.MATLAB = 'matlab'


__C.EXP_DIR = 'default'


__C.USE_GPU_NMS = True


__C.GPU_ID = 0

__C.POOLING_MODE = 'align'


__C.POOLING_SIZE = 7


__C.MAX_NUM_GT_BOXES = 20


__C.ANCHOR_SCALES = [8,16,32]


__C.ANCHOR_RATIOS = [0.5,1,2]


__C.FEAT_STRIDE = [16, ]

__C.CUDA = False

__C.CROP_RESIZE_WITH_MAX_POOL = True

import pdb
def get_output_dir(imdb, weights_filename):
  """Return the directory where experimental artifacts are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def get_output_tb_dir(imdb, weights_filename):
  """Return the directory where tensorflow summaries are placed.
  If the directory does not exist, it is created.

  A canonical path is built using the name from an imdb and a network
  (if not None).
  """
  outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'tensorboard', __C.EXP_DIR, imdb.name))
  if weights_filename is None:
    weights_filename = 'default'
  outdir = osp.join(outdir, weights_filename)
  if not os.path.exists(outdir):
    os.makedirs(outdir)
  return outdir


def _merge_a_into_b(a, b):
  """Merge config dictionary a into config dictionary b, clobbering the
  options in b whenever they are also specified in a.
  """
  if type(a) is not edict:
    return

  for k, v in a.items():

    if k not in b:
      raise KeyError('{} is not a valid config key'.format(k))


    old_type = type(b[k])
    if old_type is not type(v):
      if isinstance(b[k], np.ndarray):
        v = np.array(v, dtype=b[k].dtype)
      else:
        raise ValueError(('Type mismatch ({} vs. {}) '
                          'for config key: {}').format(type(b[k]),
                                                       type(v), k))


    if type(v) is edict:
      try:
        _merge_a_into_b(a[k], b[k])
      except:
        print(('Error under config key: {}'.format(k)))
        raise
    else:
      b[k] = v


def cfg_from_file(filename):
  """Load a config file and merge it into the default options."""
  import yaml
  with open(filename, 'r') as f:
    yaml_cfg = edict(yaml.load(f))

  _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
  """Set config keys via list (e.g., from command line)."""
  from ast import literal_eval
  assert len(cfg_list) % 2 == 0
  for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
    key_list = k.split('.')
    d = __C
    for subkey in key_list[:-1]:
      assert subkey in d
      d = d[subkey]
    subkey = key_list[-1]
    assert subkey in d
    try:
      value = literal_eval(v)
    except:

      value = v
    assert type(value) == type(d[subkey]), \
      'type {} does not match original type {}'.format(
        type(value), type(d[subkey]))
    d[subkey] = value

