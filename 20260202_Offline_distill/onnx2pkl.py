import onnx
import numpy as np


if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'int'):
    np.int = np.int32
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'bool'):
    np.bool = bool

from meghair.onnx_megdl import import_onnx
from meghair.utils.io import dump

onnxModel = '/data/jg-face-unlock-fea-impove/vivo_new/helmet/txy_light_helmet_990_10000/pth/checkpoint_428.onnx'
onnx_model = onnx.load(onnxModel)
onnx_model.ir_version = 7
net = import_onnx(onnx_model)
dump(net, fobj='/data/jg-face-unlock-fea-impove/vivo_new/helmet/txy_light_helmet_990_10000/pth/checkpoint_428.pkl')
print('done')
