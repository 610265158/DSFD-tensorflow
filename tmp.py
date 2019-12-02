import numpy as np


a=np.fromfile('/home/lz/mace/build/cmake-build/host/mace/tools/mace.out_tower_0_outputs_0',dtype=np.float32)
a=np.reshape(a,[1,4200,5])
print(a)
print(a.shape)

