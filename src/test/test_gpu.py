#!/scratch/cluster/clw4542/conda_envs/oil/bin/python

import os
import torch as th 


if __name__ == "__main__":
    # print environment variable named LD_LIBRARY_PATH
    print("LD_LIBRARY_PATH: ", os.environ.get('LD_LIBRARY_PATH'))
    print("GPU Available: ", th.cuda.is_available())
    atensor = th.tensor([1, 2, 3, 4, 5], device='cuda')
    print(atensor)