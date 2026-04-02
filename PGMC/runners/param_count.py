import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import *


def main(args):
    print('>>> In function `main`')
    
    load_models(args)
        
        
if __name__ == '__main__':
    args = get_arguments()
    # Set random seed
    set_random_seed(args.seed)
    
    main(args)
