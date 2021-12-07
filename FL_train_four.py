import time
import os
from FL_code_completion_four import FL_CodeCompletion_Four

def run(args=None):

    cc_cls = FL_CodeCompletion_Four

    parser = cc_cls.default_args()
    args = parser.parse_args()
    cc = cc_cls(args)
    cc.train()
    
 


if __name__ == "__main__":
    run()