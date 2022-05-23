import  time
import  DAQN
import DHW
import os
if __name__ == '__main__':
    print('='*80)
    print('   Source codes from CQU BDI group.   ')
    print('='*80)
    print("   Select models:")
    print("   1. DAQN        2. DHW")
    print('='*80)
    num = input('please enter the number of the model you want to run:')
    s = time.time()
    if num == "1":
        str = ("python train_scripts.py")
        os.chdir("DAQN")
        os.system(str)
    elif num == "2":
        str = ("python -u train_NDfdml.py")
        os.chdir("DHW")
        os.system(str)
    e = time.time()
    print("Running time: %f s" % (e - s))
