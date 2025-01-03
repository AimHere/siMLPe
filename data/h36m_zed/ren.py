

import os

for prefix, dirs, files in os.walk('.'):
    for f in files:
        if (f[-4:] == '.npz'):
            infile = os.path.join(prefix, f)
            
            outfile = os.path.join(prefix, "%s%s"%(f[:-4], '_zed34_test.npz'))
            print("rename (%s,%s)"%(infile, outfile))
            os.rename(infile, outfile)
                                   

