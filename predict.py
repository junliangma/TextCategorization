import sys
from document_filter import *

def main(argv):
    IFILE = argv[1]
    EFILE = argv[2]
    OFILE = argv[3]

    df    =  DocumentFilter()
    df.predict(IFILE, EFILE, OFILE)

if __name__ == '__main__':
    argv  = sys.argv
    main(argv)
