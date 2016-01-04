import sys
from document_filter import *

def main(argv):
    LANG      = argv[1]
    POS_LABEL = argv[2]
    TFILE     = argv[3]
    EFILE     = argv[4]

    df        = DocumentFilter()
    df.tune(LANG, POS_LABEL, TFILE, EFILE)

if __name__ == '__main__':
    argv      = sys.argv
    main(argv)
