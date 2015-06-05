import sys
import pickle
import traceback

if __name__ == '__main__':
    print "START READING..."

    try:
        if len(sys.argv) < 2:
            raise Exception("invalid # of arguments" + str(sys.argv))

        with open(sys.argv[1], "rb") as inputfile:
            obj = pickle.load(inputfile)

        print obj

    except:
        print traceback.format_exc()
        raise SystemExit
