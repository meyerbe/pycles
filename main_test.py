import argparse
import json


def main():

    # Parse information from the command line
    parser = argparse.ArgumentParser(prog='PyCLES')
    parser.add_argument("namelist")
    args = parser.parse_args()

    file_namelist = open(args.namelist).read()
    namelist = json.loads(file_namelist)
    del file_namelist

    print('')
    print('namelist uuid:', namelist['meta']['uuid'])
    print('')

    return



if __name__ == "__main__":
    main()
