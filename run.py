import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-d",dest="decimal_point", action="store")
parser.add_argument("-f",dest="fast_point", action="store_true")

args = parser.parse_args()

print(args.decimal_point)
print(args.fast_point)
