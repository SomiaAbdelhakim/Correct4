import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--input', '-i', help='Input path', required=True, type=str)

args = parser.parse_args()
