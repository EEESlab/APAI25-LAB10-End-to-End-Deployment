import pandas as pd
import argparse


pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser(prog='read_json')
parser.add_argument('--json_file',
                        help="json file to open",required = True)

args = parser.parse_args()
df = pd.read_json(args.json_file)
print(df)

