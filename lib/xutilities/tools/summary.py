import argparse
import os
import re
import glob
from ast import literal_eval

import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file-name', type=str)
parser.add_argument('-d', '--directory', type=str)
parser.add_argument('-o', '--output', type=str, default='summary.csv')
parser.add_argument('-r', '--recent', type=int, default=0)
parser.add_argument('--pbar', action='store_true')
parser.add_argument('--track-name', default='acc')
parser.add_argument('--track-phase', default='val')
parser.add_argument('--reverse', action='store_true', default=False)
parser.add_argument('--expand', nargs='+', default=[])

args = parser.parse_args()

assert (not (args.file_name is None and args.directory is None)), \
    'must specify a file or a directory'


def get_info(file_name):
    info = {'meta': {}, 'epochs': [], 'results': {}}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            m = re.match(
                r'.*\[DEBUG.*\].*utils.*[\s]+([\-\w]+)[\s]+ = (.*)', line)
            if m is not None:
                info_pair = m.groups()
                if info_pair[0] in args.expand:
                    _temp_dict = literal_eval(info_pair[1])
                    keys = list(_temp_dict.keys())
                    for k in keys:
                        _temp_dict[f'{info_pair[0]}_{k}'] = _temp_dict.pop(k)
                    info['meta'].update(_temp_dict)
                else:
                    info['meta'][info_pair[0]] = info_pair[1]
                continue

            m = re.match(f".+Epoch ([0-9]+), ({{'phase': '{args.track_phase}'.+}})", line)
            if m is not None:
                epoch_info = {'epoch': int(m.group(1))}
                val_info = literal_eval(m.group(2))
                epoch_info.update(
                    {f"{args.track_name}": val_info[args.track_name]}
                )
                info['epochs'].append(epoch_info)
                continue
            
            m = re.match(
                r".+Best model at epoch ([0-9]+), ({'phase': 'test'.+})", line)
            if m is not None:
                test_info = literal_eval(m.group(2))
                test_info.pop('phase')
                info['meta'].update(test_info)
                continue
        if len(info['epochs']) > 0:
            sorted_epochs = sorted(
                info['epochs'], key=lambda x: x[f'{args.track_name}'], reverse=args.reverse)
            info['meta'][f'tracked_{args.track_name}'] =\
                sorted_epochs[-1][f'{args.track_name}']
        #     # info['meta']['best_model_close'] = sorted_epochs[-1]['acc_close']
        #     # info['meta']['best_model_open'] = sorted_epochs[-1]['acc_open']
            info['meta']['best_track_epoch'] = sorted_epochs[-1]['epoch']
            info['meta']['latest_epoch'] = len(sorted_epochs)

    return info


info_list = []
if args.file_name is not None:
    info_list.append(get_info(args.file_name))

if args.directory is not None:
    files = glob.glob(args.directory + '/*/*.log')
    files.sort(key=os.path.getmtime)
    files = files[-args.recent:]
    for file in tqdm(
        files, ncols=80, ascii=True, leave=False, disable=~args.pbar
    ):
        info_list.append(get_info(file)['meta'])

df = pd.DataFrame(info_list)
df.to_csv(args.output)
