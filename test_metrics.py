import os

import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

from tests.metric_calculations import mae, phase_detection
from utils.data_utils import preprocess_phase_detection, print_precision_table, get_group_delay
from utils.plotting import plot_mae, plot_recall_precision
from utils.arg_checks import validate_test_metrics_args


def _load_pred_res(type_):
    return np.load(os.path.join(args.prediction_dir, f'{type_}.npy'))

def run_mae():
    preds = _load_pred_res('predictions')
    gts = _load_pred_res('ground_truths')

    group_delay = get_group_delay(meta['bp_filter_len'])
    errors = mae(preds, gts, group_delay)

    print('\nMean absolute errors (\u03BCV)')
    print(f'{'Whole forecast':20} MAE={errors['mae_all']:.2f}  std={errors['std_all']:.2f}')
    print(f'{'True forecast':20} MAE={errors['mae_true']:.2f}  std={errors['std_true']:.2f}')
    print('MAE as a function of time ', np.round(errors['mae_func_of_time'], 2))

    if args.plot:
        ms_per_sample = 1000 / meta['fs']
        fig = plot_mae(errors['mae_func_of_time'], group_delay, ms_per_sample)
        fig.savefig(os.path.join(args.prediction_dir, 'mae.png'), dpi=600)
        plt.close(fig)

def run_phase_detection():
    
    if os.path.isfile(os.path.join(args.prediction_dir, 'phase_detection_x.npy')):
        x = _load_pred_res('phase_detection_x')
        y = _load_pred_res('phase_detection_y')
    else:
        inputs = _load_pred_res('inputs')
        gts = _load_pred_res('ground_truths')
        preds = _load_pred_res('predictions')
        x, y = preprocess_phase_detection(inputs, gts, preds)
        np.save(os.path.join(args.prediction_dir, 'phase_detection_x.npy'), x)
        np.save(os.path.join(args.prediction_dir, 'phase_detection_y.npy'), y)

    precision, recall, thresholds = phase_detection(x[:, :, args.detection_ts], y[:, args.detection_ts])

    print_precision_table(precision, recall, thresholds)

    if args.plot:
        fig = plot_recall_precision(precision, recall, thresholds)
        fig.savefig(os.path.join(args.prediction_dir, 'precision_recall.png'), dpi=600)
        plt.close(fig)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('test_type', type=str, nargs='+', choices=['mae', 'phase_detection'],
                        help='Type of tests to be run')
    parser.add_argument('prediction_dir', type=str, default=None,
                        help='Where predictions to be tested are saved')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to plot the results. Plots are saved to prediction_dir.')
    parser.add_argument('--detection_ts', type=int, default=66,
                        help='Time stamp to be used for phase detection (default: 66, corresponding to 6 ms in the future with sampling rate of 500 Hz and filter length of 128)')

    args = parser.parse_args()

    with open(os.path.join(args.prediction_dir, 'predict_meta_data.json')) as f:
        meta = json.load(f)

    validate_test_metrics_args(args, meta)

    if 'mae' in args.test_type:
        run_mae()
    if 'phase_detection' in args.test_type:
        run_phase_detection()
