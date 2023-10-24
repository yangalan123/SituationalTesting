import argparse
import glob
import os
import pickle

import loguru
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from SitTest.utils import (
    process_gt_answer_response,
    check_error_type,
    ERROR_TYPES
)

from Data.SimpleBoxOpeningEnv.action_templates import extract_time_from_single_action


def generate_color_map(n_colors, brightness_threshold=0.65):
    # Create a colormap using the hsv color space
    hsv_colors = plt.cm.get_cmap('hsv', n_colors)

    # Convert RGB colors to HEX colors and filter out too light colors
    hex_colors = []
    for i in range(hsv_colors.N):
        rgb_color = hsv_colors(i)[:3]
        # Calculate brightness
        brightness = np.sqrt(0.299 * rgb_color[0] ** 2 + 0.587 * rgb_color[1] ** 2 + 0.114 * rgb_color[2] ** 2)
        if brightness < brightness_threshold:
            hex_colors.append(mcolors.rgb2hex(rgb_color))

    return hex_colors


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Debugging')
    parser.add_argument('--model_type', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--process_root_dir', type=str,
                        default="../GPT3Output_InteractiveDebugging/gpt-3.5-turbo")
    parser.add_argument('--log_root_dir', type=str,
                        default="../GPT3Output_InteractiveDebugging/{}/log_dir/")
    parser.add_argument('--visualization_root_dir', type=str,
                        default="../GPT3Output_InteractiveDebugging/{}/visualization_dir_final/")
    args = parser.parse_args()
    args.process_root_dir = os.path.join(args.process_root_dir, args.model_type)
    # see whether process_root_dir is valid
    if not os.path.exists(args.process_root_dir):
        raise ValueError(f"process_root_dir {args.process_root_dir} does not exist")
    # see how many directories are in process_root_dir
    dirs = glob.glob(os.path.join(args.process_root_dir, "*"))
    if len(dirs) == 0:
        raise ValueError(f"process_root_dir {args.process_root_dir} is empty")
    args.log_root_dir = args.log_root_dir.format(args.model_type)
    args.visualization_root_dir = args.visualization_root_dir.format(args.model_type)
    os.makedirs(args.log_root_dir, exist_ok=True)
    os.makedirs(args.visualization_root_dir, exist_ok=True)
    args.exp_name = "running_log_compilation"

    return args


def get_step_from_workload(workload: str):
    lines = workload.split("\n")
    assert "Step" in lines[-3] and "Question" in lines[-2] and "Answer" in lines[-1]
    return extract_time_from_single_action(lines[-3])


def plot_figures(figpath, error_keys, filtered_buf, filtered_step_idx, color_map, fig_title, marker_map,
                 include_list=None, exclude_list=None):
    assert include_list is None or exclude_list is None
    assert include_list is not None or exclude_list is not None
    plt.title(fig_title)
    plt.xlabel("Steps")
    plt.ylabel("Type occurrence")
    # resize the figure
    plt.gcf().set_size_inches(28, 18)
    plt.subplots_adjust(bottom=0.3)
    # set font size
    linewidth = 8
    markersize = 50
    plt.rcParams.update({'font.size': 62, "lines.markersize": markersize, "lines.linewidth": linewidth})

    already_legend = False
    for error_key in include_list:
        if "Not Parseable" in error_key or "Inconsistent States Number" in error_key:
            continue
        if exclude_list is not None and error_key in exclude_list:
            continue
        if include_list is not None and error_key not in include_list:
            continue
        error_type_count_mean = []
        error_type_count_std = []
        for step_item in filtered_buf:
            error_type_count_mean.append(np.mean([x[error_key] for x in step_item]))
            error_type_count_std.append(np.std([x[error_key] for x in step_item]) / np.sqrt(len(step_item)))
        if "Maintain Correct Belief" in error_key:
            ax = plt.gca()
            fig = plt.gcf()
            lower, _ = ax.get_ylim()
            ax.set_ylim(lower, 2.1)
            ax2 = ax.twinx()
            ax2.errorbar(filtered_step_idx, error_type_count_mean, yerr=error_type_count_std, ecolor="black", capsize=5,
                         label=ERROR_TYPES[error_key], color=color_map[error_key], marker=marker_map[error_key])
            ax2.set_ylabel("#(Maintain Correct Belief States)")
            ax2.tick_params(axis='y')
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()

            # Create a legend for all lines
            ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
            already_legend = True
        else:
            # plot error type error bar
            plt.errorbar(filtered_step_idx, error_type_count_mean, yerr=error_type_count_std, ecolor="black", capsize=5,
                         label=ERROR_TYPES[error_key], color=color_map[error_key], marker=marker_map[error_key])
    if not already_legend:
        lower, _ = plt.ylim()
        plt.ylim(lower, 2.1)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4)
    plt.savefig(figpath)
    plt.close()
    plt.clf()


if __name__ == '__main__':
    args = parse_args()
    logger = loguru.logger
    logger.add(f"{args.log_root_dir}/" + args.exp_name + ".log", mode='w')
    dirnames = glob.glob(os.path.join(args.process_root_dir, "*"))
    assert len(dirnames) > 0, f"process_root_dir {args.process_root_dir} is empty"
    error_keys = list(ERROR_TYPES.keys())
    color_map = {}
    color_pallete = generate_color_map(len(error_keys) * 2)
    print(color_pallete)
    for err_i, error_key in enumerate(error_keys):
        if "Parse" not in error_key:
            color_map[error_key] = color_pallete[err_i]
    # if visualization_root_dir is not empty, delete all files in it
    if len(glob.glob(os.path.join(args.visualization_root_dir, "*"))) > 0:
        logger.info(f"visualization_root_dir {args.visualization_root_dir} is not empty, deleting all files in it")
        for filename in glob.glob(os.path.join(args.visualization_root_dir, "*")):
            os.remove(filename)
    for dirname in dirnames:
        # initialize the figure
        plt.gcf().set_size_inches(28, 18)
        plt.subplots_adjust(bottom=0.3)
        # set font size
        linewidth = 8
        markersize = 50
        plt.rcParams.update({'font.size': 62, "lines.markersize": markersize, "lines.linewidth": linewidth})
        # initialization done, now searching for data
        cur_exp_name = os.path.basename(dirname)
        cur_exp_name_path = dirname
        logger.info(f"Processing {cur_exp_name_path}")
        all_filenames = glob.glob(os.path.join(cur_exp_name_path, "responses_dir", "*pkl"))
        output_image_name = os.path.join(args.visualization_root_dir, cur_exp_name + ".pdf")
        if len(all_filenames) == 0:
            logger.info(f"{cur_exp_name} has no pkl file")
            continue
        # load the data
        all_steps_acc = [[], ]
        all_steps_acc_cf = [[], ]
        all_steps_f1 = [[], ]
        all_steps_f1_cf = [[], ]
        all_error_types = [[], ]
        reverse_gt_val = False
        for filename in all_filenames:
            data = pickle.load(open(filename, "rb"))
            file_responses = data[-1]
            if len(file_responses) > len(all_steps_acc) and len(all_steps_acc) <= 10:
                for _ in range(len(file_responses) - len(all_steps_acc)):
                    all_steps_acc.append([])
                    all_steps_acc_cf.append([])
                    all_steps_f1.append([])
                    all_steps_f1_cf.append([])
                    all_error_types.append([])
            step_recorder = set()
            for i, response in enumerate(file_responses):
                gt_answer = response['gt_answer']
                step_id = get_step_from_workload(response['workload'])
                if step_id not in step_recorder:
                    step_recorder.add(step_id)
                else:
                    continue
                if step_id == len(all_steps_acc):
                    all_steps_acc.append([])
                    all_steps_acc_cf.append([])
                    all_steps_f1.append([])
                    all_steps_f1_cf.append([])
                    all_error_types.append([])
                _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, response, logger,
                                                                                         reverse_gt_val=False, f1=True)
                all_steps_f1[step_id].append(_stat_acc['recall'] if "recall" in _stat_acc else 0.0)
                all_steps_acc[step_id].append(
                    1.0 if _stat_acc['recall'] == 1.0 and _stat_acc['precision'] == 1.0 else 0.0)
                _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, response, logger,
                                                                                         reverse_gt_val=True, f1=True)
                all_steps_f1_cf[step_id].append(_stat_acc['recall'] if "recall" in _stat_acc else 0.0)
                all_steps_acc_cf[step_id].append(
                    1.0 if _stat_acc['recall'] == 1.0 and _stat_acc['precision'] == 1.0 else 0.0)
                if all_steps_f1_cf[step_id][-1] > all_steps_f1[step_id][-1]:
                    reverse_gt_val = True
                    if "cf" not in cur_exp_name:
                        logger.info(f"Reverse gt value at {cur_exp_name}")
                if i > 0:
                    prev_response = file_responses[i - 1]
                    prev_gt_answer = prev_response['gt_answer']
                    error_type = check_error_type(gt_answer, response, prev_gt_answer, prev_response, logger,
                                                  reverse_gt_val)
                    all_error_types[step_id].append(error_type)
        # compute the average
        all_steps_acc_x_axis = [i for i in range(len(all_steps_acc)) if len(all_steps_acc[i]) > 0]
        all_steps_acc = [x for x in all_steps_acc if len(x) > 0]
        all_steps_acc_cf_x_axis = [i for i in range(len(all_steps_acc_cf)) if len(all_steps_acc_cf[i]) > 0]
        all_steps_acc_cf = [x for x in all_steps_acc_cf if len(x) > 0]
        all_steps_acc_mean = [np.mean(x) for x in all_steps_acc]
        all_steps_acc_std = [np.std(x) / np.sqrt(len(x)) for x in all_steps_acc]
        all_steps_acc_cf_mean = [np.mean(x) for x in all_steps_acc_cf]
        all_steps_acc_cf_std = [np.std(x) / np.sqrt(len(x)) for x in all_steps_acc_cf]
        all_steps_f1_x_axis = [i for i in range(len(all_steps_f1)) if len(all_steps_f1[i]) > 0]
        all_steps_f1 = [x for x in all_steps_f1 if len(x) > 0]
        all_steps_f1_cf_x_axis = [i for i in range(len(all_steps_f1_cf)) if len(all_steps_f1_cf[i]) > 0]
        all_steps_f1_cf = [x for x in all_steps_f1_cf if len(x) > 0]
        all_steps_f1_mean = [np.mean(x) for x in all_steps_f1]
        all_steps_f1_std = [np.std(x) / np.sqrt(len(x)) for x in all_steps_f1]
        all_steps_f1_cf_mean = [np.mean(x) for x in all_steps_f1_cf]
        all_steps_f1_cf_std = [np.std(x) / np.sqrt(len(x)) for x in all_steps_f1_cf]
        # plot the line chart with error bar, error bar needs to have tips up and down
        if reverse_gt_val:
            plt.errorbar(all_steps_acc_cf_x_axis, all_steps_acc_cf_mean, yerr=all_steps_acc_cf_std, ecolor="black",
                         capsize=5, label="Step-EM", color="red")
            plt.errorbar(all_steps_f1_cf_x_axis, all_steps_f1_cf_mean, yerr=all_steps_f1_cf_std, ecolor="black",
                         capsize=5, label="State-EM", color="blue")
        else:
            plt.errorbar(all_steps_acc_x_axis, all_steps_acc_mean, yerr=all_steps_acc_std, ecolor="black",
                         capsize=5, label="Step-EM", color="red")
            plt.errorbar(all_steps_f1_x_axis, all_steps_f1_mean, yerr=all_steps_f1_std, ecolor="black",
                         capsize=5, label="State-EM", color="blue")
        plt.legend()
        plt.title(cur_exp_name)
        plt.xlabel("Steps")
        plt.ylabel("Acc")
        # resize the figure
        plt.savefig(output_image_name)
        plt.close()
        plt.clf()
        # plot the error type
        output_pickle_name = os.path.join(args.visualization_root_dir, cur_exp_name + "_vis_data.pickle")
        with open(output_pickle_name, 'wb') as f:
            pickled_data = {
                'all_error_types': all_error_types,
                'all_steps_acc': all_steps_acc,
                'all_steps_f1': all_steps_f1,
                'all_steps_acc_cf': all_steps_acc_cf,
                'all_steps_f1_cf': all_steps_f1_cf,
                'all_steps_acc_mean': all_steps_acc_mean,
                'all_steps_acc_std': all_steps_acc_std,
                'all_steps_f1_mean': all_steps_f1_mean,
                'all_steps_f1_std': all_steps_f1_std,
                'all_steps_acc_cf_mean': all_steps_acc_cf_mean,
                'all_steps_acc_cf_std': all_steps_acc_cf_std,
                'all_steps_f1_cf_mean': all_steps_f1_cf_mean,
                'all_steps_f1_cf_std': all_steps_f1_cf_std,
                'all_steps_acc_x_axis': all_steps_acc_x_axis,
                'all_steps_f1_x_axis': all_steps_f1_x_axis,
                'all_steps_acc_cf_x_axis': all_steps_acc_cf_x_axis,
                'all_steps_f1_cf_x_axis': all_steps_f1_cf_x_axis,
                "dirname": dirname,
                "reverse_gt_val": reverse_gt_val,
            }
            pickle.dump(pickled_data, f)

        filtered_buf = []
        filtered_step_idx = []
        for step_i, step_item in enumerate(all_error_types):
            if len(step_item) > 0:
                _tmp_buf = [x for x in step_item if x['Not Parseable'] == 0 and x['Not Parseable (prev)'] == 0 and x[
                    'Inconsistent States Number'] == 0]
                if len(_tmp_buf) > 0:
                    filtered_buf.append(_tmp_buf)
                    filtered_step_idx.append(step_i)
        # plot the error type
        color_map = {
            # figure b, d in Figure 4
            "Hallucinated Updates (DR)": "#EEB5EB",
            "Void Action": "#c26dbc",
            "Correct Updates": "#3cacae",
            "Maintain Correct Belief": "#57DBD8",
            # figure a, c in Figure 4
            "Maintain Wrong Belief (Dirty Read)": "#ff5765",
            "Updates over Wrong Belief (Dirty Write)": "#f6c324",
            "Not Following Updates": "#8a6fdf",
            "Hallucinated Updates": "#A8E10C",
        }
        marker_map = {
            "Hallucinated Updates (DR)": "o",
            "Hallucinated Updates": "*",
            "Void Action": "s",
            "Correct Updates": "^",
            "Maintain Correct Belief": "p",
            "Maintain Wrong Belief (Dirty Read)": "h",
            "Updates over Wrong Belief (Dirty Write)": "D",
            "Not Following Updates": "v",
        }
        plot_figures(os.path.join(args.visualization_root_dir, cur_exp_name + "_incorrect_outputs.pdf"), error_keys,
                     filtered_buf, filtered_step_idx, color_map, "Analysis for Incorrect Predictions",
                     include_list=["Maintain Wrong Belief (Dirty Read)", "Updates over Wrong Belief (Dirty Write)",
                                   "Not Following Updates", "Hallucinated Updates"], marker_map=marker_map)
        plot_figures(os.path.join(args.visualization_root_dir, cur_exp_name + "_correct_outputs.pdf"), error_keys,
                     filtered_buf, filtered_step_idx, color_map, "Analysis for Correct Predictions",
                     include_list=["Hallucinated Updates (DR)", "Void Action", "Correct Updates",
                                   "Maintain Correct Belief", ], marker_map=marker_map)
