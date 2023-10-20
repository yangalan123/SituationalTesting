# plot figure 2 in the paper, merge the result for intermediate state probing over both SL and NL
import pickle

import matplotlib.pyplot as plt


def plot_data(data, suffix, colors, markers):
    reverse_gt_val, all_steps_acc_cf_x_axis, all_steps_acc_cf_mean, all_steps_acc_cf_std, all_steps_f1_cf_x_axis, all_steps_f1_cf_mean, all_steps_f1_cf_std, all_steps_acc_x_axis, all_steps_acc_mean, all_steps_acc_std, all_steps_f1_x_axis, all_steps_f1_mean, all_steps_f1_std \
        = data['reverse_gt_val'], data['all_steps_acc_cf_x_axis'], data['all_steps_acc_cf_mean'], data[
        'all_steps_acc_cf_std'], data['all_steps_f1_cf_x_axis'], data['all_steps_f1_cf_mean'], data[
        'all_steps_f1_cf_std'], data['all_steps_acc_x_axis'], data['all_steps_acc_mean'], data['all_steps_acc_std'], \
    data['all_steps_f1_x_axis'], data['all_steps_f1_mean'], data['all_steps_f1_std']
    linewidth = 8
    markersize = 50
    linestyle = "-" if suffix == "SL" else ":"

    if reverse_gt_val:
        plt.errorbar(all_steps_acc_cf_x_axis, all_steps_acc_cf_mean, yerr=all_steps_acc_cf_std, ecolor="black",
                     capsize=5, label=f"Step-EM ({suffix})", color=colors[0], linewidth=linewidth, marker=markers[0],
                     markersize=markersize, linestyle=linestyle)
        plt.errorbar(all_steps_f1_cf_x_axis, all_steps_f1_cf_mean, yerr=all_steps_f1_cf_std, ecolor="black",
                     capsize=5, label=f"State-EM ({suffix})", color=colors[1], linewidth=linewidth, marker=markers[1],
                     markersize=markersize, linestyle=linestyle)
    else:
        plt.errorbar(all_steps_acc_x_axis, all_steps_acc_mean, yerr=all_steps_acc_std, ecolor="black",
                     capsize=5, label=f"Step-EM ({suffix})", color=colors[0], linewidth=linewidth, marker=markers[0],
                     markersize=markersize, linestyle=linestyle)
        plt.errorbar(all_steps_f1_x_axis, all_steps_f1_mean, yerr=all_steps_f1_std, ecolor="black",
                     capsize=5, label=f"State-EM ({suffix})", color=colors[1], linewidth=linewidth, marker=markers[1],
                     markersize=markersize, linestyle=linestyle)


if __name__ == '__main__':
    root_dir = "../GPT3Output_InteractiveDebugging/gpt-3.5-turbo/visualization_dir_final"
    shot_num = 2
    for shot_num in [2, 5]:
        exp_name_1 = f"50sample_10boxes_irreg_func_irreg_arg_{shot_num}shot_init"
        exp_name_2 = f"50sample_10boxes_NL_func_NL_arg_{shot_num}shot_init"
        suffix_1 = "SL"
        suffix_2 = "NL"
        data_1 = pickle.load(open(f"{root_dir}/{exp_name_1}_vis_data.pickle", "rb"))
        data_2 = pickle.load(open(f"{root_dir}/{exp_name_2}_vis_data.pickle", "rb"))
        # resize the figure
        plt.gcf().set_size_inches(30, 18)
        # set font size
        font_size = 62
        plt.rcParams.update({'font.size': font_size})
        plt.title(f"Intermediate State Probing ({shot_num}-shot)")
        plt.xlabel("Steps")
        plt.ylabel("Acc")
        # make it look nicer (camera ready phase)
        markers = ["*", "^", "o", "s"]
        plot_data(data_1, suffix_1, ["#2FF3E0", "#F8D210"], markers[:2])
        plot_data(data_2, suffix_2, ["#FA26A0", "#603F8B"], markers[2:])
        plt.subplots_adjust(bottom=0.4)
        plt.axvline(x=shot_num - 1, color='purple', linestyle='--', lw=8)
        # Annotate Phase-1 and Phase-2
        plt.text(shot_num / 2, -0.2, 'In-Context', color='black',
                 horizontalalignment='right', verticalalignment='top',
                 fontsize=font_size, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

        plt.text(shot_num * 1.5, -0.2, 'Testing', color='black',
                 horizontalalignment='left', verticalalignment='top',
                 fontsize=font_size, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        for spine in plt.gca().spines.values():
            spine.set_linewidth(4)
            spine.set_edgecolor('black')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=2)
        plt.savefig(f"{root_dir}/{exp_name_1}_{exp_name_2}_merge.pdf")
        plt.show()
        plt.clf()
