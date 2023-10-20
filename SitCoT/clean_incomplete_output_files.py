# Author: Chenghao Yang
# As OpenAI API can temporarily lose connection or refuse to response, we need to clean up the incomplete output files
# and rerun them. This script is used to identify the incomplete output files and suggest rerunning them.
import glob
import os
import pickle
import shutil

import loguru
from SitCoT.utils import process_state_representation

if __name__ == '__main__':
    logger = loguru.logger
    os.makedirs("../logs", exist_ok=True)
    logger.add("../logs/cleaning_log_{time}.txt")
    all_truncate_file_counter = 0
    do_not_necessarilly_filter_out_due_to_em = 0
    delete_source_file_flag = True
    suggested_rerunning_dirs = []
    for model_name in ["text-davinci-003", "code-davinci-002"]:
        for output_dir in glob.glob(f"../GPT3Output/{model_name}/*"):
            exp_name = os.path.basename(output_dir)
            logger.info(f"processing {output_dir}")
            path_truncate_file_counter = 0
            partial_correct_path = os.path.join(output_dir, "truncate_so_far_all_correct")
            os.makedirs(partial_correct_path, exist_ok=True)
            partial_incorrect_path = os.path.join(output_dir, "truncate_so_far_already_incorrect")
            os.makedirs(partial_incorrect_path, exist_ok=True)

            all_cached_files_paths = glob.glob(os.path.join(output_dir, "*.pkl"))
            for path in all_cached_files_paths:
                possibly_delete_flag = False
                with open(path, "rb") as f_in:
                    response = pickle.load(f_in)
                    gt_answer = response["gt_answer"]
                    finish_reason = response['choices'][0]['finish_reason']
                    sample_answer = response["choices"][0]['text']
                    if finish_reason == "length":
                        gt_status, gt_args, gt_vals = process_state_representation(gt_answer, logger=logger)
                        try:
                            pred_status, pred_args, pred_vals = process_state_representation(sample_answer,
                                                                                             logger=logger)
                            pred_status_keys = set(pred_status.keys())
                            gt_status_keys = set(gt_status.keys())
                            if len(pred_status_keys) < len(gt_status_keys):
                                basename = os.path.basename(path)
                                possibly_delete_flag = True
                                all_truncate_file_counter += 1
                                path_truncate_file_counter += 1
                                logger.info(
                                    f"Identify {all_truncate_file_counter}-th truncated file, this is the {path_truncate_file_counter} / {len(all_cached_files_paths)} under the same setting")
                                intersection_count = 0
                                intersection_keys = set(gt_status.keys()) & set(pred_status.keys())
                                for key in intersection_keys:
                                    if gt_status[key] == pred_status[key]:
                                        intersection_count += 1
                                precision = intersection_count / len(gt_status.keys())
                                recall = intersection_count / len(pred_status)
                                if recall < 1:
                                    do_not_necessarilly_filter_out_due_to_em += 1
                                    logger.info(
                                        f"for this file, we do not necessarily need to filter it our for EM accuracy catch as precision={precision}, recall={recall}")
                                    shutil.copy(path, os.path.join(partial_incorrect_path, basename))
                                    logger.info(f"backup this file at {os.path.join(partial_incorrect_path, basename)}")
                                else:
                                    shutil.copy(path, os.path.join(partial_correct_path, basename))
                                    logger.info(f"backup this file at {os.path.join(partial_correct_path, basename)}")

                        except AssertionError as e:
                            logger.info(f"parsing error happening: {e}")
                            logger.info(f"Parsing ERROR Path: {path}")
                            logger.info(f"Parsing ERROR GT-ANSWER: {gt_answer}")
                            logger.info(f"Parsing ERROR PRED-ANSWER: {sample_answer}")
                if possibly_delete_flag and delete_source_file_flag:
                    os.remove(path)
                    logger.info(f"successfully remove file as it was somewhat truncated: {path}")
            if path_truncate_file_counter / (len(all_cached_files_paths) + 1e-7) > 0.6:
                suggested_rerunning_dirs.append((output_dir, path_truncate_file_counter, len(all_cached_files_paths)))
    suggested_rerunning_dirs.sort(key=lambda x: x[1] / x[2], reverse=True)
    for line in suggested_rerunning_dirs:
        logger.warning(f"We suggest rerunning {line}")
