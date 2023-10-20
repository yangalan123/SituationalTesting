import argparse
import copy
import glob
import os
import pickle
import random

import loguru
import numpy as np
from SitCoT.utils import (
    process_gt_answer_response,
    get_answer_from_response,
)
from tqdm import tqdm

from APIServer import OpenAIServer
from Const import pricing
from Data.SimpleBoxOpeningEnv.action_templates import extract_arguments_from_single_action
from Data.SimpleBoxOpeningEnv.instruction_gen_pipeline import extract_num_boxes_and_num_keys, \
    extract_logic_functor_for_box_and_keys
from utils import get_answer_from_response, get_gpt_tokenizer, compose_answer_from_status


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive Debugging')
    parser.add_argument('--model_type', type=str, default="gpt-3.5-turbo")
    parser.add_argument('--process_root_dir', type=str, default="/data/chenghao/SituatioalTesing-CoT/GPT3Output")
    parser.add_argument('--output_root_dir', type=str,
                        default="/data/chenghao/SituatioalTesing-CoT/GPT3Output_InteractiveDebugging/")
    parser.add_argument('--log_root_dir', type=str,
                        default="/data/chenghao/SituatioalTesing-CoT/GPT3Output_InteractiveDebugging/{}/log_dir/")
    parser.add_argument('--accounting_only', action='store_true', help='whether to do budget accounting only without sending any requests')
    parser.add_argument("--f1", action="store_true", help="whether to use f1 as the metric")
    parser.add_argument("--ignore_reverse_gt_val", action="store_true",
                        help="whether to ignore reverse gt val (note that in early version of run.py, we did not reverse gt val, so we need to ignore it)")
    args = parser.parse_args()
    args.process_root_dir = os.path.join(args.process_root_dir, args.model_type)
    # see whether process_root_dir is valid
    if not os.path.exists(args.process_root_dir):
        raise ValueError(f"process_root_dir {args.process_root_dir} does not exist")
    # see how many directories are in process_root_dir
    dirs = glob.glob(os.path.join(args.process_root_dir, "*"))
    if len(dirs) == 0:
        raise ValueError(f"process_root_dir {args.process_root_dir} is empty")
    try:
        encoding = get_gpt_tokenizer(args.model_type)
    except:
        raise ValueError(f"Model type {args.model_type} not supported")
    args.output_root_dir = os.path.join(args.output_root_dir, args.model_type)
    os.makedirs(args.output_root_dir, exist_ok=True)
    args.log_root_dir = args.log_root_dir.format(args.model_type)
    os.makedirs(args.log_root_dir, exist_ok=True)
    args.exp_name = "running_log" if not args.accounting_only else "accounting_log"

    return args


def process_file(file, args, logger, agent: OpenAIServer):
    response = pickle.load(open(file, "rb"))
    workload = response["workload"]
    env = response["env"] if "env" in response else None
    lines = workload.split("\n")
    box_name, key_name = None, None
    num_boxes, num_keys = None, None
    logic_functor = None
    all_states = dict()
    all_histories = []
    all_prefixes = []
    for line_i, line in enumerate(lines):
        if "Instruction" in line:
            if env is None:
                # we have to use our best guess here from instruction
                # step-1: solve the number of boxes and keys first
                num_boxes, num_keys = extract_num_boxes_and_num_keys(line)
                # step-2: solve the logic functor
                logic_functor = extract_logic_functor_for_box_and_keys(line)
            else:
                num_boxes, num_keys = env.num_boxes, env.num_keys
                logic_functor = env.logic_functor
        if "Step" in line and "Step-0" not in line:
            step, box_i, key_i, _box_name, _key_name = extract_arguments_from_single_action(line, env)
            if box_name is None and key_name is None:
                box_name = _box_name
                key_name = _key_name
                for i in range(num_boxes):
                    all_states["{}-{}".format(box_name, i)] = False
                for j in range(num_keys):
                    all_states["{}-{}".format(key_name, j)] = False
                # note this is not duplicated -- this is for the step-0
                all_histories.append(copy.deepcopy(all_states))
                all_prefixes.append("\n".join(lines[:line_i]))
                continue
            assert box_name == _box_name and key_name == _key_name
            all_states["{}-{}".format(box_name, box_i)] = True
            all_states["{}-{}".format(key_name, key_i)] = True
            all_histories.append(copy.deepcopy(all_states))
            all_prefixes.append("\n".join(lines[:line_i + 1]))

    assert num_boxes is not None and num_keys is not None and logic_functor is not None
    reconstructed_gt_answer = compose_answer_from_status(all_states, box_name, key_name, num_boxes, num_keys,
                                                         logic_functor) + "."
    gt_answer = response['gt_answer']
    _pseudo_response = copy.deepcopy(response)

    _pseudo_response['choices'][0]['message']['content'] = reconstructed_gt_answer
    _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, _pseudo_response, logger,
                                                                             reverse_gt_val=args.reverse_gt_val,
                                                                             f1=args.f1)
    assert (args.f1 and _stat_acc['f1'] == 1) or (
            not args.f1 and _stat_acc == 1), "gt_answer not reconstructed correctly: \ngt_answer: {}\nreconstructed: {}".format(
        gt_answer, reconstructed_gt_answer)
    _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, response, logger,
                                                                             reverse_gt_val=args.reverse_gt_val,
                                                                             f1=args.f1)
    ret_new_workload = []
    all_budgets = []
    if (not args.f1 and _stat_acc < 1) or (args.f1 and _stat_acc['f1'] < 1):
        for history_i in range(1, len(all_histories)):
            cur_history = all_histories[history_i]
            cur_prefix = all_prefixes[history_i]
            cur_answer = compose_answer_from_status(cur_history, box_name, key_name, num_boxes, num_keys, logic_functor)
            cur_question = compose_answer_from_status(cur_history, box_name, key_name, num_boxes, num_keys,
                                                      logic_functor, is_answer=False)
            new_prefix = "\n".join([cur_prefix, f"Question: {cur_question}", "Answer: "])
            ret_new_workload.append([new_prefix, cur_answer])
            accounting_message_workload = new_prefix + cur_answer
            if agent.is_chat_model():
                accounting_message_workload = agent.prepare_chat_workload(accounting_message_workload)
            num_tokens = agent.send_accounting(accounting_message_workload)
            pricing_type = pricing[args.model_type]
            all_budgets.append(num_tokens / pricing_type[1] * pricing_type[0])
    return ret_new_workload, all_budgets, _stat_acc


if __name__ == '__main__':
    # write an argumentparser to handle process_root_dir, output_dir, and other arguments
    args = parse_args()
    logger = loguru.logger
    logger.add(f"{args.log_root_dir}/" + args.exp_name + ".log", mode='w')
    agent = OpenAIServer(args.model_type)
    all_budgents = 0
    example_flag = True
    example_count = 5
    for exp_path in glob.glob(args.process_root_dir + "/*NL_func_NL_arg*"):
        # for exp_path in glob.glob(args.process_root_dir + "/*"):
        logger.info("processing exp path: {}".format(exp_path))
        response_pickle_path = os.path.join(args.output_root_dir, args.model_type, os.path.basename(exp_path),
                                            "responses_dir")
        if not os.path.exists(response_pickle_path):
            os.makedirs(response_pickle_path)
        try:
            new_workload = []
            new_budgets = []
            stat_acc = []
            # deal with truth flipped case
            args.reverse_gt_val = True if "cf" in exp_path and not args.ignore_reverse_gt_val else False
            filenames = glob.glob(os.path.join(exp_path, "*"))
            for filename in tqdm(filenames, position=0, leave=True):
                target_filename = os.path.join(response_pickle_path, os.path.basename(filename) + ".pkl")
                if os.path.exists(target_filename):
                    try:
                        _tmp_data = pickle.load(open(target_filename, "rb"))
                        logger.info(
                            f"detected processing file {filename} has been intereactly debugged and output is at {target_filename}, skipping..")
                        continue
                    except:
                        pass
                _new_workload, _new_budgets, _stat_acc = process_file(filename, args, logger, agent)
                new_workload.extend(_new_workload)
                new_budgets.extend(_new_budgets)
                if isinstance(_stat_acc, dict):
                    stat_acc.append(_stat_acc['f1'])
                else:
                    stat_acc.append(_stat_acc)
                file_em_acc = 0
                file_stat_acc = 0
                file_parseable_acc = 0
                file_tv_all = []
                file_responses = []
                file_counter = 0
                for _item_i, _item in enumerate(tqdm(_new_workload, position=1, leave=True)):
                    _workload, _gt_answer = _item
                    try:
                        response = agent.send(_workload, logprobs=5)
                    except Exception as e:
                        # network issue, too long context, etc.
                        logger.warning(e)
                        continue
                    file_counter += 1
                    sample_answer = get_answer_from_response(response)
                    _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(_gt_answer, response,
                                                                                             logger,
                                                                                             reverse_gt_val=args.reverse_gt_val)
                    file_em_acc += _em_acc
                    file_stat_acc += _stat_acc
                    file_tv_all += _tv_all
                    file_parseable_acc += _parseable_acc
                    response['gt_answer'] = _gt_answer
                    response['workload'] = _workload
                    response['source_filename'] = filename
                    response['debug_item_no'] = _item_i
                    file_responses.append(response)

                file_em_acc /= (file_counter + 1e-10)
                file_stat_acc /= (file_counter + 1e-10)
                file_parseable_acc /= (file_counter + 1e-10)
                pickle.dump([file_em_acc, file_stat_acc, file_tv_all, file_parseable_acc, file_responses],
                            open(target_filename, "wb"))

            logger.info("new workload size: {}".format(len(new_workload)))
            logger.info("new budgets: {}".format(sum(new_budgets)))
            logger.info("stat acc: {}".format(np.mean(stat_acc)))
            if example_flag and example_count > 0 and len(new_workload) > 0:
                logger.info("example workload: {}".format(random.sample(new_workload, 1)[0]))
                example_count -= 1
            all_budgents += sum(new_budgets)
        except Exception as e:
            logger.info("error in processing exp path: {}".format(exp_path))
            logger.info("error: {}".format(e))
    logger.info("all budgets: {}".format(all_budgents))

