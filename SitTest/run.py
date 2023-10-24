import argparse
import glob
import json
import os
import pickle
import random
import shutil
from time import sleep

import loguru
import numpy as np
import torch
from SitTest.Data.SimpleBoxOpeningEnv.action_templates import get_action_templates
from SitTest.Data.SimpleBoxOpeningEnv.envs import generateSimpleBoxOpenningSample
from SitTest.Data.SimpleBoxOpeningEnv.instruction_gen_pipeline import generate_initial_instruction_pipeline
from SitTest.Data.SimpleBoxOpeningEnv.logic_functor import logic_functor, generate_random_functor
from SitTest.utils import (
    process_box_key_reference,
    process_gt_answer_response,
    generateRandomWord,
    get_answer_from_response,
    generate_verbose_sentence
)
from tqdm import trange

from SitTest.APIServer import OpenAIServer
from SitTest.Const import (pricing)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ApplyActionPerturbation(step_i, args, action_seqs, corpus):
    processed_action_seqs = [action_seqs[0], ] + [x.split(f"Step-{step_i + 1}: ")[1] for x in action_seqs[1:]]
    if args.action_perturbation in ['verbose', "combined"]:
        processed_action_seqs = [seq.replace("\n", generate_verbose_sentence(corpus) + "\n") for seq in
                                 processed_action_seqs]
    if args.action_perturbation in ['non-atomic', "combined"]:
        return " ".join([x.replace("\n", "") for x in processed_action_seqs]) + "\n"
    else:
        assert len(action_seqs) == 1
        return processed_action_seqs[0]


def parse_args():
    parser = argparse.ArgumentParser(description="SitTest Testing Environment Setting.")
    parser.add_argument("--seed", type=int, default=1111, help="random_seed")
    parser.add_argument("--num_box", type=int, default=20, help="number of boxes", required=True)
    parser.add_argument("--num_key", type=int, default=20, help="number of keys", required=True)
    parser.add_argument("--num_steps", type=int, default=-1,
                        help="fixed number of steps, by default use sample from U(1, min(num_boxes, num_key)-1)")
    parser.add_argument("--all_sample_num", type=int, default=200, help="number of testing samples", required=True)
    parser.add_argument("--shots_num", type=int, default=1, help="number of shots given", required=True)
    parser.add_argument("--num_logic_functors", type=int, default=5, help="number of random logic functor candidates", )
    parser.add_argument("--max_functor_length", type=int, default=10, help="length of each functor name candidate", )
    parser.add_argument("--max_item_length", type=int, default=10, help="length of each item name candidate", )
    parser.add_argument("--output_root_dir", type=str, default="../GPT3Output", help="path to store output files", )
    parser.add_argument("--instruction_path", type=str, default="Data/instructions.json",
                        help="path to load instructions from")
    parser.add_argument("--model", type=str, default="text-davinci-003", help="the GPT3 variant you want to use")
    parser.add_argument("--with_init", type=str2bool, nargs='?', const=True, default=False,
                        help="whether add initialization in in-context samples")
    parser.add_argument("--with_memorization", type=str2bool, nargs='?', const=True, default=False,
                        help="whether testing memorization")
    parser.add_argument("--with_irreg_func", type=str2bool, nargs='?', const=True, default=False,
                        help="whether using irregular functor")
    parser.add_argument("--with_irreg_arg", type=str2bool, nargs='?', const=True, default=False,
                        help="whether using irregular argument")
    parser.add_argument("--with_hints", type=str2bool, nargs='?', const=True, default=False,
                        help="whether using hints in the middle")
    parser.add_argument("--with_incomplete_supervision", type=str2bool, nargs='?', const=True, default=False,
                        help="whether using incomplete supervision in k-shot")
    parser.add_argument("--incomplete_sup_state_sample_ratio", type=float, default=0.5,
                        help="probability of dropping a logic state in the incomplete_supervision scenario", )
    parser.add_argument("--with_incomplete_query", type=str2bool, nargs='?', const=True, default=False,
                        help="whether query for incomplete state set in testing time")
    parser.add_argument("--instruction_counterfactual_nl", type=str2bool, nargs='?', const=True, default=False,
                        help="whether add counterfactual NL in the instruction part")
    parser.add_argument("--instruction_counterfactual_logic", type=str2bool, nargs='?', const=True, default=False,
                        help="whether add counterfactual logic in the instruction part")
    parser.add_argument("--query_state_sample_ratio", type=float, default=1,
                        help="probability of dropping a query for logic state in the incomplete_supervision scenario", )
    parser.add_argument("--debug", type=str2bool, nargs='?', const=True, default=False, help="whether enter debug mode")
    parser.add_argument("--accounting_only", type=str2bool, nargs='?', const=True, default=False,
                        help="whether only do accounting")
    parser.add_argument("--use_llama", type=str2bool, nargs='?', const=True, default=False, help="whether use llama")
    parser.add_argument("--llama_model_name", type=str, default="vicuna", help="the llama model name")
    parser.add_argument("--llama_server_url", type=str, default="http://localhost:8000/v1", )
    parser.add_argument("--chat_style_probing", type=str2bool, nargs='?', const=True, default=False,
                        help="whether use chat style probing")
    parser.add_argument("--action_style_probing", type=str2bool, nargs='?', const=True, default=False,
                        help="whether use action probing")
    parser.add_argument("--action_perturbation", type=str, default="verbose",
                        help="what kind of perturbation you want to apply?")
    parser.add_argument("--prerun_max_steps", type=int, default=0, help="max steps for pre-run")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_root_dir = os.path.join(args.output_root_dir, args.model)
    assert args.num_box >= args.shots_num + 1, "at least we need to have one test"
    if args.with_incomplete_supervision:
        assert args.incomplete_sup_state_sample_ratio >= 0 and args.incomplete_sup_state_sample_ratio <= 1, "the sampling ratio for incomplete supervision should be within [0,1]"
    if args.with_incomplete_query:
        assert args.query_state_sample_ratio >= 0 and args.query_state_sample_ratio <= 1, "the sampling ratio for query should be within [0,1]"
    if args.num_steps > 0:
        assert args.num_steps <= min(args.num_box,
                                     args.num_key), "you cannot specify steps larger than min(num_box, num_key)"
    args.exp_name = f"{args.all_sample_num}sample_{args.num_box}boxes" \
                    f"{'_fixed_steps_' + str(args.num_steps) if args.num_steps > 0 else ''}" \
                    f"{'_irreg_func' if args.with_irreg_func else '_NL_func'}{'_irreg_arg' if args.with_irreg_arg else '_NL_arg'}" \
                    f"_{args.shots_num}shot" \
                    f"{'_incomp_sup_' + str(args.incomplete_sup_state_sample_ratio) if args.with_incomplete_supervision else ''}" \
                    f"{'_incomp_query_' + str(args.query_state_sample_ratio) if args.with_incomplete_query else ''}" \
                    f"{'_init' if args.with_init else ''}" \
                    f"{'_memorization' if args.with_memorization else ''}{'_hint' if args.with_hints else ''}{'_incomp_supervision' if args.with_incomplete_supervision else ''}" \
                    f"{'_cf_nl' if args.instruction_counterfactual_nl else ''}{'_cf_logic' if args.instruction_counterfactual_logic else ''}" \
                    f"{'_llama' if args.use_llama else ''}{'_{}'.format(args.llama_model_name) if args.use_llama else ''}" \
                    f"{'_chat_style_probing' if args.chat_style_probing else ''}" \
                    f"{'_action_style_probing_' + args.action_perturbation if args.action_style_probing else ''}" \
                    f"{'_prerun_' + str(args.prerun_max_steps) if args.prerun_max_steps > 0 else ''}" \
        # +1 because we need at least one test
    args.min_steps = args.shots_num + 1
    if args.with_init:
        assert args.shots_num >= 1, "if you use initialization, then at least you already have one shot"
        args.shots_num -= 1
    args.reverse_gt_val = args.instruction_counterfactual_logic or args.instruction_counterfactual_nl
    if args.action_style_probing:
        assert args.action_perturbation in ['verbose', "non-atomic",
                                            "combined"], "action perturbation should be one of ['verbose', 'non-atomic', 'combined']"

    return args


if __name__ == '__main__':
    args = parse_args()
    sample_num = args.all_sample_num
    DEBUG_MODE = args.debug
    os.makedirs(args.output_root_dir, exist_ok=True)
    logger = loguru.logger
    if not DEBUG_MODE:
        os.makedirs(f"../logs/{args.model}", exist_ok=True)
        logger.add(f"../logs/{args.model}/" + args.exp_name + ".log")
    else:
        os.makedirs(f"../logs_debug/{args.model}", exist_ok=True)
        logger.add(f"../logs_debug/{args.model}/" + args.exp_name + ".log")
    logger.info(f"Now Running Experiment: {args.exp_name}")
    logger.info(f"Environment Setup: \n{json.dumps(vars(args), indent=4)}")
    output_dir = os.path.join(args.output_root_dir, args.exp_name)
    overall_token_num = 0
    if args.accounting_only:
        output_dir = os.path.join(output_dir, "accounting_only")
    os.makedirs(output_dir, exist_ok=True)
    # copy the current program to the output dir
    shutil.copy(__file__, output_dir)
    shutil.copy("APIServer.py", output_dir)

    # get instructions
    initial_instruction = generate_initial_instruction_pipeline(args=args, logic_functor=logic_functor)

    sample_env, the_key_to_the_door, judge_condition = generateSimpleBoxOpenningSample(
        num_box=args.num_box,
        num_key=args.num_key,
        logic_functor=logic_functor,
    )
    action_templates = get_action_templates(sample_env)

    logic_functor_candidates = generate_random_functor(args)
    if args.use_llama:
        # vicuna replicate OpenAI GPT-3 API
        # see https://github.com/lm-sys/FastChat/blob/main/docs/openai_api.md
        gpt3_agent = OpenAIServer(args.llama_model_name, api_base=args.llama_server_url, is_llama=True)
    else:
        gpt3_agent = OpenAIServer(args.model)

    if args.action_style_probing:
        from nltk.corpus import brown

        corpus = brown.sents()
        sentences = [" ".join(s).strip() for s in corpus if "\n" not in s]
        corpus = random.sample(sentences, 1000)
    else:
        corpus = None

    em_acc = 0
    stat_acc = 0
    parseable_acc = 0
    tv_all = []
    # loading already downloaded data
    all_cached_files_paths = glob.glob(os.path.join(output_dir, "*.pkl"))
    for path in all_cached_files_paths[:args.all_sample_num]:
        response = pickle.load(open(path, "rb"))
        gt_answer = response["gt_answer"]
        if not args.accounting_only:
            _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, response, logger,
                                                                                     reverse_gt_val=args.reverse_gt_val)
            em_acc += _em_acc
            stat_acc += _stat_acc
            tv_all += _tv_all
            parseable_acc += _parseable_acc
        else:
            response = dict()
            try:
                token_num = gpt3_agent.send_accounting(response['workload'])
            except Exception as e:
                logger.warning(e)
                continue
            response['token_num'] = token_num
            overall_token_num += token_num
        sample_num -= 1
    ####

    if len(all_cached_files_paths) > 0:
        if not args.accounting_only:
            logger.info(
                f"(cached data) sample_num: {args.all_sample_num - sample_num}, em_acc: {em_acc / len(all_cached_files_paths)}, stat_acc: {stat_acc / len(all_cached_files_paths)},\n, parseable_acc: {parseable_acc / len(all_cached_files_paths)}"
            )
        else:
            logger.info(
                f"Estimated Price: {overall_token_num / pricing[gpt3_agent.model][1] * pricing[gpt3_agent.model][0]} USD (token_num: {overall_token_num})")

    for sample_i in trange(sample_num):
        if (sample_i + 1) % 5 == 0 and "code" in args.model:
            sleep(30)
        # environment initialize
        action_template = action_templates[0]
        sample_env.init()
        # preparing perturbation
        if args.num_steps <= 0:
            num_steps = random.sample(range(args.min_steps, min(args.num_box, args.num_key)), 1)[0]
        else:
            num_steps = args.num_steps
        if args.with_irreg_func:
            _functor = random.sample(logic_functor_candidates, 1)[0]
            sample_env.logic_functor = _functor
            new_instruction = initial_instruction.replace(logic_functor["box"], _functor["box"])
            new_instruction = new_instruction.replace(logic_functor["key"], _functor["key"])
        else:
            new_instruction = initial_instruction
        if args.with_irreg_arg:
            box_reference = generateRandomWord(args.max_item_length)
            key_reference = generateRandomWord(args.max_item_length)
            rename_argument = lambda x: process_box_key_reference(x, box_reference, key_reference)
        else:
            # do-nothing
            rename_argument = lambda x: x
        arg_renamed_new_instruction = rename_argument(new_instruction)
        if args.chat_style_probing:
            workload = [{"role": "user", "content": f"Instructions: {arg_renamed_new_instruction}"}]
        else:
            workload = f"Instructions: {arg_renamed_new_instruction}\n"
        if args.with_init:
            if args.prerun_max_steps > 0:
                # at least, we need to leave enough action spaces for few-shot plus 1 test sample
                # note that shots num has already been subtracted by 1
                prerun_steps = random.sample(range(1, min(min(sample_env.num_boxes,
                                                              sample_env.num_keys) - args.shots_num - 1,
                                                          args.prerun_max_steps)), 1)[0]
                pre_actions = []
                for _ in range(prerun_steps):
                    all_available_boxes = sample_env.get_available_boxes()
                    if len(all_available_boxes) == 0:
                        raise ValueError("No available boxes for pre-running, please check the environment setting.")
                    box_i = random.sample(all_available_boxes, 1)[0]
                    key_i = torch.argmax(sample_env.box2keys[box_i]).cpu().item()
                    sample_env.update(box_id=box_i, key_id=key_i, by_id=True, known_states_update=False)
                    pre_actions.append({"box_id": box_i, "key_id": key_i})
                sample_env.pre_run_history = pre_actions

            question, answer = sample_env.representation(reverse_value=args.reverse_gt_val)
            if args.chat_style_probing:
                # add step-0
                workload.append({"role": "user", "content": f"Step-0: Initialization. Do nothing."})
                workload.append({"role": "user", "content": f"Question: {rename_argument(question)}"})
                workload.append({"role": "assistant", "content": f"Answer: {rename_argument(answer)}"})
            else:
                workload += f"Step-0: Initialization. Do nothing. \n" \
                            f"Question: {rename_argument(question)}\n" \
                            f"Answer: {rename_argument(answer)}\n"
        step_i = 0

        # for step_i in range(num_steps):
        # as we can do action style probing, we cannot use range -- we may skip some steps because we will combine them together
        action_buf = []
        need_to_wait_action_num = 0
        while step_i < num_steps:
            # run perturbation
            all_available_boxes = sample_env.get_available_boxes()
            if len(all_available_boxes) == 0:
                break
            box_i = random.sample(all_available_boxes, 1)[0]
            key_i = torch.argmax(sample_env.box2keys[box_i]).cpu().item()
            if step_i >= args.shots_num and args.action_style_probing:
                # even if we need to wait for some actions in non-atomic updates, all actions has been "factually" asserted
                # so len(all_available_boxes) is a real indicator of how many actions we can do
                if args.action_perturbation in ['non-atomic', 'combined']:
                    if need_to_wait_action_num == 0:
                        # only resample need_to_wait_action_num while 1) we are doing action probing 2) previous perturbation is done
                        max_operation_num = len(all_available_boxes)
                        need_to_wait_action_num = random.sample(range(1, max_operation_num + 1), 1)[0]
                else:
                    need_to_wait_action_num = 1
            else:
                # otherwise, always do no buffering
                need_to_wait_action_num = 0

            # add step: step_i
            if args.chat_style_probing:
                if not args.action_style_probing:
                    workload.append(
                        {"role": "user", "content": f"{rename_argument(action_template(box_i, key_i, step_i + 1))}"})
                else:
                    if step_i >= args.shots_num:
                        action_buf.append(rename_argument(action_template(box_i, key_i, step_i + 1)))
                        need_to_wait_action_num -= 1
                        if need_to_wait_action_num == 0:
                            workload.append(
                                {"role": "user", "content": ApplyActionPerturbation(step_i, args, action_buf, corpus)})
                    else:
                        workload.append({"role": "user",
                                         "content": f"{rename_argument(action_template(box_i, key_i, step_i + 1))}"})
            else:
                if not args.action_style_probing:
                    workload += f"{rename_argument(action_template(box_i, key_i, step_i + 1))}"
                else:
                    if step_i >= args.shots_num:
                        action_buf.append(rename_argument(action_template(box_i, key_i, step_i + 1)))
                        need_to_wait_action_num -= 1
                        if need_to_wait_action_num == 0:
                            workload += ApplyActionPerturbation(step_i, args, action_buf, corpus)
                    else:
                        workload += f"{rename_argument(action_template(box_i, key_i, step_i + 1))}"
            # do not update known states at the last step_i, as it is the intended test
            if len(action_buf) == 0:
                sample_env.update(box_id=box_i, key_id=key_i, by_id=True, known_states_update=step_i != num_steps - 1)
            else:
                final_flag = step_i == num_steps - 1 or len(sample_env.get_available_boxes()) == need_to_wait_action_num
                # either we have used up all steps, or we WILL use up all available boxes after this update
                # both situations means we have reached the end
                sample_env.update(box_id=box_i, key_id=key_i, by_id=True, known_states_update=not final_flag)
                if not final_flag and need_to_wait_action_num == 0:
                    # if we are not at the end, we need to regroup the last n items
                    history_len = len(list(sample_env.updated_history.values())[0])
                    sample_env.regroup_history(history_len - len(action_buf), history_len)

            if step_i < args.shots_num:
                # step_i starting from 0, this is not releated to whether using init or not as this is just a counter variable
                question, answer = sample_env.representation(sample_states=args.with_incomplete_supervision,
                                                             sample_ratio=args.incomplete_sup_state_sample_ratio,
                                                             reverse_value=args.reverse_gt_val)
                if args.chat_style_probing:
                    workload.append({"role": "user", "content": f"Question: {rename_argument(question)}"})
                    workload.append({"role": "assistant", "content": f"Answer: {rename_argument(answer)}"})
                else:
                    workload += f"Question: {rename_argument(question)}\n" \
                                f"Answer: {rename_argument(answer)}\n"
            if need_to_wait_action_num <= 0:
                action_buf = []
                step_i += 1
        # note here: we do not flip the gt_answer, no matter what reverse_gt_val is -- that is because we will later flip the criterion when given args.reverse_gt_val
        # it is kind of a weird logic, I know, and I plan to change it later
        question, gt_answer = sample_env.representation(hints_on_known_states=args.with_hints,
                                                        sample_states=args.with_incomplete_query,
                                                        sample_ratio=args.query_state_sample_ratio,
                                                        )
        if args.chat_style_probing:
            workload.append({"role": "user", "content": f"Question: {rename_argument(question)}"})
            # to be honest, our only bet is ChatGPT will learn to say sth starting with "Answer: "
        else:
            workload += f"Question: {rename_argument(question)}\n" \
                        f"Answer: "
        gt_answer = rename_argument(gt_answer)
        if DEBUG_MODE:
            logger.info("----workload----- \n{}".format(workload, ))
            logger.info("----gt_answer----- \n" + gt_answer)
            logger.info("----env.updated_history----- \n" + str(sample_env.updated_history))
            exit()
        if not args.accounting_only:
            try:
                if args.use_llama:
                    response = gpt3_agent.send(workload, logprobs=5, max_tokens=650)
                else:
                    response = gpt3_agent.send(workload, logprobs=5)
            except Exception as e:
                # network issue, too long context, etc.
                logger.warning(e)
                continue
            sample_answer = get_answer_from_response(response)
            _em_acc, _stat_acc, _tv_all, _parseable_acc = process_gt_answer_response(gt_answer, response, logger,
                                                                                     reverse_gt_val=args.reverse_gt_val)
            em_acc += _em_acc
            stat_acc += _stat_acc
            tv_all += _tv_all
            parseable_acc += _parseable_acc
        else:
            response = dict()
            try:
                token_num = gpt3_agent.send_accounting(workload)
            except Exception as e:
                logger.warning(e)
                continue
            response['token_num'] = token_num
            overall_token_num += token_num
        response['gt_answer'] = gt_answer
        if args.chat_style_probing:
            response['workload'] = "\n".join([x['content'] for x in workload])
            response['chat-style-workload'] = workload
        else:
            response['workload'] = workload
        response['env'] = sample_env
        pickle.dump(response, open(os.path.join(output_dir, f"{response['id']}.pkl"), "wb"))
    if not args.accounting_only:
        logger.info(
            f"sample_num: {args.all_sample_num - sample_num}, em_acc: {em_acc / args.all_sample_num}, stat_acc: {stat_acc / args.all_sample_num}, "
            f"parseable_success_rate: {parseable_acc / args.all_sample_num}\n"
            f"mean_tv: {np.mean(tv_all)}, std_tv: {np.std(tv_all)}"
        )
    else:
        logger.info(
            f"Estimated Price: {overall_token_num / pricing[gpt3_agent.model][1] * pricing[gpt3_agent.model][0]} USD (token_num: {overall_token_num})")
    if "code" in args.model:
        # for codex free version, it has a limit on request rates, so we need to slow down a bit
        sleep(30)
