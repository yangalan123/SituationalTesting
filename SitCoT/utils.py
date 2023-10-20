import copy
import traceback
import tiktoken
import random
import string
import re


def compose_answer_from_status(all_states, box_name, key_name, num_boxes, num_keys, logic_functor, is_answer=True,
                               give_hints=False, hint_states=None):
    buf = []
    if not give_hints:
        for i in range(num_boxes):
            buf.append("{}({}-{})={}".format(logic_functor['box'], box_name, i,
                                             all_states["{}-{}".format(box_name, i)] if is_answer else "?"))
        for i in range(num_keys):
            buf.append("{}({}-{})={}".format(logic_functor['key'], key_name, i,
                                             all_states["{}-{}".format(key_name, i)] if is_answer else "?"))
    else:
        assert not is_answer, "is_answer and give_hints cannot be both true"
        for i in range(num_boxes):
            buf.append("{}({}-{})={}".format(logic_functor['box'], box_name, i,
                                             all_states["{}-{}".format(box_name, i)] if "{}-{}".format(box_name,
                                                                                                       i) in hint_states else "?"))
        for i in range(num_keys):
            buf.append("{}({}-{})={}".format(logic_functor['key'], key_name, i,
                                             all_states["{}-{}".format(key_name, i)] if "{}-{}".format(key_name,
                                                                                                       i) in hint_states else "?"))
    return ".".join(buf) if is_answer else "".join(buf)


def process_box_key_reference(string, box_reference, key_reference):
    return string.replace("BOX", box_reference).replace("KEY", key_reference)


def process_state_representation(string: str, reverse_value=False, logger=None, is_llama=False):
    status = {}
    arguments = []
    values = []
    ans = copy.copy(string).strip()
    if ans.startswith("Answer: ") or ans.startswith("GT Answer: "):
        ans = ans[ans.index("Answer: ") + len("Answer: "):]
    if "\n" in ans:
        ans = ans.split("\n")[0]

    reg_prefix = "([a-zA-Z0-9]+)\(([a-zA-Z0-9]+-\d)\)=(True|true|False|false)"
    if is_llama:
        # allow zero or more spaces between = and value
        reg_prefix = "([a-zA-Z0-9]+)\(([a-zA-Z0-9]+-\d)\)\s*=\s*(True|true|False|false)"
    props = re.findall(reg_prefix, ans)
    assert len(props) > 0, f"no parseable response found: \n{string}"
    for prop in props:
        functor, argument, value = prop
        # assert argument not in status, "prop: {}\nargument: {}\nstatus{}\n".format(prop, argument, status)
        if argument in status:
            if status[argument] == value.lower():
                continue
            else:
                if logger is not None:
                    logger.warning(f"argument: {argument} has multiple values: {status[argument]} and {value}")
                assert False, "argument: {} has multiple values: {} and {}\nProcessed Answer:{}\nreverse_value:{}".format(
                    argument, status[argument], value, string, reverse_value)
        value = value.lower()
        if reverse_value:
            if value == "false":
                value = "true"
            elif value == "true":
                value = 'false'
            else:
                raise ValueError(f"Not recognizable value: {value}")
        status[argument] = value
        arguments.append(argument)
        values.append(value)

    return status, arguments, values


def judge_is_chat_model(response):
    if "turbo" in response["model"] or "chat" in response['object'] or "message" in response['choices'][0]:
        return True
    return False

def extract_states(s, action, functor):
    pattern = fr'{action}\({functor}-(\d+)\)=(True|False)'
    res = re.findall(pattern, s)
    return {f"{functor}-{int(k)}": v == 'True' for k, v in res}

def generate_verbose_sentence(sentences):
    sentence = random.choice(sentences)
    if isinstance(sentence, list):
        sentence = " ".join(sentence)
    return sentence

def get_pred_probs(response, pred_vals):
    try:
        tokens = response["choices"][0]['tokens']
        top_logp = response["choices"][0]['top_logprobs']
    except KeyError:
        # sometimes OpenAI will have mysterious internal error
        tokens = response['choices'][0]['logprobs']['tokens']
        top_logp = response['choices'][0]['logprobs']['top_logprobs']

    pointer_val = 0
    all_probs = []
    for tok_i, tok in enumerate(tokens):
        if tok.lower() == "True".lower() or tok.lower() == "False".lower():
            assert tok.lower() == pred_vals[pointer_val].lower()
            logPs = top_logp[tok_i]
            logPs_items = list(logPs.items())
            logP_keys = set(x[0] for x in logPs_items)
            tr_prob = -999999
            for possible_pr_key in ["True", "true"]:
                if possible_pr_key in logP_keys:
                    tr_prob = max(logPs[possible_pr_key], tr_prob)
            false_prob = -999999
            for possible_false_key in ["False", "false"]:
                if possible_false_key in logP_keys:
                    false_prob = max(false_prob, logPs[possible_false_key])
            all_probs.append([tr_prob, false_prob])
            pointer_val += 1
            if pointer_val == len(pred_vals):
                break
    return all_probs


def get_total_variation(gt_status, pred_args, pred_vals, pred_probs):
    tv = 0
    if set(gt_status.keys()) != set(pred_args):
        return 999999
    for arg_i, arg in enumerate(pred_args):
        if pred_vals[arg_i] != gt_status[arg]:
            tv += abs(pred_probs[arg_i][0] - pred_probs[arg_i][1])
    return tv


def get_answer_from_response(response):
    is_chat_model = judge_is_chat_model(response)
    if not is_chat_model:
        sample_answer = response["choices"][0]['text']
    else:
        assert response['choices'][0]['message']['role'] == "assistant", f"incorrect roles found: {response['choices']}"
        sample_answer = response['choices'][0]['message']['content']
    return sample_answer


def set_answer_to_response(response, answer):
    is_chat_model = judge_is_chat_model(response)
    if not is_chat_model:
        response["choices"][0]['text'] = answer
    else:
        assert response['choices'][0]['message']['role'] == "assistant", f"incorrect roles found: {response['choices']}"
        response['choices'][0]['message']['content'] = answer
    return response


ERROR_TYPES = {
    "Not Parseable": "NP",
    "Not Parseable (prev)": "NP_prev",
    "Error Propagation": "EP",
    # do not ask you to do sth, but anyway you do it
    "Hallucinated Updates": "HU (IO)",
    "Hallucinated Updates (DR)": "HU (AC)",
    # ask you to do sth, but you do not do it
    "Not Following Updates (DW)": "NFU (DW)",
    "Not Following Updates": "NFU",
    "Maintain Wrong Belief (Dirty Read)": "DR",
    "Updates over Wrong Belief (Dirty Write)": "DW",
    "Accidentally Correct": "AC",
    "Maintain Correct Belief": "MC",
    "Correct Updates": "CU",
    "Inconsistent States Number": "ISN",
    "Void Action": "VA",
}

def check_error_type(gt_answer, response, prev_gt_answer, prev_response, logger, reverse_gt_val=False):
    # if the queried model do not return logprob, like GPT-3-turbo, llama, then we should not compute_tv
    gt_status, gt_args, gt_vals = process_state_representation(gt_answer, reverse_value=reverse_gt_val, logger=logger)
    prev_gt_status, prev_gt_args, prev_gt_vals = process_state_representation(prev_gt_answer,
                                                                              reverse_value=reverse_gt_val,
                                                                              logger=logger)
    sample_answer = get_answer_from_response(response)
    prev_sample_answer = get_answer_from_response(prev_response)
    error_types = dict()
    for err_key in ERROR_TYPES:
        error_types[err_key] = 0
    try:
        pred_status, pred_args, pred_vals = process_state_representation(sample_answer, logger=logger)
    except AssertionError as e:
        logger.warning(f"Exception met when parsing current response, Details: \n{e}")
        logger.warning(f"gt answer: {gt_answer}")
        logger.warning(f"sample answer: {sample_answer}")
        traceback.print_exc()
        error_types['Not Parseable'] += 1
        return error_types
    try:
        prev_pred_status, prev_pred_args, prev_pred_vals = process_state_representation(prev_sample_answer,
                                                                                        logger=logger)
    except AssertionError as e:
        logger.warning(f"Exception met when parsing previous response, Details: \n{e}")
        logger.warning(f"gt answer: {gt_answer}")
        logger.warning(f"sample answer: {sample_answer}")
        traceback.print_exc()
        error_types['Not Parseable (prev)'] += 1
        return error_types
    gt_status_delta = {}
    assert set(gt_status.keys()) == set(
        prev_gt_status.keys()), f"gt_status: {gt_status}, prev_gt_status: {prev_gt_status}"
    for arg in gt_status:
        if gt_status[arg] != prev_gt_status[arg]:
            gt_status_delta[arg] = gt_status[arg]
    try:
        assert set(pred_status.keys()) == set(
            prev_pred_status.keys()), f"response: {response}, pred_status: {pred_status}, prev_pred_status: {prev_pred_status}"
    except AssertionError as e:
        logger.warning(f"Exception met when comparing current and previous response, Details: \n{e}")
        logger.warning(f"gt answer: {gt_answer}")
        logger.warning(f"sample answer: {sample_answer}")
        traceback.print_exc()
        error_types['Inconsistent States Number'] += 1
        return error_types
    pred_status_delta = {}
    for arg in pred_status:
        if pred_status[arg] != prev_pred_status[arg]:
            pred_status_delta[arg] = pred_status[arg]

    # if previous state is already wrong, then there is always some possibility for error propagation
    for arg in prev_pred_status:
        if prev_pred_status[arg] != prev_gt_status[arg]:
            error_types["Error Propagation"] += 1
            if pred_status[arg] != gt_status[arg]:
                if arg in gt_status_delta:
                    # previously wrong, currently wrong, should be updated
                    error_types["Updates over Wrong Belief (Dirty Write)"] += 1
                    error_types["Not Following Updates (DW)"] += 1
                else:
                    error_types["Maintain Wrong Belief (Dirty Read)"] += 1
            else:
                error_types["Accidentally Correct"] += 1
                if arg in gt_status_delta:
                    # previously wrong, currently correct, should be updated
                    # previously wrong, this is the state that should be updated, but currently it matches GT, so it is
                    # accidentally correct, but not correct update, because it is not following the update
                    # [deprecated] example: (previous) T (gt) VS F (pred) -> (currently) F (gt) VS ? == -> F (pred) (symmetric)
                    # [deprecated] error_types["Not Following Updates"] += 1
                    # no sorry, the above situation won't happen, because our updates are permanents and uni-directional
                    # so the only possible example is:
                    # (previous) F (gt) VS T (pred) -> (currently) T (gt) VS ? == -> T (pred)
                    error_types["Void Action"] += 1
                else:
                    # previously wrong, currently correct, should not be updated, so it is accidentally correct
                    # this is the case that the model hallucinates updates
                    # example: (previous) T (gt) VS F (pred) -> (currently) T (gt) VS ? == -> T (pred) (symmetric)
                    error_types["Hallucinated Updates (DR)"] += 1
        else:
            if pred_status[arg] != gt_status[arg]:
                if arg in gt_status_delta:
                    # True-False-True
                    # previously correct, currently wrong, should be updated, but not updated
                    # example: (previous) T (gt) VS T (pred) -> (currently) F (gt) VS ? != -> T (pred) (symmetric)
                    error_types["Not Following Updates"] += 1
                else:
                    # True-False-False
                    # previously correct, currently wrong, should not be updated, but updated
                    # example: (previous) T (gt) VS T (pred) -> (currently) F (gt) VS ? != -> F (pred) (symmetric)
                    error_types["Hallucinated Updates"] += 1
            else:
                if arg in gt_status_delta:
                    # True-True-True
                    # previously correct, currently correct, should be updated, and updated
                    # example: (previous) T (gt) VS T (pred) -> (currently) T (gt) VS ? == -> T (pred) (symmetric)
                    error_types["Correct Updates"] += 1
                else:
                    # True-True-False
                    # previously correct, currently correct, should not be updated, not updated
                    # example: (previous) T (gt) VS T (pred) -> (currently) T (gt) VS ? == -> F (pred) (symmetric)
                    error_types["Maintain Correct Belief"] += 1
    return error_types


def process_gt_answer_response(gt_answer, response, logger, f1=False, reverse_gt_val=False, is_llama=False):
    # if the queried model do not return logprob, like GPT-3-turbo, llama, then we should not compute_tv
    is_chat_model = judge_is_chat_model(response)
    do_not_compute_tv = is_chat_model
    em_acc, stat_acc, tv_all = 0, 0, []
    if f1:
        stat_acc = {"recall": 0, "precision": 0, "f1": 0}
    parseable_acc = 1
    gt_status, gt_args, gt_vals = process_state_representation(gt_answer, reverse_value=reverse_gt_val, logger=logger)
    sample_answer = get_answer_from_response(response)
    if is_llama:
        sample_answer = sample_answer.strip().split("\n")[-1]
    if sample_answer == gt_answer:
        em_acc += 1
    try:
        pred_status, pred_args, pred_vals = process_state_representation(sample_answer, logger=logger, is_llama=is_llama)
    except AssertionError as e:
        logger.warning(f"Exception met when parsing, Details: \n{e}")
        logger.warning(f"gt answer: {gt_answer}")
        logger.warning(f"sample answer: {sample_answer}")
        logger.warning(f"stacktrace: {traceback.format_exc()}")
        parseable_acc = 0
        return em_acc, stat_acc, [9999999, ], parseable_acc
    if gt_status == pred_status:
        if not f1:
            stat_acc += 1
        else:
            stat_acc['precision'] = stat_acc['recall'] = stat_acc['f1'] = 1
    else:
        if f1:
            intersection_count = 0
            intersection_keys = set(gt_status.keys()) & set(pred_status.keys())
            for key in intersection_keys:
                if gt_status[key] == pred_status[key]:
                    intersection_count += 1
            precision = stat_acc["precision"] = intersection_count / len(pred_status)
            recall = stat_acc['recall'] = intersection_count / len(gt_status.keys())
            # in case there is no intersection and divided by 0
            stat_acc["f1"] = 2 * precision * recall / (precision + recall + 1e-7)
        if do_not_compute_tv:
            tv_all.append(-1)
        else:
            try:
                pred_probs = get_pred_probs(response, pred_vals)
                total_variation = get_total_variation(gt_status, pred_args, pred_vals, pred_probs)
                tv_all.append(total_variation)
            except AssertionError:
                logger.warning(
                    f"Found irregular logprob values as return values, please take notes: ID-{response['id']}")
                tv_all.append(99999999)
    return em_acc, stat_acc, tv_all, parseable_acc


def generateRandomWord(num_characters):
    alphas = string.ascii_letters
    sample_num_char = random.sample(range(1, num_characters), 1)[0]
    word = []
    for _ in range(sample_num_char):
        word.append(alphas[random.sample(range(len(alphas)), 1)[0]])
    return "".join(word)


def get_gpt_tokenizer(model_type):
    # to suppress countless warnings
    if model_type == "gpt-3.5-turbo":
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0301")
    else:
        encoding = tiktoken.encoding_for_model(model_type)
    return encoding

def compose_chat_style_wordload_from_normal_workload(workload):
    if isinstance(workload, str):
        lines = workload.split("\n")
    else:
        assert isinstance(workload, list)
        lines = workload

    chat_style_workload = []
    for line in lines:
        if line.startswith("Instruction"):
            assert len(chat_style_workload) == 0, "chat_style_workload should be empty at the beginning, before the instruction"
            chat_style_workload.append({"role": "user", "content": line})
        elif line.startswith("Step") or line.startswith("Question:"):
            chat_style_workload.append({"role": "user", "content": line})
        elif line.startswith("Answer:") and len(line.strip()) > len("Answer:"):
            chat_style_workload.append({"role": "assistant", "content": line})

    return chat_style_workload




if __name__ == '__main__':
    example = 'Instructions: We now are playing a box-opening game. As an agent, your task is to keep track of logical states. We will give you initial states, several actions and you should update logical states accordingly. There are 10 boxes and 10 keys here. Boxes are identified as MH-X and Keys are identified as MGOsHe-X.  OPENED(MH-3)=True means that MH-3 has been opened. OBTAINED(MGOsHe-3)=True means that MGOsHe-3 has been obtained. OPENED(MH-3)=False means that MH-3 has not been opened. OBTAINED(MGOsHe-3)=False means that MGOsHe-3 has not been obtained.\nStep-0: Initialization. Do nothing. \nQuestion: OPENED(MH-0)=?OPENED(MH-1)=?OPENED(MH-2)=?OPENED(MH-3)=?OPENED(MH-4)=?OPENED(MH-5)=?OPENED(MH-6)=?OPENED(MH-7)=?OPENED(MH-8)=?OPENED(MH-9)=?OBTAINED(MGOsHe-0)=?OBTAINED(MGOsHe-1)=?OBTAINED(MGOsHe-2)=?OBTAINED(MGOsHe-3)=?OBTAINED(MGOsHe-4)=?OBTAINED(MGOsHe-5)=?OBTAINED(MGOsHe-6)=?OBTAINED(MGOsHe-7)=?OBTAINED(MGOsHe-8)=?OBTAINED(MGOsHe-9)=?\nAnswer: OPENED(MH-0)=False.OPENED(MH-1)=False.OPENED(MH-2)=False.OPENED(MH-3)=False.OPENED(MH-4)=False.OPENED(MH-5)=False.OPENED(MH-6)=False.OPENED(MH-7)=False.OPENED(MH-8)=False.OPENED(MH-9)=False.OBTAINED(MGOsHe-0)=False.OBTAINED(MGOsHe-1)=False.OBTAINED(MGOsHe-2)=False.OBTAINED(MGOsHe-3)=False.OBTAINED(MGOsHe-4)=False.OBTAINED(MGOsHe-5)=False.OBTAINED(MGOsHe-6)=False.OBTAINED(MGOsHe-7)=False.OBTAINED(MGOsHe-8)=False.OBTAINED(MGOsHe-9)=False.\nStep-1: Open MH-6 and retrieve MGOsHe-6.\nQuestion: OPENED(MH-0)=?OPENED(MH-1)=?OPENED(MH-2)=?OPENED(MH-3)=?OPENED(MH-4)=?OPENED(MH-5)=?OPENED(MH-6)=?OPENED(MH-7)=?OPENED(MH-8)=?OPENED(MH-9)=?OBTAINED(MGOsHe-0)=?OBTAINED(MGOsHe-1)=?OBTAINED(MGOsHe-2)=?OBTAINED(MGOsHe-3)=?OBTAINED(MGOsHe-4)=?OBTAINED(MGOsHe-5)=?OBTAINED(MGOsHe-6)=?OBTAINED(MGOsHe-7)=?OBTAINED(MGOsHe-8)=?OBTAINED(MGOsHe-9)=?\nAnswer: OPENED(MH-0)=False.OPENED(MH-1)=False.OPENED(MH-2)=False.OPENED(MH-3)=False.OPENED(MH-4)=False.OPENED(MH-5)=False.OPENED(MH-6)=True.OPENED(MH-7)=False.OPENED(MH-8)=False.OPENED(MH-9)=False.OBTAINED(MGOsHe-0)=False.OBTAINED(MGOsHe-1)=False.OBTAINED(MGOsHe-2)=False.OBTAINED(MGOsHe-3)=False.OBTAINED(MGOsHe-4)=False.OBTAINED(MGOsHe-5)=False.OBTAINED(MGOsHe-6)=True.OBTAINED(MGOsHe-7)=False.OBTAINED(MGOsHe-8)=False.OBTAINED(MGOsHe-9)=False.\nStep-2: Open MH-5 and retrieve MGOsHe-3.\nStep-3: Open MH-2 and retrieve MGOsHe-9.\nStep-4: Open MH-3 and retrieve MGOsHe-8.\nStep-5: Open MH-8 and retrieve MGOsHe-1.\nStep-6: Open MH-7 and retrieve MGOsHe-2.\nStep-7: Open MH-0 and retrieve MGOsHe-7.\nStep-8: Open MH-1 and retrieve MGOsHe-5.\nQuestion: OPENED(MH-0)=?OPENED(MH-1)=?OPENED(MH-2)=?OPENED(MH-3)=?OPENED(MH-4)=?OPENED(MH-5)=?OPENED(MH-6)=?OPENED(MH-7)=?OPENED(MH-8)=?OPENED(MH-9)=?OBTAINED(MGOsHe-0)=?OBTAINED(MGOsHe-1)=?OBTAINED(MGOsHe-2)=?OBTAINED(MGOsHe-3)=?OBTAINED(MGOsHe-4)=?OBTAINED(MGOsHe-5)=?OBTAINED(MGOsHe-6)=?OBTAINED(MGOsHe-7)=?OBTAINED(MGOsHe-8)=?OBTAINED(MGOsHe-9)=?\nAnswer: '
    for line in compose_chat_style_wordload_from_normal_workload(example):
        print(line)
