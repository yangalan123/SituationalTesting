# part of instruction
import re
from typing import Dict
import json
import random

def generate_initial_instruction_pipeline(args, logic_functor: Dict):

    # note: here we will generate the original instruction without any perturbation on functor or argument
    instructions = json.load(open(args.instruction_path, "r", encoding='utf-8'))
    # get goal specification and env introduction
    selected_instruction = instructions["instructions"][0]['text']
    # get env specification
    selected_instruction += f"There are {args.num_box} boxes and {args.num_key} keys here. "
    # get item specification
    selected_instruction += f"Boxes are identified as BOX-X and Keys are identified as KEY-X. "
    # generate True/False specification
    selected_sample_box_id = random.sample(range(args.num_box), 1)[0]
    selected_sample_key_id = random.sample(range(args.num_key), 1)[0]
    counterfactual_NL = args.instruction_counterfactual_nl
    counterfactual_Logic = args.instruction_counterfactual_logic
    additional_spec = logic_true_state_explainer(logic_functor, selected_sample_box_id, selected_sample_key_id, counterfactual_NL, counterfactual_Logic)
    additional_spec += logic_false_state_explainer(logic_functor, selected_sample_box_id, selected_sample_key_id, counterfactual_NL, counterfactual_Logic)
    # combine
    selected_instruction += additional_spec

    return selected_instruction

def logic_true_state_explainer(logic_functor:Dict, selected_sample_box_id, selected_sample_key_id, counterfactual_NL, counterfactual_Logic):
    return f" {logic_functor['box']}(BOX-{selected_sample_box_id})={'True' if not counterfactual_Logic else 'False'} means that BOX-{selected_sample_box_id} has {'' if not counterfactual_NL else 'not '}been opened. " \
           f"{logic_functor['key']}(KEY-{selected_sample_key_id})={'True' if not counterfactual_Logic else 'False'} means that KEY-{selected_sample_key_id} has {'' if not counterfactual_NL else 'not '}been obtained."

def logic_false_state_explainer(logic_functor:Dict, selected_sample_box_id, selected_sample_key_id, counterfactual_NL, counterfactual_Logic):
    return f" {logic_functor['box']}(BOX-{selected_sample_box_id})={'False' if not counterfactual_Logic else 'True'} means that BOX-{selected_sample_box_id} has {'not ' if not counterfactual_NL else ''}been opened. " \
           f"{logic_functor['key']}(KEY-{selected_sample_key_id})={'False' if not counterfactual_Logic else 'True'} means that KEY-{selected_sample_key_id} has {'not ' if not counterfactual_NL else ''}been obtained."


def extract_num_boxes_and_num_keys(instruction: str):
    pattern = r"There are (\d+) boxes and (\d+) keys here."
    match = re.search(pattern, instruction)
    assert match is not None, f"Cannot find the number of boxes and keys in the instruction: {instruction}"
    num_boxes = int(match.group(1))
    num_keys = int(match.group(2))
    return num_boxes, num_keys

def extract_logic_functor_for_box_and_keys(instruction: str):
    box_pattern = r"([^ ]*)\((.*)-(\d*)\)=(True|False) means that (.*)-(\d*) has (not )?been opened."
    match = re.search(box_pattern, instruction)
    assert match is not None, f"Cannot find the logic functor for box in the instruction: {instruction}"
    box_logic_functor = match.group(1)
    key_pattern = r"([^ ]*)\((.*)-(\d*)\)=(True|False) means that (.*)-(\d*) has (not )?been obtained."
    match = re.search(key_pattern, instruction)
    assert match is not None, f"Cannot find the logic functor for keys in the instruction: {instruction}"
    key_logic_functor = match.group(1)
    return {"box": box_logic_functor, "key": key_logic_functor}

