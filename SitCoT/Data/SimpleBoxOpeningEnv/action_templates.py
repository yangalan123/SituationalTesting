from SitCoT.Data.SimpleBoxOpeningEnv.envs import SimpleBoxOpenGameEnv
import re
def get_action_templates(env: SimpleBoxOpenGameEnv):

    action_templates = [
        lambda box_i, key_i,
               step_i=1: f"Step-{step_i}: Open {env.get_box_name_by_id(box_i)} and retrieve {env.get_key_name_by_id(key_i)}.\n",
        lambda box_i, key_i,
               step_i=1: f"Step-{step_i}: Get {env.get_key_name_by_id(key_i)} from {env.get_box_name_by_id(box_i)}.\n",
        # step_i = 1: f"Step-{step_i}: Get {id2label_mapping[num_box + key_i]} from {id2label_mapping[box_i]}\n",
        lambda box_i, key_i,
               step_i=1: f"Step-{step_i}: {env.get_box_name_by_id(box_i)} contains {env.get_key_name_by_id(key_i)} - open it to get the key.\n",
        # step_i = 1: f"Step-{step_i}: {id2label_mapping[box_i]} contains {id2label_mapping[num_box + key_i]} - open it to get the key\n",
        lambda box_i, key_i,
               step_i=1: f"Step-{step_i}: {env.get_key_name_by_id(key_i)} can be found in {env.get_box_name_by_id(box_i)} - open the box to get the key.\n",
        # step_i = 1: f"Step-{step_i}: {id2label_mapping[num_box + key_i]} can be found in {id2label_mapping[box_i]} - open the box to get the key\n",
        lambda box_i, key_i,
               step_i=1: f"Step-{step_i}: Find {env.get_key_name_by_id(key_i)} by opening {env.get_box_name_by_id(box_i)}.\n"
        # step_i = 1: f"Step-{step_i}: Find {id2label_mapping[num_box + key_i]} by opening {id2label_mapping[box_i]}\n"
    ]
    return action_templates

def extract_arguments_from_single_action(action:str, env: SimpleBoxOpenGameEnv):
    action = action.strip()
    # assuming we always use the first template
    pattern = "Step-(\d*): Open (.*) and retrieve (.*)\."
    step, box, key = re.findall(pattern, action)[0]
    box_i = re.search(r"(.*)-(\d*)", box).group(2)
    box_name = re.search(r"(.*)-(\d*)", box).group(1)
    key_i = re.search(r"(.*)-(\d*)", key).group(2)
    key_name = re.search(r"(.*)-(\d*)", key).group(1)
    if env is not None and hasattr(env, "updated_history"):
        assert len(env.updated_history['box']) == len(env.updated_history['key'])
        if len(env.updated_history["box"]) >= int(step):
            assert env.updated_history['box'][int(step) - 1] == int(box_i)
            assert env.updated_history['key'][int(step) - 1] == int(key_i)
    return int(step), int(box_i), int(key_i), box_name, key_name

def extract_arguments_from_multi_actions(action:str, env: SimpleBoxOpenGameEnv):
    # now it is possible that have multiple actions in one line
    # example-1: Step-X: Open Box-1, Box-2 and retrieve Key-3, Key-4
    # example-2: Step-X: Open Box-1 and retrieve Key-3, Open Box-2 and retrieve Key-4
    # we need to return step (1), box_i (1, 2), key_i (3, 4)
    # Get step number
    s = action.strip()
    step_match = re.search(r'Step-(\d+):', s)
    if not step_match:
        raise ValueError('Step not found')
    step = int(step_match.group(1))

    # Initialize box_ids, key_ids, box_name and key_name
    box_ids = []
    key_ids = []
    box_name = None
    key_name = None

    # Split the instructions into sequences of "Open... and retrieve..."
    sequences = re.split(r'(?=Open)', s)[1:]  # Use positive lookahead to split without consuming "Open"

    # For each sequence, extract the box IDs and key IDs
    for sequence in sequences:
        boxes, keys = re.search(r'Open (.*?) and retrieve (.*?)$', sequence).groups()

        # Extract box_name and key_name
        box_name_curr = re.search(r'(\w+)-\d+', boxes).group(1)
        key_name_curr = re.search(r'(\w+)-\d+', keys).group(1)

        # Check if box_name and key_name are consistent
        if box_name is None and key_name is None:
            box_name, key_name = box_name_curr, key_name_curr
        elif box_name != box_name_curr or key_name != key_name_curr:
            raise ValueError('Inconsistent box_name or key_name')

        box_ids.extend(map(int, re.findall(r'\w+-(\d+)', boxes)))
        key_ids.extend(map(int, re.findall(r'\w+-(\d+)', keys)))

    # Combine the results
    # return f'step={step}, box_ids={box_ids}, key_ids={key_ids}'
    return step, box_ids, key_ids, box_name, key_name


def extract_time_from_single_action(action:str):
    pattern = "Step-(\d*):"
    step = re.findall(pattern, action)[0]
    return int(step)
