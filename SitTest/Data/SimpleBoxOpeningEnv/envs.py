import copy
import json

import torch
import random
from random import shuffle


def generateSimpleBoxOpenningSample(num_box, num_key, logic_functor):
    box_list = list(range(num_box))
    key_list = list(range(num_key))
    shuffle(key_list)
    shuffle(box_list)
    # normal label2id_mapping
    label2id_mapping = dict()
    for box_i in range(num_box):
        label2id_mapping[f"BOX-{box_i}"] = box_i
    for key_i in range(num_key):
        label2id_mapping[f"KEY-{key_i}"] = num_box + key_i
    the_key_to_the_door = random.sample(range(num_key), 1)[0]
    # judge_condition = lambda box, key: True if key[the_key_to_the_door] else False
    judge_condition = the_key_to_the_door
    box2keys = torch.zeros((num_box, num_key))
    # let's start with the case that each box can only contain <=1 key
    for box_i, key_i in zip(box_list, key_list):
        box2keys[box_i][key_i] = 1

    env = SimpleBoxOpenGameEnv(
        box2keys, label2id_mapping, logic_functor, judge_condition
    )

    return env, the_key_to_the_door, judge_condition


class SimpleBoxOpenGameEnv:
    def __init__(self, box2keys: torch.Tensor, label2id_mapping: dict, logic_functor: dict, judge_condition):
        # box2keys: num_box * num_keys
        self.box2keys = box2keys
        self.num_boxes = box2keys.shape[0]
        self.num_keys = box2keys.shape[1]
        # judge_condition() takes the current status and decide whether the game is ended
        self.judge_condition = judge_condition
        self.label2id_mapping = label2id_mapping
        self.id2label_mapping = {v: k for k, v in self.label2id_mapping.items()}
        self.logic_functor = logic_functor
        self.original_functor = copy.copy(logic_functor)
        self.updated_history = {
            "box": [],
            "key": []
        }
        self.pre_run_history = []
        self.init()

    def init(self):
        self.key_status = torch.zeros([self.num_keys, ])
        self.box_status = torch.zeros([self.num_boxes, ])
        self.is_end = False
        for k in self.updated_history:
            self.updated_history[k].clear()
        self.logic_functor = self.original_functor
        self.pre_run_history.clear()

    def is_end(self):
        judge_condition = lambda box, key: True if key[self.judge_condition] else False
        return judge_condition(self.box_status, self.key_status)

    def update(self, by_id=False, **kwargs):
        assert by_id == True, "currently, update_by_name is under investigation and no guarantee on bug"
        if not by_id:
            self.update_by_name(**kwargs)
        else:
            self.update_by_id(**kwargs)

    def update_by_name(self, box_name, key_name=None):
        # [02/26/2023, Chenghao] this function is no longer used as the name of argument can be further perturbed
        # this box has been checked
        box_id = self.label2id_mapping[box_name]
        self.box_status[box_id] = 1
        if key_name is None:
            for i in range(self.num_keys):
                if self.box2keys[box_id][i]:
                    # this key has been obtained
                    self.key_status[i] = 1
        else:
            # it is possible that the user only takes one key from the box, but not all the key
            key_id = self.label2id_mapping[key_name] - self.num_boxes
            if self.box2keys[box_id][key_id]:
                self.key_status[key_id] = 1
            else:
                raise ValueError
                # print(f"Illegal operation, {key_name} not in {box_name}")

    def update_by_id(self, box_id, key_id=None, known_states_update=False):
        # this box has been checked
        # box_id = self.label2id_mapping[box_name]
        self.box_status[box_id] = 1
        if key_id is None:
            for i in range(self.num_keys):
                if self.box2keys[box_id][i]:
                    # this key has been obtained
                    self.key_status[i] = 1
        else:
            # it is possible that the user only takes one key from the box, but not all the key
            if self.box2keys[box_id][key_id]:
                self.key_status[key_id] = 1
            else:
                raise ValueError
        if known_states_update:
            self.updated_history["box"].append(box_id)
            self.updated_history["key"].append(key_id)
        # print(f"Illegal operation, {key_name} not in {box_name}")
    def regroup_history(self, start_index, end_index):
        # regroup the history from start_index to end_ind
        backup_box_history = copy.deepcopy(self.updated_history["box"])
        backup_key_history = copy.deepcopy(self.updated_history["key"])
        self.updated_history['box'] = backup_box_history[:start_index] + [backup_box_history[start_index: end_index], ] + backup_box_history[end_index:]
        self.updated_history['key'] = backup_key_history[:start_index] + [backup_key_history[start_index: end_index], ] + backup_key_history[end_index:]


    def get_available_items(self, collection):
        # items can be either boxes or keys
        # boxes -- pass in self.box_status as "collection"
        # keys -- pass in self.key_status as "collection"
        res = []
        for i in range(len(collection)):
            if not collection[i]:
                res.append(i)
        return res

    def get_available_keys(self):
        return self.get_available_items(self.key_status)

    def get_available_boxes(self):
        return self.get_available_items(self.box_status)

    def get_key_id(self, number: int):
        if number >= self.num_keys:
            raise ValueError(f"there is no key with number {number} as there are only {self.num_keys} keys!")
        return number + self.num_boxes

    def get_box_id(self, number: int):
        if number >= self.num_boxes:
            raise ValueError(f"there is no box with number {number} as there are only {self.num_boxes} boxes!")
        return number

    def get_key_name_by_id(self, key_id: int):
        return self.id2label_mapping[self.get_key_id(key_id)]

    def get_box_name_by_id(self, box_id: int):
        return self.id2label_mapping[self.get_box_id(box_id)]

    def get_serialization(self):
        return json.dumps({
            "box2keys": self.box2keys.tolist(),
            "label2id_mapping": self.label2id_mapping,
            "logic_functor": self.logic_functor,
            "judge_condition": self.judge_condition
        })

    def representation(self, sample_states=False, hints_on_known_states=False, sample_ratio=0.5, reverse_value=False):
        # answer and query must be paired, as it looks pretty weird that you only generate answer to partial question
        # it might be an interesting experiment tbh to see partial q + full a / full q + partial a,
        # but I think it is out of our scope as of 02/26/2023
        # furthermore, you can always get that by calling this function with different setting twice :-)
        def print_item_representation(item_kw, num_of_items, item_status, func_getID, query_res, answer_res):
            true_value = "True" if not reverse_value else "False"
            false_value = "False" if not reverse_value else "True"
            for item_i in range(num_of_items):
                # item_i: [0, num_boxes-1] / [0, num_keys-1], the true id in environment
                # func_getID(item_i): "id" used to index label
                # maybe in the future we will have a better way to do these indexing, but for now let's do sth dirty
                if item_status[item_i]:
                    answer_state = f"{self.logic_functor[item_kw]}({self.id2label_mapping[func_getID(item_i)]})={true_value}."
                else:
                    answer_state = f"{self.logic_functor[item_kw]}({self.id2label_mapping[func_getID(item_i)]})={false_value}."

                query_state = f"{self.logic_functor[item_kw]}({self.id2label_mapping[func_getID(item_i)]})=?"

                if hints_on_known_states:
                    if item_i in set(self.updated_history[item_kw]):
                        query_res.append(answer_state)
                    else:
                        query_res.append(query_state)
                else:
                    query_res.append(query_state)
                answer_res.append(answer_state)


        query_res = []
        answer_res = []
        print_item_representation("box", self.num_boxes, self.box_status, self.get_box_id, query_res, answer_res)
        print_item_representation("key", self.num_keys, self.key_status, self.get_key_id, query_res, answer_res)

        all_states_id = list(range(len(query_res)))
        if sample_states:
            sampled_ids = random.sample(all_states_id, int(sample_ratio * len(all_states_id)))
            sampled_ids.sort()
            query_res = [query_res[x] for x in sampled_ids]
            answer_res = [answer_res[x] for x in sampled_ids]

        return "".join(query_res), "".join(answer_res)
