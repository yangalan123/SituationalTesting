from SitCoT.utils import generateRandomWord
logic_functor = {
    "box": "OPENED",
    "key": "OBTAINED"
}

def generate_random_functor(args):
    logic_functor_candidates = []
    for _ in range(args.num_logic_functors):
        box_func = generateRandomWord(args.max_functor_length)
        key_func = generateRandomWord(args.max_functor_length)
        logic_functor_candidates.append(
            {
                "box": box_func,
                "key": key_func
            }
        )
    return logic_functor_candidates
