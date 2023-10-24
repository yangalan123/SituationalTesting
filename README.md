# SituationalTesting
Codebase for the EMNLP 2023 paper "Can You Follow Me? Testing Situational Understanding for ChatGPT"

Author: Chenghao Yang (yangalan1996@gmail.com) (University of Chicago)

Project Supervisor: [Allyson Ettinger](https://aetting.github.io/) (Allen Institute for Artificial Intelligence)

## Reference
If you use this code as part of any published research, please acknowledge the following paper (it encourages researchers who publish their code!):

```
@inproceedings{yang-2023-situational,
  author =  "Yang, Chenghao and Ettinger, Allyson",
  title =  "Can You Follow Me? Testing Situational Understanding for ChatGPT",
  booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
  year =  "2023"
}
```

## Dependency Installation
```bash
conda create -p ./env python=3.9
conda activate ./env # the environment position is optional, you can choose whatever places you like to save dependencies. Here I choose ./env for example.
pip install -e .
```

## Running Instruction

### Setup Prompted Model
For OpenAI GPT-3.5/4, you need to first obtain their API key to prompt these models. We refer readers to OpenAI official documents on [Developer Quickstart](https://platform.openai.com/docs/quickstart?context=python) for more details on setup OpenAI API access. Once you have obtained API key, please update `SitTest/Const.py` with your own API key to start prompting.

For Vicuna, we use [FastChat](https://github.com/lm-sys/FastChat) to serve Vicuna-13B with OpenAI-style API in our experiments at Appendix E (the experiments we have done is before the release of LLaMA2, so there might be additional accomodation needed if you want to try latest Vicuna). Please checkout their latest release and update the codes for prompting Vicuna at `SitTest/APIServer.py`. After launching the API, Controller and Model Worker of FastChat, you should obtain a local address (e.g., `http://localhost:8000/v1`) where Vicuna is serving. Specify `--use_llama`, ``--llama_server_url [address]`` and ``--model_type [vicuna-model-type]`` (e.g., `"vicuna-13b-v1.1"`) when you run `run.py`. We did not do intermediate state probing (We term it as `interactive debugging` in this codebase) for LLaMA-based models as we find it struggles to perform well even in a simplified 5-box environment and it is very hard to parse the outputs. But if you want to try, you should be able to migrate these LLaMA-related arguments to intermediate state probing codes. 

### For Main Testing Experiment (Section 4)
```bash
bash run.sh
```
Notes for `run.py` (executed by `run.sh`):
1. If you only want to check the input sample under different settings, pass `--debug True` when run `run.py`. You can find the first Python command of `run.sh` for an example running setup. 
1. If you want to estimate how much budget you need to pay, pass `--accounting_only`. We follow [OpenAI Official Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb) to use `tiktoken` library to estimate the budgets. This is only an estimation in the sense that we do not know how many tokens the prompted models will exactly generate. Also the pricing is changing for OpenAI models (you can update `pricing` of `Const.py` to reflect up-to-date cost).  
1. We did not have budget to fully explore various prompting technique and we believed overly-engineered prompts will distract our evaluation and analysis usages. But if you want to try different prompting format, we recommend you take a look at `prepare_chat_workload()` and `send_completion()` of `APIServer.py`. 
1. `--chat_style_probing` triggers "Faked Multi-Round Format". See Appendix C for more details. We stick to Traditional Format in the paper. You do not need extra config to trigger Traditional Format as this has been set as default. However, `--action_style_probing` is for distractor-related experiments (Appendix A) only. 
1. `--with_irreg_func` and `--with_irreg_arg` means whether you want to run with Synthetic Languages. You can play with these flags to achieve partial usages of synthetic langauges (Appendix B). 
1. `--instruction_counterfactual_nl` and `--instruction_counterfactual_logic` are for setting up counter-intuitive instruction (shown in Figure 1 and Table 1). 
1. You can run `python run.py --help` to get explanations for other important arguments. We generally recommend running not less than 50 samples to stabilize the performance numbers. There are some arguments that might be of your interest but we do not report the results in the paper, like we can also support partial query and incomplete supervision. We leave further investigation for future work. 

### For State Complexity and Intermediate State Probing (Section 5)

To verify the effect of state complexity as a confounder, you can run `run.py` with `--prerun_max_steps [num_steps]`. We set `prerun_max_steps=3` in the experiment at Section 5.1 paragraph 1.

To do Intermediate State Probing (termed as "Interactive Debugging" in this codebase), use:
```bash
bash run_interactive_debugging.sh
```
Please note that as this will do step-by-step verification as in Section 5, it will consume your budget very quickly and you should run with care. Also, as there might be many files to analyze in each setting, I have written a multi-process program to enable fast testing. Please make sure your device has multi-processing support. 

### Result Analysis
1. `compile_experiment_results_via_output.py`: Reproducing Table 1 and Table 2 (For Table 2, we may need a bit tweek on the running argument, check out the argument part).
1. `merge_interactive_debugging_figure.py`: Reproducing Figure 2.
1. `compile_experiment_results_interactive_debugging.py`: Reproducing Figure 3.



## Prompt-Responses Data
Under preparation. We are checking out whether there is personal information leaked in the responses and figuring out how to share a great amount of data. 
