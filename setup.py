from setuptools import setup, find_packages

with open("SitTest/requirements.txt") as fp:
    requirements = fp.read().splitlines()

setup(
    name="SitTest",
    version="0.0.0",
    author="Chenghao Yang",
    packages=find_packages(exclude=["logs", "GPT3Output", "logs_debug", "NL-Augmenter", "paraphrasing_models", "tw_data", "visualization"]),
    install_requires=requirements
)
