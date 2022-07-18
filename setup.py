from setuptools import setup

from qaner import __version__

with open("README.md", mode="r", encoding="utf-8") as fp:
    long_description = fp.read()


setup(
    name="qaner",
    version=__version__,
    description="Unofficial implementation of QaNER: Prompting Question Answering Models for Few-shot Named Entity Recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Dani El-Ayyass",
    author_email="dayyass@yandex.ru",
    license_files=["LICENSE"],
    url="https://github.com/dayyass/qaner",
    packages=["qaner"],
    install_requires=[
        "numpy==1.21.6",
        "tensorboard==2.9.0",
        "torch==1.8.1",
        "tqdm==4.64.0",
        "transformers==4.19.2",
    ],
)
