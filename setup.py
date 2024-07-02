from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    desc = f.read()

setup(
    name="space_filling_pytorch",
    version="0.0.2",
    description="A set of pytorch implementations for space filling curve using OpenAI Triton",
    author="Kitsunetic",
    author_email="jh.shim.gg@gmail.com",
    url="https://github.com/Kitsunetic/space-filling-pytorch",
    packages=find_packages(include=["space_filling_pytorch", "space_filling_pytorch.*"]),
    zip_safe=False,
    install_requires=[
        "torch",
        "triton",
    ],
    long_description=desc,
    long_description_content_type="text/markdown",
)
