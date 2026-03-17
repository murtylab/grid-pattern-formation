import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="grid_pattern_formation",
    version="0.0.2",
    description="grid_pattern_formation",
    author="",
    author_email="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/grid_pattern_formation/grid_pattern_formation",
    packages=setuptools.find_packages(),
    install_requires=["einops", "torch", "numpy", "wandb"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
