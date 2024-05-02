import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchzeug",
    version="0.1.0",
    author="Paul Gavrikov",
    author_email="paul.gavrikov@hs-offenburg.de",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/paulgavrikov/torchzeug",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "numpy",
        "pynvml"
    ],
    python_requires=">=3.6",
)