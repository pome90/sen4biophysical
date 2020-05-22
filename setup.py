import setuptools
import os

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sen4biophysical", # Replace with your own username
    version="0.0.1",
    author="Andrea Pomente",
    author_email="andrea.pomente@nodriver.ai",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pome90/sen4biophysical",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU 3 License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    entry_points = {
        "console_scripts": [
            "sen4biophysical = scripts.__main__:main",
        ]
    },
    install_requires=parse_requirements("requirements.txt")
)