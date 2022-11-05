from setuptools import find_packages,setup
from typing import List


def get_requirements() -> List[str]:
    """
    This function will return list of requirements
    """
    requirement_list:List[str] = []
    with open('requirements.txt') as f:
        requirement_list=[require.strip() for require in f.readlines()]
        print(requirement_list)
    return requirement_list

setup(

    name='sensor',
    version='0.0.1',
    author='ineuron',
    author_email='ayugupta2506@gmail.com',
    packages=find_packages(),
    #install_requires = get_requirements(),
)