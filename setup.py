from setuptools import setup, find_packages
from typing import List


def get_requirements(file_path:str)->List[str]:
    """
    This function returns a list of requirements
    """
    
    requirements = []
    with open(file_path, 'r') as f:
        requirements = f.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
            
    return requirements

setup(
    name = "mlproject",
    version="0.0.1",
    author='Prasanth',
    author_email='amulyaprasanth301@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)