from setuptools import find_packages, setup

setup(
        name='dlai_project',
        version='0',
        author='Dennis Rotondi',
        author_email='rotondi.1834864@studenti.uniroma1.it',
        url='https://github.com/DennisRotondi/dlai_project',
        packages=find_packages(),
        data_files=[('.', ['.env'])],
    )