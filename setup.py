from setuptools import setup
from pathlib import Path

README = Path('README.md').read_text()

setup(
    name='QGO',
    version='0.0.1',
    packages=setuptools.find_packages(),
)
