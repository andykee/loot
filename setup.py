import re

from setuptools import setup

with open('loot/__init__.py') as f:
    version = re.search('__version__ = "(.*?)"', f.read()).group(1)

setup(
    name='loot',
    version=version,
    install_requires=[
        'pandas',
        'numpy',
        'numpy-financial',
        'python-dateutil'
    ]
)
