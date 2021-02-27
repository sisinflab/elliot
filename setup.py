from setuptools import Extension, setup, find_packages, dist
from codecs import open
from os import path

dist.Distribution().fetch_build_eggs(['numpy>=1.11.2'])
try:
    import numpy as np
except ImportError:
    exit('Please install numpy>=1.11.2 first.')

try:
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

__version__ = '1.0.0'

here = path.abspath(path.dirname(__file__))
# Get the long description from README.md
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

ext = '.pyx' if USE_CYTHON else '.c'
cmdclass = {}

ext = '.pyx' if USE_CYTHON else '.c'

setup(
    name='elliot',
    version=__version__,
    author='',
    author_email='',
    maintainer='sisnflab',
    maintainer_email='',
    license='Apache License',
    packages=['elliot'],
    platforms=['all'],
    description=('A Comprehensive and RigorousFramework for Reproducible Recommender Systems Evaluation'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/sisnflab/elliot',
    keywords='recommender recommendation system evaluation framework',
    cmdclass=cmdclass,
    install_requires=install_requires,

    classifiers=[
        'Development Status :: 4 - Beta',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache License',
        'Programming Language :: Python',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Software Development :: Libraries'
    ]
)

# how to send a package
# 1. python setup.py sdist build / python setup.py bdist_wheel --universal
# 2. pip install twine
# 3. twine upload dist/*