from setuptools import setup
from setuptools import find_packages


setup(name='keraflow',
      version='0.0.1',
      description='Deep Learning for Python',
      author='Shih-Ming Wang',
      author_email='swang150@ucsc.edu',
      url='https://github.com/ipod825/keraflow',
      download_url='https://github.com/ipod825/keraflow/tarball/0.0.1',
      license='MIT',
      install_requires=['theano', 'six', 'tqdm'],
      extras_require={
          'hickle': ['hickle'],
      },
      packages=find_packages())
