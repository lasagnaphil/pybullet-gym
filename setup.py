from setuptools import setup, find_packages
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pybulletgym'))

VERSION = 0.1

setup_py_dir = os.path.dirname(os.path.realpath(__file__))

setup(name='pybulletgym',
      version=VERSION,
      description='PyBullet Gym is an open-source implementation of the OpenAI Gym MuJoCo environments for use with the OpenAI Gym Reinforcement Learning Research Platform in support of open research.',
      url='https://github.com/benelot/pybullet-gym',
      author='Benjamin Ellenberger',
      author_email='be.ellenberger@gmail.com',
      license='',
      packages=[package for package in find_packages()
                if package.startswith('pybulletgym')],
      zip_safe=False,
      install_requires=[
          'pybullet>=1.7.8',
      ],
      include_package_data=True
)
