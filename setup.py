from setuptools import setup, find_packages

VERSION = 0.1

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
