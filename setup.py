from setuptools import setup
from setuptools import find_packages

setup(name='dti_gcmc',
      version='0.1',
      description='Drug-Target Interaction using GCMC',
      author='Vedang Waradpande',
      author_email='vedang.waradpande@gmail.com',
      license='MIT',
      install_requires=['numpy',
                        'tensorflow',
                        'scipy',
                        'pandas',
                        'h5py'
                        ],
      package_data={'dti_gcmc': ['README.md']},
      packages=find_packages())
