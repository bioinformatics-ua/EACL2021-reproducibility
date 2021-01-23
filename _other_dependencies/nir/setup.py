from setuptools import setup

setup(
    name='nir',
    version='0.0.1',
    description='tools for neural IR aplications',
    license='MIT',
    packages=['nir'],
    author='Tiago Almeida',
    author_email='tiagomeloalmeida@ua.pt',
    keywords=['nir','tokenization'],
    install_requires=["fasttext==0.9.1"],
    url='https://github.com/T-Almeida/DL-IR-tools'
)
