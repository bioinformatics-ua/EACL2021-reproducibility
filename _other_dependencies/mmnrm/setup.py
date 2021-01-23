from setuptools import setup

setup(
    name='mmnrm',
    version='0.0.2',
    description='tensorflow-keras layers and models for neural IR',
    license='MIT',
    packages=['mmnrm','mmnrm.layers'],
    author='Tiago Almeida',
    author_email='tiagomeloalmeida@ua.pt',
    keywords=['nir','neural','networks','keras','tensorflow'],
    install_requires=["tensorflow>=2.0.0"],
    url='https://github.com/T-Almeida/mmnrm'
)
