## EACL reproducibility repository

This repository holds the code used for the experiments presented on the paper [Benchmarking a transformer-FREE model for ad-hoc retrieval](https://www.aclweb.org/anthology/2021.eacl-main.293/)

To replicate the experiments, first it is required to create the same test environment, for that run the **setup.sh** script. This should download the weights, tokenizers and other larger files, after this the python environment (*eacl2021-env*) will be created and all the dependencies will be installed. Note that the python 3.6 version will be used, this can be changed in the script. However, the minimum requirement would be the 3.6 version and no guarantee is given that will work in the post versions.

To run the experiments, just execute the scripts that start with *run_*. Note that in the case of GPU only one will be used so some configuration may be required by the user.


