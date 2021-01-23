#!/bin/bash

python3 setup.py sdist

sudo pip3 uninstall nir-tools
sudo pip3 install $1

