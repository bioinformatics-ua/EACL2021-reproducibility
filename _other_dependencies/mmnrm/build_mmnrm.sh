#!/bin/bash

cd /home/tiagoalmeida/mmnrm

rm -r ./dist

python setup.py sdist

pip install ./dist/mmnrm-0.0.2.tar.gz

