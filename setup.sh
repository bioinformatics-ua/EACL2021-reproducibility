#!/bin/bash

# exit ehen any command fails
set -e

PYTHON=python3

$PYTHON -m venv eacl2021-env

PYTHON=eacl2021-env/bin/python
PIP=eacl2021-env/bin/pip
# update pip

$PYTHON -m pip install -U pip

$PIP install -r requirements.txt


