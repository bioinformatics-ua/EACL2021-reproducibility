#!/bin/bash

# exit ehen any command fails
set -e

ZIP_FILE="download_folder.zip"

# DOWNLOAD SOME BIG FILES THAT ARE REQUIRED

if [ ! -f "$ZIP_FILE" ]; then
	echo "Downloading a zip folder with embeddings and tokenizers"
	./_other_dependencies/download_folder.sh
fi	        
		    
echo "Starting the unziping" 
unzip $ZIP_FILE


# PYTHON DEPENDENCIES
PYTHON=python3

echo "Creating a python environment (eacl2021)"
$PYTHON -m venv eacl2021-env

PYTHON=$(pwd)/eacl2021-env/bin/python
PIP=$(pwd)/eacl2021-env/bin/pip
# update pip

echo "Updating pip"
$PYTHON -m pip install -U pip

echo "Installing python requirements"
$PIP install -r requirements.txt

echo "Manually install mmnrm python library"
cd _other_dependencies/mmnrm/

if [ -d "./dist" ]
then
	rm -r ./dist
fi

$PYTHON setup.py sdist
$PIP install ./dist/mmnrm-0.0.2.tar.gz
cd ../../


