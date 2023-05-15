#!/bin/bash

FN=$1
ROOT_DIR=$2
BASE_FN=${FN%??????}

NARGS=$#
if [ $NARGS -ne 2 ]; then
	echo "Usage: $0 <notebook> <target_dir>"
	exit
fi

if [ ! -f ${FN} ]; then
	echo "File ${FN} does not exist"
	exit
fi

if [ ! -d ${ROOT_DIR} ]; then
	echo "Directory ${ROOT_DIR} does not exist"
	exit
fi

NB_DIR="${ROOT_DIR}/_includes/notebooks"
mkdir -p ${NB_DIR}

IMG_DIR="${ROOT_DIR}/img"
mkdir -p ${IMG_DIR}

echo $NB_DIR
echo $IMG_DIR
FILES_DIR="${IMG_DIR}/${BASE_FN}_files"
# exit
# Convert notebook to html
# jupyter nbconvert ${FN} --to html --output-dir html --template lab
jupyter nbconvert ${FN} --to html --output-dir html --template lab --HTMLExporter.preprocessors="nbconvert.preprocessors.ExtractOutputPreprocessor"
python nb_strip.py html/${BASE_FN}.html -o ${NB_DIR}/${BASE_FN}.html

echo $IMG_DIR
echo $FILES_DIR
if [[ -d $FILES_DIR ]]; then
	rm -rf $FILES_DIR
fi

mv -f html/${BASE_FN}_files ${IMG_DIR}/
