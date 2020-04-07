#!/bin/bash
set -e

if [[ $* == *--user* ]]
then
    python -m pip install --user -r requirements.txt
else
    python -m pip install -r requirements.txt
fi