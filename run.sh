#!/bin/bash

filename="input/13.jpg"

python3 -m venv venv
source venv/bin/actiate
pip3 install -r requirements.txt

python3 scr/main.py $filename