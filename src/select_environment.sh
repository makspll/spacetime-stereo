#!/bin/bash

help_message="select_environments:
    run this command like:
    ./select_environment [leastereo|]
"

eval "$(conda shell.bash hook)"

if [ "$1" = "leastereo" ]; then
    conda activate leastereo
else
    echo $help_message
fi
