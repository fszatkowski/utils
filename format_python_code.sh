#!/bin/bash

# Runs black and isort formatting, then flake8 and mypy linting recursively on given directory.

path=$1

black $path;
isort $path --profile black;
flake8 $path;
mypy $path;
