#!/bin/bash

# Removes all branches in the current repository not named 'main'

git branch | grep -v "main" | xargs git branch -D 
