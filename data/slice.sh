#!/bin/bash

set -ex

find . -iname '*.stl' -exec prusa-slicer --center 150,150 -g {} \;
