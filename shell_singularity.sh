#!/usr/bin/env bash
IMAGE=container/singularity.sif
singularity shell --cleanenv --no-home --env PYTHONNOUSERSITE=1 --nv -B $(pwd):$(pwd) ${IMAGE}
