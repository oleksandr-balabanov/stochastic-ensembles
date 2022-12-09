#!/usr/bin/env bash
IMAGE=stochastic_ensembles:0
docker run --gpus=all --rm -it -u $(id -u):$(id -g) -w $(pwd) -v $(pwd):$(pwd) ${IMAGE} /bin/bash
