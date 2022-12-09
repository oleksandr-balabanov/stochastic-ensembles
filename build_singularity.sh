#!/usr/bin/env bash
pushd container
singularity build singularity.sif singularity.def
popd
