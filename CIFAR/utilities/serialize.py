#!/usr/bin/env python3
import tempfile
import pickle
import dill
import os
import json
import pandas
from dataclasses import asdict


def serialize(data):
    return dill.dumps(data)


def deserialize(bytes):
    return dill.loads(bytes)


def save(data, path):
    with open(path, "wb") as f:
        f.write(serialize(data))


def load(path):
    with open(path, "rb") as f:
        return deserialize(f.read())


def save_tmp(data):
    fdhandle, path = tempfile.mkstemp()
    os.write(fdhandle, serialize(data))
    os.close(fdhandle)
    return path


def sanitize_dict(data):
    ret = dict(data)
    for key, val in data.items():
        if callable(val):
            del ret[key]
    return ret


def dataclass_to_normalized_json(data):
    return json.loads(pandas.json_normalize(asdict(data)).to_json(orient="records"))[0]
