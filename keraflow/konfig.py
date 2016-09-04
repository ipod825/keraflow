import json
import os


def _get_config_dir():
    res = os.path.expanduser('~')
    if not os.access(res, os.W_OK):
        res = '/tmp'
    res = os.path.join(res, '.keraflow')

    if not os.path.exists(res):
        os.makedirs(res)
    return res


def get_config():
    if os.path.exists(keraflow_config):
        with open(keraflow_config) as f:
            config = json.load(f)
        return config
    else:
        return {}


def save_config(config):
    with open(keraflow_config, 'w') as f:
        json.dump(config, f, indent=2)

keraflow_dir = _get_config_dir()
keraflow_config = os.path.join(keraflow_dir, 'keraflow.json')
