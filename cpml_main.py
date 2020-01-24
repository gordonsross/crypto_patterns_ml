# Cryptocurrency Market Patterns derived from Machine Learning

import datetime as dt
import json
import logging
import numpy as np
import os
import yaml



log = logging.getLogger(__name__)
config = None

def load_config():
    with open('config.yml', 'r') as yml_file:
        return yaml.load(yml_file))

def setup_logging():
    root_log = logging.getLogger()
    log_config.set


def main():
    pass

if __name__ == '__main__':
    config = load_config()
    setup_logging()

    main()