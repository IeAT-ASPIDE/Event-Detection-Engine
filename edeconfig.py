"""
Copyright 2019, Institute e-Austria, Timisoara, Romania
    http://www.ieat.ro/
Developers:
 * Gabriel Iuhasz, iuhasz.gabriel@info.uvt.ro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from configparser import SafeConfigParser
from edelogger import logger
from util import checkFile
from datetime import datetime
import time
import yaml
import sys


def readConf(file):
    '''
    :param file: location of config file
    :return: conf file as dict
    '''
    if not checkFile(file):
        logger.error('[{}] : Configuration file not found at {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), file))
        sys.exit(1)
    file_extension = file.split('.')[-1]
    if file_extension == 'ini':
        parser = SafeConfigParser()
        parser.read(file)
        conf = {}
        for selection in parser.sections():
            inter = {}
            for name, value in parser.items(selection):
                inter[name] = value
            conf[selection] = inter
    elif file_extension == 'yaml' or file_extension == 'yml':
        with open(file) as cf:
            conf = yaml.unsafe_load(cf)
        # try:
        #     with open(file) as cf:
        #         conf = yaml.unsafe_load(cf)
        # except Exception as inst:
        #     logger.error('[{}] : Failed to parse configuration file with {} and {}'.format(
        #         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
        #     sys.exit(1)
    else:
        logger.error('[{}] : Unsupported configuration file extension {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), file_extension))
        sys.exit(1)
    return conf



