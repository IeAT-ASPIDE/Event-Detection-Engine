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

import logging
from logging.handlers import RotatingFileHandler
import os
import datetime

logger = logging.getLogger("EDE Log")
logger.setLevel(logging.INFO)

loggerESt = logging.getLogger('elasticsearch.trace')
loggerESt.setLevel(logging.INFO)
loggerES = logging.getLogger('elasticsearch')
loggerES.setLevel(logging.INFO)
loggerurl3 = logging.getLogger("urllib3")
loggerurl3.setLevel(logging.INFO)


# add a rotating handler
logFile = os.path.join('ede.log')
handler = RotatingFileHandler(logFile, maxBytes=100000000,  backupCount=5)
logger.addHandler(handler)
loggerESt.addHandler(handler)
loggerES.addHandler(handler)
loggerurl3.addHandler(handler)

consoleHandler = logging.StreamHandler()

logger.addHandler(consoleHandler)

