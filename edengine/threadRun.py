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

import threading
import time
from datetime import datetime
from edelogger import logger


class EdePointThread(threading.Thread):
    def __init__(self, engine,
                 threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.engine = engine

    def run(self):
        logger.info('[{}] : [INFO] Starting Engine Point thread {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.threadID))
        self.engine.detectPointAnomalies()


class EdeTrainThread(threading.Thread):
    def __init__(self, engine,
                 threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.engine = engine

    def run(self):
        logger.info('[{}] : [INFO] Starting Engine Train thread {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.threadID))
        self.engine.trainMethod()


class EdeDetectThread(threading.Thread):
    def __init__(self, engine,
                 threadID):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.engine = engine

    def run(self):
        logger.info('[{}] : [INFO] Starting Engine Detect thread {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.threadID))
        self.engine.detectAnomalies()
