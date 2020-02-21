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

import multiprocessing
import time
from datetime import datetime
from edelogger import logger


def test(times, processID):
    logger.info('[{}] : [INFO] Starting Engine Point process {}'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), processID))
    time.sleep(times)
    logger.info('[{}] : [INFO] Exit Engine Point process {}'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), processID))


class EdePointProcess():
    def __init__(self, engine,
                 processID):
        self.processID = processID
        self.engine = engine

    def run(self):
        logger.info('[{}] : [INFO] Starting engine Point process  {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.processID))
        p = multiprocessing.Process(target=self.engine.detectPointAnomalies)
        return p


class EdeTrainProcess():
    def __init__(self, engine,
                 processID):
        self.processID = processID
        self.engine = engine

    def run(self):
        logger.info('[{}] : [INFO] Starting engine Train process  {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.processID))
        p = multiprocessing.Process(target=self.engine.trainMethod)
        return p


class EdeDetectProcess():
    def __init__(self, engine,
                 processID):
        self.processID = processID
        self.engine = engine

    def run(self):
        logger.info('[{}] : [INFO] Starting engine Detect process  {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.processID))
        p = multiprocessing.Process(target=self.engine.detectAnomalies)
        return p


# jobs = []
# testProcess = AdpPointProcess('engine', 'PointProcess')
# testProcess2 = AdpTrainProcess('engine', 'TrainProcess')
# testProcess3 = AdpDetectProcess('engine', 'Detect')
# initProcess = testProcess.run()
# jobs.append(initProcess)
# initProcess2 = testProcess2.run()
# jobs.append(initProcess2)
# initProcess3 = testProcess3.run()
# jobs.append(initProcess3)
#
# initProcess.start()
# initProcess2.start()
# initProcess3.start()
#
# for j in jobs:
#     j.join()
#     print '%s.exitcode = %s' % (j.name, j.exitcode)
