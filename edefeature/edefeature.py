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

from sklearn.feature_selection import VarianceThreshold
import pandas
from edelogger import logger
import time
from datetime import datetime
import sys


class EdeFeatureSelector:

    def __init__(self):
        self.author = 'Constructor for dmon-adp  feature selection methods'

    def varianceSelection(self, df, threashold=.8):
        if not isinstance(df, pandas.core.frame.DataFrame):
            logger.error('[%s] : [ERROR] Variance selection only possible on Dataframe not %s',
                                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(df))
            sys.exit(1)
        sel = VarianceThreshold(threshold=(threashold * (1 - threashold)))
        sel.fit_transform(df)
        return df[[c for (s, c) in zip(sel.get_support(), df.columns.values) if s]]

