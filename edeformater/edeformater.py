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

import pandas as pd


class DataFormatter():
    def __init__(self, dataloc):
        self.dataloc = dataloc

    def aggJsonToCsv(self):
        return "CSV file"

    def expTimestamp(self):
        return "Expand metric timestamp"

    def window(self):
        return "Window metrics"

    def pivot(self):
        return "Pivot values"

    def addID(self):
        return "Add new ID as index"

    def removeID(self):
        return "Remove selected column as index"

    def renameHeader(self):
        return "Rename headers"

    def normalize(self):
        return "Normalize data"

    def denormalize(self):
        return "Denormalize data"

input_table = pd.read_csv("metrics.csv")


for index, row in input_table.iterrows():
    input_table = input_table.append([row]*9)

input_table = input_table.sort_values(['row ID'])
input_table = input_table.reset_index(drop=True)

for index, rows in input_table.iterrows():

    if int(index) > 59:
        print("Index to big!")
    time = rows[0].split(", ", 1) #In Knime row for timestamp is row(55) last one
    timeHour = time[1].split(":", 2)
    timeHourSeconds = timeHour[2].split(".", 1)
    timeHourSecondsDecimal = timeHour[2].split(".", 1)
    timeHourSecondsDecimal[0] = str(index)
    if len(timeHourSecondsDecimal[0]) == 1:
        timeHourSecondsDecimal[0] = '0%s' %timeHourSecondsDecimal[0]
    decimal = '.'.join(timeHourSecondsDecimal)
    timeHour[2] = decimal
    timenew = ':'.join(timeHour)
    time[1] = timenew
    finalString = ', '.join(time)
    input_table.set_value(index, 'row ID', finalString)

input_table.to_csv('out.csv')


