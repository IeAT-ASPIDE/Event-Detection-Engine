import os, sys
from dataformatter import DataFormatter
from pyQueryConstructor import QueryConstructor
from edeconnector import Connector


if __name__ == '__main__':
    dataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    prometheus_endpoint = '194.102.62.155'
    prometheus_port = '9090'
    print("Collecting data from Monitoring at: {}".format(prometheus_endpoint))
    prometheus_query = {"query": '''{__name__=~"node.+"}[80m]'''}
    # prometheus_query = qContructor.pr_query_node(time="1h")
    edeConnector = Connector(prEndpoint=prometheus_endpoint, MInstancePort=prometheus_port)

    test = edeConnector.pr_targets()
    print("Current target information:")
    print(test)
    test1 = edeConnector.pr_labels('cpu')
    print(test1)
    test2 = edeConnector.pr_status()
    print("Status information")
    print(test2)
    print("Executing query ....")
    test3 = edeConnector.pr_query(query=prometheus_query)
    dformat = DataFormatter(dataDir)
    print("Query completed ....")
    print("Saving ...")
    test_format = dformat.prtoDF(test3, checkpoint=True, verbose=True)
    print("Saved")
