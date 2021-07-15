import os, sys
from dataformatter import DataFormatter
from pyQueryConstructor import QueryConstructor
from edeconnector import Connector
from edeconfig import readConf


if __name__ == '__main__':
    dataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    prometheus_endpoint = '194.102.62.155'
    # prometheus_endpoint = '10.9.8.136'
    # prometheus_endpoint = 'hal720m.sage.ieat.ro'
    prometheus_port = '9090'
    print("Collecting data from Monitoring at: {}".format(prometheus_endpoint))
    prometheus_query = {"query": '''{__name__=~"node.+"}[65m]'''}
    # prometheus_query = qContructor.pr_query_node(time="1h")
    edeConnector = Connector(prEndpoint=prometheus_endpoint, MInstancePort=prometheus_port)

    # test0 = edeConnector.pr_health_check()
    # print(test0)
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
    # print(test1['data']['result'])
    dformat = DataFormatter(dataDir)
    print("Query completed ....")
    print("Saving ...")
    test_format = dformat.prtoDF(test3, checkpoint=True, verbose=True)
    print("Saved")

    # test4 = edeConnector.localData('ede_data_exp.csv')



    # #Standard query values
    # # qte = 1475842980000
    # # qlte = 1475845200000
    # qgte = 1521390795179
    # qlte = 1521477195179
    # qsize = 0
    # qinterval = "10s"
    #
    #
    # dmonConnector = Connector('85.120.206.59')
    # qConstructor = QueryConstructor(queryDir='/Users/Gabriel/Documents/workspaces/diceWorkspace/dmon-adp/queries')
    # dformat = DataFormatter(dataDir)
    #
    # test = dmonConnector.clusterHealth()
    # test2 = dmonConnector.clusterSettings()
    # test3 = dmonConnector.clusterState()
    # test4 = dmonConnector.nodeInfo()
    # test5 = dmonConnector.nodeState()
    # test6 = dmonConnector.getIndex('logstash-*')
    # test7 = dmonConnector.getIndexSettings('logstash-*')
    #
    # # body = {
    # #         'timestamp': datetime.utcnow(),
    # #         'anomaly': 'complex',
    # #         'host': '10.0.0.0'
    # #     }
    # #
    # # test8 = dmonConnector.pushAnomaly('testme', doc_type='d', body=body)
    #
    # print(test)
    # print(test2)
    # print(test3)
    # print(test4)
    # print(test5)
    # print(test6)
    # print(test7)
    # # print test8
    #
    # nodes = ['dice.cdh.master', 'dice.cdh.slave1', 'dice.cdh.slave2', 'dice.cdh.slave3']
    # checkpoint = True
    # lload = []
    # lmemory = []
    # linterface = []
    # lpack = []
    # for node in nodes:
    #     load, load_file = qConstructor.loadString(node)
    #     memory, memory_file = qConstructor.memoryString(node)
    #     interface, interface_file = qConstructor.interfaceString(node)
    #     packet, packet_file = qConstructor.packetString(node)
    #
    #     # Queries
    #     qload = qConstructor.systemLoadQuery(load, qgte, qlte, qsize, qinterval)
    #     qmemory = qConstructor.systemMemoryQuery(memory, qgte, qlte, qsize, qinterval)
    #     qinterface = qConstructor.systemInterfaceQuery(interface, qgte, qlte, qsize, qinterval)
    #     qpacket = qConstructor.systemInterfaceQuery(packet, qgte, qlte, qsize, qinterval)
    #
    #     # Execute query and convert response to csv
    #     qloadResponse = dmonConnector.aggQuery(qload)
    #     gmemoryResponse = dmonConnector.aggQuery(qmemory)
    #     ginterfaceResponse = dmonConnector.aggQuery(qinterface)
    #     gpacketResponse = dmonConnector.aggQuery(qpacket)
    #
    #     if checkpoint:
    #         linterface.append(dformat.dict2csv(ginterfaceResponse, qinterface, interface_file, df=checkpoint))
    #         lmemory.append(dformat.dict2csv(gmemoryResponse, qmemory, memory_file, df=checkpoint))
    #         lload.append(dformat.dict2csv(qloadResponse, qload, load_file, df=checkpoint))
    #         lpack.append(dformat.dict2csv(gpacketResponse, qpacket, packet_file, df=checkpoint))
    #     else:
    #         dformat.dict2csv(ginterfaceResponse, qinterface, interface_file)
    #         dformat.dict2csv(gmemoryResponse, qmemory, memory_file)
    #         dformat.dict2csv(qloadResponse, qload, load_file)
    #         dformat.dict2csv(gpacketResponse, qpacket, packet_file)
    #
    # if not checkpoint:
    #     dformat.chainMergeSystem()
    #                 # Merge system metricsall
    #     merged_df = dformat.chainMergeNR()
    #     dformat.df2csv(merged_df, os.path.join(dataDir, "System.csv"))
    # else:
    #     df_interface, df_load, df_memory, df_packet = dformat.chainMergeSystem(linterface=linterface,
    #                                                                                 lload=lload, lmemory=lmemory,
    #                                                                                 lpack=lpack)
    #     merged_df = dformat.chainMergeNR(interface=df_interface, memory=df_memory,
    #                                           load=df_load, packets=df_packet)
    #     merged_df.set_index('key', inplace=True)
    #     merged_df.to_csv(os.path.join(dataDir, 'System_2.csv'))