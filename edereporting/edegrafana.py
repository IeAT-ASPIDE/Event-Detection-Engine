"""
Copyright 2022, Institute e-Austria and
    West University of Timsioara, Timisoara, Romania
    https://www.ieat.ro/
    https://www.uvt.ro

Developers:
 * Gabriel Iuhasz, iuhasz.gabriel@info.uvt.ro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from grafana_client import GrafanaApi
from edelogger import logger
from datetime import datetime
import time


class EDEGrafanaDash:
    def __init__(self, grafana_token, grafana_url):
        self.dash_dict = {
                  "dashboard": {
                    "id": None,
                    "uid": None,
                    "title": "UVT Serrano dash",
                    "tags": ["test_dash"],
                    "timezone": "browser",
                    "schemaVersion": 16,
                    "version": 0,
                    "refresh": "10s",
                    "time": {
                      "from": "now-6h",
                      "to": "now"
                    },
                    "panels": [
                    {
                      "datasource": {
                        "type": "prometheus",
                        "uid": "Y8Y4A4mVz"
                      },
                      "fieldConfig": {
                        "defaults": {
                          "color": {
                            "mode": "palette-classic"
                          },
                          "custom": {
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                              "legend": False,
                              "tooltip": False,
                              "viz": False
                            },
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {
                              "type": "linear"
                            },
                            "showPoints": "auto",
                            "spanNones": False,
                            "stacking": {
                              "group": "A",
                              "mode": "none"
                            },
                            "thresholdsStyle": {
                              "mode": "off"
                            }
                          },
                          "mappings": [],
                          "thresholds": {
                            "mode": "absolute",
                            "steps": [
                              {
                                "color": "green",
                                "value": False
                              },
                              {
                                "color": "red",
                                "value": 80
                              }
                            ]
                          }
                        },
                        "overrides": []
                      },
                      "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 0
                      },
                      "id": 12,
                      "options": {
                        "legend": {
                          "calcs": [],
                          "displayMode": "list",
                          "placement": "bottom"
                        },
                        "tooltip": {
                          "mode": "single",
                          "sort": "none"
                        }
                      },
                      "targets": [
                        {
                          "datasource": {
                            "type": "prometheus",
                            "uid": "Y8Y4A4mVz"
                          },
                          "editorMode": "builder",
                          "expr": "node_procs_running",
                          "legendFormat": "__auto",
                          "range": True,
                          "refId": "A"
                        }
                      ],
                      "title": "Running Processes UVT K8S Grafana",
                      "type": "timeseries"
                    },
                    {
                      "datasource": {
                        "type": "prometheus",
                        "uid": "Y8Y4A4mVz"
                      },
                      "fieldConfig": {
                        "defaults": {
                          "color": {
                            "mode": "palette-classic"
                          },
                          "custom": {
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                              "legend": False,
                              "tooltip": False,
                              "viz": False
                            },
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {
                              "type": "linear"
                            },
                            "showPoints": "auto",
                            "spanNones": False,
                            "stacking": {
                              "group": "A",
                              "mode": "none"
                            },
                            "thresholdsStyle": {
                              "mode": "off"
                            }
                          },
                          "mappings": [],
                          "thresholds": {
                            "mode": "absolute",
                            "steps": [
                              {
                                "color": "green",
                                "value": None
                              },
                              {
                                "color": "red",
                                "value": 80
                              }
                            ]
                          }
                        },
                        "overrides": []
                      },
                      "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 0
                      },
                      "id": 4,
                      "options": {
                        "legend": {
                          "calcs": [],
                          "displayMode": "list",
                          "placement": "bottom"
                        },
                        "tooltip": {
                          "mode": "single",
                          "sort": "none"
                        }
                      },
                      "targets": [
                        {
                          "datasource": {
                            "type": "prometheus",
                            "uid": "Y8Y4A4mVz"
                          },
                          "editorMode": "builder",
                          "expr": "node_memory_MemFree_bytes",
                          "legendFormat": "__auto",
                          "range": True,
                          "refId": "A"
                        }
                      ],
                      "title": "Memory Free K8S UVT",
                      "type": "timeseries"
                    },
                    {
                      "datasource": {
                        "type": "prometheus",
                        "uid": "Y8Y4A4mVz"
                      },
                      "fieldConfig": {
                        "defaults": {
                          "color": {
                            "mode": "palette-classic"
                          },
                          "custom": {
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                              "legend": False,
                              "tooltip": False,
                              "viz": False
                            },
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {
                              "type": "linear"
                            },
                            "showPoints": "auto",
                            "spanNones": False,
                            "stacking": {
                              "group": "A",
                              "mode": "none"
                            },
                            "thresholdsStyle": {
                              "mode": "off"
                            }
                          },
                          "mappings": [],
                          "thresholds": {
                            "mode": "absolute",
                            "steps": [
                              {
                                "color": "green",
                                "value": None
                              },
                              {
                                "color": "red",
                                "value": 80
                              }
                            ]
                          }
                        },
                        "overrides": []
                      },
                      "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 0,
                        "y": 8
                      },
                      "id": 6,
                      "options": {
                        "legend": {
                          "calcs": [],
                          "displayMode": "list",
                          "placement": "bottom"
                        },
                        "tooltip": {
                          "mode": "single",
                          "sort": "none"
                        }
                      },
                      "targets": [
                        {
                          "datasource": {
                            "type": "prometheus",
                            "uid": "Y8Y4A4mVz"
                          },
                          "editorMode": "builder",
                          "expr": "node_load5",
                          "legendFormat": "__auto",
                          "range": True,
                          "refId": "A"
                        }
                      ],
                      "title": "Load 5 UVT K8S Grafana",
                      "type": "timeseries"
                    },
                    {
                      "datasource": {
                        "type": "prometheus",
                        "uid": "Y8Y4A4mVz"
                      },
                      "fieldConfig": {
                        "defaults": {
                          "color": {
                            "mode": "palette-classic"
                          },
                          "custom": {
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                              "legend": False,
                              "tooltip": False,
                              "viz": False
                            },
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {
                              "type": "linear"
                            },
                            "showPoints": "auto",
                            "spanNones": False,
                            "stacking": {
                              "group": "A",
                              "mode": "none"
                            },
                            "thresholdsStyle": {
                              "mode": "off"
                            }
                          },
                          "mappings": [],
                          "thresholds": {
                            "mode": "absolute",
                            "steps": [
                              {
                                "color": "green",
                                "value": None
                              },
                              {
                                "color": "red",
                                "value": 80
                              }
                            ]
                          }
                        },
                        "overrides": []
                      },
                      "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 8
                      },
                      "id": 10,
                      "options": {
                        "legend": {
                          "calcs": [],
                          "displayMode": "list",
                          "placement": "bottom"
                        },
                        "tooltip": {
                          "mode": "single",
                          "sort": "none"
                        }
                      },
                      "targets": [
                        {
                          "datasource": {
                            "type": "prometheus",
                            "uid": "Y8Y4A4mVz"
                          },
                          "editorMode": "builder",
                          "expr": "rate(node_network_receive_bytes_total[$__rate_interval])",
                          "legendFormat": "__auto",
                          "range": True,
                          "refId": "A"
                        }
                      ],
                      "title": "Network Received Bytes (Rate) UVT K8S Grafana",
                      "type": "timeseries"
                    },
                    {
                      "datasource": {
                        "type": "prometheus",
                        "uid": "Y8Y4A4mVz"
                      },
                      "fieldConfig": {
                        "defaults": {
                          "color": {
                            "mode": "palette-classic"
                          },
                          "custom": {
                            "axisLabel": "",
                            "axisPlacement": "auto",
                            "barAlignment": 0,
                            "drawStyle": "line",
                            "fillOpacity": 0,
                            "gradientMode": "none",
                            "hideFrom": {
                              "legend": False,
                              "tooltip": False,
                              "viz": False
                            },
                            "lineInterpolation": "linear",
                            "lineWidth": 1,
                            "pointSize": 5,
                            "scaleDistribution": {
                              "type": "linear"
                            },
                            "showPoints": "auto",
                            "spanNones": False,
                            "stacking": {
                              "group": "A",
                              "mode": "none"
                            },
                            "thresholdsStyle": {
                              "mode": "off"
                            }
                          },
                          "mappings": [],
                          "thresholds": {
                            "mode": "absolute",
                            "steps": [
                              {
                                "color": "green",
                                "value": None
                              },
                              {
                                "color": "red",
                                "value": 80
                              }
                            ]
                          }
                        },
                        "overrides": []
                      },
                      "gridPos": {
                        "h": 9,
                        "w": 12,
                        "x": 0,
                        "y": 16
                      },
                      "id": 2,
                      "options": {
                        "legend": {
                          "calcs": [],
                          "displayMode": "list",
                          "placement": "bottom"
                        },
                        "tooltip": {
                          "mode": "single",
                          "sort": "none"
                        }
                      },
                      "targets": [
                        {
                          "datasource": {
                            "type": "prometheus",
                            "uid": "Y8Y4A4mVz"
                          },
                          "editorMode": "builder",
                          "expr": "node_load1",
                          "legendFormat": "__auto",
                          "range": True,
                          "refId": "A"
                        }
                      ],
                      "title": "Load 1 UVT K8S Grafana",
                      "type": "timeseries"
                    },
                    {
                      "cards": {},
                      "color": {
                        "cardColor": "#b4ff00",
                        "colorScale": "sqrt",
                        "colorScheme": "interpolateSpectral",
                        "exponent": 0.5,
                        "mode": "spectrum"
                      },
                      "dataFormat": "timeseries",
                      "datasource": {
                        "type": "prometheus",
                        "uid": "Y8Y4A4mVz"
                      },
                      "gridPos": {
                        "h": 8,
                        "w": 12,
                        "x": 12,
                        "y": 16
                      },
                      "heatmap": {},
                      "hideZeroBuckets": False,
                      "highlightCards": True,
                      "id": 8,
                      "legend": {
                        "show": False
                      },
                      "reverseYBuckets": False,
                      "targets": [
                        {
                          "datasource": {
                            "type": "prometheus",
                            "uid": "Y8Y4A4mVz"
                          },
                          "editorMode": "builder",
                          "expr": "node_entropy_available_bits",
                          "legendFormat": "__auto",
                          "range": True,
                          "refId": "A"
                        }
                      ],
                      "title": "Entropy Available Bits UVT K8S Grafana",
                      "tooltip": {
                        "show": True,
                        "showHistogram": False
                      },
                      "type": "heatmap",
                      "xAxis": {
                        "show": True
                      },
                      "yAxis": {
                        "format": "short",
                        "logBase": 1,
                        "show": True
                      },
                      "yBucketBound": "auto"
                    }
                  ],
                  },
                  "folderId": 0,
                  # "folderUid": "l3KqBxCMz",
                  "folderUid": None,
                  "message": "Dash generated by EDE",
                  "overwrite": True   # todo handle overwrite better, now is default, will retag existing dash
                }
        self.grafana_token = grafana_token
        self.grafana_url = grafana_url
        self. grafana = GrafanaApi.from_url(url=self.grafana_url, credential=self.grafana_token)
        self.dash_uid = None
        self.dash_url = None
        self.dash_id = None

    def get_dash(self, tag, working_dash=False):
        """
        gets dash information based on tag.
        :param tag: Tag to search for.
        :param working_dash: If dashboard is found and working_dash set to true it will be set as the current working dash
        :return: Dash UID, Dash URL, Dash ID
        """
        dashboards = self.grafana.search.search_dashboards(tag=tag)

        if not dashboards:
            logger.error('[{}] : [ERROR] No Grafana dashes found with tag: {}'.format(
              datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), tag))
            return 0, 0, 0
        dash_uid = dashboards[0]['uid']
        if len(dashboards) > 1:  # Check if more than one dash has the same tag and select the first one
            logger.warning('[{}] : [WARN] More then one dashboard found with the tag, selecting first dash with uid: {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), dash_uid))
        dash_url = dashboards[0]['url']
        dash_id = dashboards[0]['id']
        logger.info(
            '[{}] : [INFO] Dashboard found with uid {} and url {} for tag {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), dash_uid, dash_url, tag))
        if working_dash:
            self.dash_uid = dash_uid
            self.dash_url = dash_url
            self.dash_id = dash_id
        return dash_uid, dash_url, dash_id

    def generate_dash(self,
                      tag,
                      title="UVT Serrano dash",
                      timezone='browser',
                      refresh='10s',
                      dtime={'from': 'now-6h',
                            'to': 'now'}
                      ):
        """
        Generates dash descriptor
        :param tag: Tag to be used for dashboard
        :param title: Title of Dashboard
        :param timezone: Timezone to be used for
        :param refresh: Refresh rate
        :param dtime: Time interval to be shown by default
        :return:  Dash descriptor dictionary
        """
        dash_inf = self.get_dash(tag=tag, working_dash=False)
        if not dash_inf[1]:
            self.dash_dict['dashboard']['title'] = title
            self.dash_dict['dashboard']['timezone'] = timezone
            self.dash_dict['dashboard']['refresh'] = refresh
            self.dash_dict['dashboard']['time'] = dtime
            self.dash_dict['dashboard']['tags'] = [tag]  # TODO add support for more than one tag

            logger.warning(
              '[{}] : [WARN] No dash found with tag {} creating ...'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), tag))

        else:
            logger.warning(
              '[{}] : [WARN] Dash found with uid {} and url {} skipping generation of new dash'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), dash_inf[0], dash_inf[1]))
            return 0  # TODO enable override of existing dash
        return self.dash_dict

    def create_dash(self):
        """
        Creates dashboad based on dash JSON Descriptor
        :return:  dashboard UID, URL, ID
        """
        try:
            dash_inf = self.grafana.dashboard.update_dashboard(dashboard=self.dash_dict)
            self.dash_uid = dash_inf['uid']
            self.dash_url = dash_inf['url']
            self.dash_id = dash_inf['id']
            logger.info(
              '[{}] : [INFO] Created new dashboard with id {}, url {} and tag {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                self.dash_id, self.dash_url, "test_dash"))  # Todo default tag is set in template
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed to create dashboard with {} and {}'.format(
              datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            return 0
        return dash_inf

    def delete_dash(self, uid=None):
        """
        Deletes dashboard based on uid
        :param uid: UID of dashboard to b deleted, if not set the default dashboard will be deleted
        """
        dashboard_uid = 0
        try:
            if uid:
                dashboard_uid = uid
            else:
                dashboard_uid = self.dash_uid
            self.grafana.dashboard.delete_dashboard(dashboard_uid=dashboard_uid)
            logger.info(
              '[{}] : [INFO] Dashboard with uid {} deleted'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), dashboard_uid))
        except Exception as inst:
            logger.error(
              '[{}] : [ERROR] Failed deleting dashboard with uid {} with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), dashboard_uid,
                type(inst), inst.args))

    def push_annotation(self, time_from, time_to, anomaly_tags, message, dash_id=None):
        """
        Pushes annotations to grafana
        :param time_from: Start of annotation in utc
        :param time_to: End of annotation in utc
        :param anomaly_tags: Tag for the anomaly to be used
        :param message: Message when creating anomaly
        :param dash_id: ID of the dash where annotations are to be pushed
        :return: annotation descriptor
        """
        if self.dash_id is not None:
            logger.info('[{}] : [INFO] Dash id initialized to {}'.format(
              datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.dash_id))
            l_dash_id = self.dash_id
        elif dash_id is not None:
            logger.info('[{}] : [INFO] Dash id set to {}'.format(
              datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), dash_id))
            l_dash_id = dash_id
        else:
            logger.error('[{}] : [ERROR] Dash id not initialized or provided'.format(
              datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            return 0

        try:
            annotation = self.grafana.annotations.add_annotation(dashboard_id=l_dash_id,
                                                                 time_from=time_from,
                                                                 time_to=time_to,
                                                                 tags=anomaly_tags,
                                                                 text=message)
            logger.info(
              '[{}] : [INFO] Detected annotations {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), len(annotation)))
        except Exception as inst:
            logger.error('[{}] : [ERROR] "Failed to push annomalies with {} and {}'.format(
              datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            return 0
        return annotation

    def get_annotations(self,
                        time_from=None,
                        time_to=None,
                        alert_id=None,
                        dash_id=None,
                        panel_id=None,
                        user_id=None,
                        ann_type=None,
                        tags=None,
                        limit=None):
        """
        Wrapper for get_annotation method from grafana_client.

        :param time_from: Start of query period
        :param time_to:  End of query period
        :param alert_id: Alert ID
        :param dash_id: Dasboard ID
        :param panel_id: Panel ID
        :param user_id: User ID
        :param ann_type: Annotation type
        :param tags: Tags used
        :param limit: Limit
        :return: List of annotation descriptors
        """
        try:
            annotations = self.grafana.annotations.get_annotation(
              time_from=time_from,
              time_to=time_to,
              alert_id=alert_id,
              dashboard_id=dash_id,
              panel_id=panel_id,
              user_id=user_id,
              ann_type=ann_type,
              tags=tags,
              limit=limit)
        except Exception as inst:
            logger.error(
              '[{}] : [ERROR] Failed fetching annotations with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            return 0
        return annotations

    def delete_annotation(self, annotations):
        """
        Delete Annotations

        :param annotations: list of annotation descritors
        """
        try:
            for annotation in annotations:
                annotation_id = annotation['id']
                self.grafana.annotations.delete_annotations_by_id(annotations_id=annotation_id)
                logger.info(
                  '[{}] : [INFO] Deleting annotation with id {}, dashboard id {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), annotation_id, annotation['dashboardId']))
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed to delete annotation with {} and {}'.format(
              datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            return 0
        return 1


if __name__ == "__main__":

    # Test Credentials


    url = ""
    credential = ""

    test = EDEGrafanaDash(grafana_token=credential, grafana_url=url)

    print(test.get_dash(tag="test_tag"))

    print(test.generate_dash(tag="test_tag_41", title="Test gen 41"))
    print(test.create_dash())

    # Delete dash based on uid if left blank as initialized in object
    # print(test.delete_dash(uid='HoR8TpWVk'))

    # Testing anotation push
    utc_end = int(time.time() * 1000)
    utc_start = utc_end + 7000


    anomaly_tags = ['tagged_anomaly']
    push_msg = "This is a message used for pushed anomalies"

    # Dash id to be used
    dashinf = test.get_dash(tag="test_tag_41", working_dash=False)
    duid = dashinf[0]
    durl = dashinf[1]
    did = dashinf[2]

    print(did)
    test.push_annotation(utc_end, utc_start, anomaly_tags, push_msg, dash_id=did)


    print(test.get_annotations(dash_id=did))
    print(len(test.get_annotations(dash_id=did)))

   # Delete annotations
   #  test.delete_annotation(test.get_annotations(dash_id=did))
