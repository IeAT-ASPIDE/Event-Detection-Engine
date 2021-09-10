"""
Copyright 2021, Institute e-Austria, Timisoara, Romania
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

from fpdf import FPDF
from glob import glob
import argparse


class EDEPDF(FPDF):
    def __init__(self, folder, name, model=None, type=None):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        self.folder = folder
        self.name = name
        self.model = model
        self.type = type

    def header(self):
        # Custom logo and positioning
        # Create an `assets` folder and put any wide and short image inside
        # Name the image `logo.png`
        self.image('assets/aspide.png', 8, 6, 33)
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, f'EDE {self.name}', 0, 0, 'R')
        self.ln(20)

    def footer(self):
        # Page numbers in the footer
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, 'Page ' + str(self.page_no()), 0, 0, 'C')

    def page_body(self, image):
        # Determine how many plots there are per page and set positions
        # and margins accordingly
        self.image(image, 15, 25, self.WIDTH - 30)
        # self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        # self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)


    def analysis_feature_plot(self, images, classif=False):
        if len(images) == 3:
            self.image(images[0], 15, 25, self.WIDTH - 30)
            self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
            self.image(images[2], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
        elif len(images) == 2:
            if classif:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 90, self.WIDTH - 30)
            else:
                self.image(images[0], 15, 25, self.WIDTH - 30)
                self.image(images[1], 15, self.WIDTH / 2 + 5, self.WIDTH - 30)
        elif len(images) == 1:
            self.image(images[0], 15, 25, self.WIDTH - 30)

    def __split_list(self, lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def print_page(self, image):
        # Generates the report
        if isinstance(image, int):
            print("Received no image! Skipping")
        else:
            try:
                self.add_page()
                self.page_body(image)
            except Exception as inst:
                print(f"Exception while adding image! With {type(inst)}, {inst.args}")

    def print_analysis_feature_plot(self, images, classif=False):
        self.add_page()
        self.analysis_feature_plot(images, classif=classif)  # todo remove

    def generate_classification_report(self):
        list_images = glob(self.folder)
        print(list_images)
        diagrams = {
            'confusion_matrix': [],
            'feature_importance': [],
            'learning_curve': 0,
            'validation_curve': 0,
            'rfe': 0,
            'prc': 0,
            'auc': 0,
            'decision_boundary': 0
        }
        # print(list_images)
        for elem in list_images:
            if "CM_" in elem:
                diagrams['confusion_matrix'].append(elem)
                # print(elem.split("_")[-1].split(".")[0])
            elif "FI_" in elem:
                diagrams['feature_importance'].append(elem)
            elif "Learning_Curve_" in elem:
                diagrams['learning_curve'] = elem
            elif "Validation_Curve_" in elem:
                diagrams['validation_curve'] = elem
            elif "Recursive_Feature_Elimination_" in elem:
                diagrams['rfe'] = elem
            elif "Precision_Recall_Curve_" in elem:
                diagrams['prc'] = elem
            elif "ROCAUC_Curve_" in elem:
                diagrams['auc'] = elem
            elif "Decision Boundary_" in elem:
                diagrams['decision_boundary'] = elem

        # Names Sorting based on fold non lexical
        diagrams['confusion_matrix'].sort(key=lambda fname: int(fname.split('_')[-1].split('.')[0]))
        diagrams['feature_importance'].sort(key=lambda fname: int(fname.split('_')[-1].split('.')[0]))
        for cm, fi in zip(diagrams['confusion_matrix'], diagrams['feature_importance']):
            self.print_analysis_feature_plot([cm, fi], classif=True)

        self.print_page(diagrams['learning_curve'])
        self.print_page(diagrams['validation_curve'])
        self.print_page(diagrams['rfe'])
        self.print_page(diagrams['prc'])
        self.print_page(diagrams['auc'])
        self.print_page(diagrams['decision_boundary'])

    def generate_clustering_report(self):
        list_images = glob(self.folder)
        diagrams = {
            'feature_separation': [],
            'decision_boundary': 0,
            'other': []
        }

        # Split Analysis Reports
        for elem in list_images:

            if "Feature_Separation_" in elem:
                diagrams['feature_separation'].append(elem)
            elif 'Decision_Boundary' in elem:
                diagrams['decision_boundary'] = elem
            else:
                diagrams['other'].append(elem)

        self.print_page(diagrams['decision_boundary'])

        feature_outputs = self.__split_list(diagrams['feature_separation'], 2)
        for feature_output in feature_outputs:
            self.print_analysis_feature_plot(feature_output)

    def generate_analysis_report(self):
        list_images = glob(self.folder)
        diagrams = {
            'feature_plots': [],
            'pearson': [],
            'pca': [],
            'manifold': [],
            'rank1d': [],
            'spearman': [],
            'other': []
        }
        # Split Analysis Reports
        for elem in list_images:
            if "Feature_plot_" in elem.split('/')[-1]:
                diagrams['feature_plots'].append(elem)

            elif 'Pearson_' in elem.split('/')[-1]:
                diagrams['pearson'].append(elem)

            elif 'PrincipalComponent' in elem.split('/')[-1]:
                diagrams['pca'].append(elem)

            elif 'Manifold_' in elem.split('/')[-1]:
                diagrams['manifold'].append(elem)

            elif 'Rank1D_' in elem.split('/')[-1]:
                diagrams['rank1d'].append(elem)

            elif 'Correlation_spearman' in elem.split('/')[-1]:
                diagrams['spearman'].append(elem)
            else:
                diagrams['other'].append(elem)

        self.print_page(diagrams['pearson'][0])
        self.print_page(diagrams['pca'][0])
        self.print_page(diagrams['manifold'][0])
        self.print_page(diagrams['rank1d'][0])
        self.print_page(diagrams['spearman'][0])

        feature_outputs = self.__split_list(diagrams['feature_plots'], 3)
        for feature_output in feature_outputs:
            self.print_analysis_feature_plot(feature_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EDE Report Generator')
    parser.add_argument('--report', type=str, nargs='?',
                        help='Define type of report; analysis, classification, clustering')
    args = parser.parse_args()
    if args.report == 'analysis':
        print("Generating Analysis report ... ")
        analysis_folder = "/Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis/*.png"
        pdf = EDEPDF(folder="/Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis/*.png",
                     name="Analysis")
        pdf.generate_analysis_report()
        pdf.output('EDE_Analysis_Repot.pdf', 'F')
        print("Report generated!")
    elif args.report == 'classification':
        print("Generating Classification report ... ")
        model_folder = "/Users/Gabriel/Documents/workspaces/Event-Detection-Engine/models/*.png"
        # Classification
        pdf = EDEPDF(folder=model_folder,
                     name="Classification Analysis")
        pdf.generate_classification_report()
        pdf.output('EDE_Classification_Analysis.pdf', 'F')
        print("Report generated!")
    elif args.report == 'clustering':
        print("Generating Clustering report ... ")
        model_folder = "/Users/Gabriel/Documents/workspaces/Event-Detection-Engine/models/*.png"
        # Clustering
        pdf = EDEPDF(folder=model_folder,
                     name="Clustering Analysis")
        pdf.generate_clustering_report()
        pdf.output('EDE_Clustering_Analysis.pdf', 'F')
        print("Report generated!")
    else:
        parser.error(f"Invalid report type: {args.report}")



