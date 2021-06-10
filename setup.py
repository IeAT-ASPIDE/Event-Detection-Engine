from setuptools import setup

setup(
    name='Event-Detection-Engine',
    version='0.5',
    packages=['exp', 'data', 'misc', 'test', 'models', 'edeuser', 'edeweka', 'queries', 'service', 'edengine',
              'edepoint', 'edescikit', 'edefeature', 'edeformater', 'experiments', 'ederulengine',
              'ederulengine.ruledefs', 'edetensorflow'],
    url='https://www.aspide-project.eu/',
    license='Apache License 2.0',
    author='Gabriel Iuhasz',
    author_email='iuhasz.gabriel@e-uvt.ro',
    description='Event Detection Engine'
)
