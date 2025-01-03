from setuptools import setup, find_packages

setup(
    name='Insurance_Price_Prediction',
    version='1.0.0',
    packages=find_packages(where=['src', 'tests']),
    install_requires=[
        'mlflow==2.19.0',
        'cloudpickle==3.0.0',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'psutil==5.9.0',
        'scikit-learn==1.5.1',
        'scipy==1.13.1',
        'xgboost==2.1.1',
        'flask==3.0.3',
        'pytest==8.3.4'
    ],
)