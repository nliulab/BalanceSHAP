from setuptools import setup, find_packages

setup(
    name="balanceSHAP",
    version="1.0",
    author="Mingxuan Liu",
    author_email="m.liu@duke-nus.edu.sg",
    description="Use balanced SHAP for explanations with imbalance data",
    # url="http://iswbm.com/", 
    packages=find_packages(),
    install_requires=['shap'],
    classifiers = [
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ]
)