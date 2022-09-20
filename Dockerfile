FROM --platform=linux/amd64 python:3.8

RUN python3 -m pip install numpy scikit-learn pytest
