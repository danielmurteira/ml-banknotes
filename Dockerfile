FROM python:slim

RUN pip install numpy
RUN pip install -U scikit-learn
RUN pip install matplotlib

COPY . /home
CMD cd /home && python banknote_authentication.py