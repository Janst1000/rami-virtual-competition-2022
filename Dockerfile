FROM ramimarinerobots/virtual-competition2022
USER root
RUN apt-get update
RUN apt-get install python3 python3-pip -y
RUN pip install --upgrade pip3

COPY src ./

RUN apt-get install build-essential apt-utils -y

#RUN pip3 install -r ./requirements.txt
CMD [ "python3", "-u", "main.py" ]
