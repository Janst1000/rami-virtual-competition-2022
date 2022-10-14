FROM ramimarinerobots/virtual-competition2022
USER root
RUN apt-get update
RUN apt-get install python3 python3-pip -y


COPY src ./

CMD [ "python3", "-u", "main.py" ]
