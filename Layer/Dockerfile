FROM amazonlinux:2023

# Install Python 3.9 and pip

RUN dnf install -y python3.11 python3.11-pip zip

WORKDIR /lambda-package

# Install dependencies into the folder

RUN python3.11 -m pip install pydantic==2.5.3 pydantic-core==2.14.6 -t .
RUN python3.11 -m pip install jiter -t .


COPY portkey_ai ./portkey_ai
COPY lambda_function.py .

RUN zip -r /lambda-package.zip .
