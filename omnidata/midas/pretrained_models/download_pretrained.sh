##!/usr/bin/env bash
# Copied & modified from:
# https://github.com/EPFL-VILAB/omnidata/blob/main/omnidata_tools/torch/tools/

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
sudo apt-get install apt-transport-https ca-certificates gnupg
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
sudo apt-get update -y && sudo apt-get install -y google-cloud-sdk

sudo apt install -y imagemagick

pip install gdown

gdown https://drive.google.com/uc?id=1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI -O . # omnidata depth (v2)
gdown https://drive.google.com/uc?id=1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR -O . # omnidata normals (v2)