#!/bin/bash
fileid="1c5t1pfeggKgDNcCWJkP0GkouOFHN-IqL"
filename="checkpoint.pth"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
