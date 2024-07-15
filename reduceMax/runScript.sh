#!/bin/bash

make

./cudaReduceMax 1000 30 > teste0.txt
./cudaReduceMax 1000000 30 > teste1.txt
./cudaReduceMax 16000000 30 > teste2.txt
./cudaReduceMax 40000000 30 > teste3.txt