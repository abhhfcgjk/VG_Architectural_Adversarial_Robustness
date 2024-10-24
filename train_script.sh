#!/bin/bash

nohup uv run DBCNN.py > DBCNN.out&
nohup uv run DBCNN.py --cayley > DBCNN_cayley.out&
nohup uv run DBCNN.py --cayley2 > DBCNN_cayley2.out&
nohup uv run DBCNN.py --cayley3 > DBCNN_cayley3.out&
nohup uv run DBCNN.py --cayley4 > DBCNN_cayley3.out&
nohup uv run DBCNN.py --activation relu_elu > DBCNN_relu_elu.out&