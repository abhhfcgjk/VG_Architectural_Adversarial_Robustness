#!/bin/bash

# python DBCNN.py > DBCNN.out&
python DBCNN.py --cayley > DBCNN_cayley.out&
python DBCNN.py --cayley2 > DBCNN_cayley2.out&
python DBCNN.py --cayley3 > DBCNN_cayley3.out&
python DBCNN.py --cayley4 > DBCNN_cayley3.out&
python DBCNN.py --activation relu_elu > DBCNN_relu_elu.out&
python DBCNN.py --gr > DBCNN_gr.out&