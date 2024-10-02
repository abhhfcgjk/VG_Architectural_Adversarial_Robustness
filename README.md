
# DBCNN-Pytorch
[DBCNN](https://github.com/zwx8981/DBCNN-PyTorch) with CayleyBlocks modifications.
## Options
```bash
python DBCNN.py # Dafault DBCNN
```
```bash
python DBCNN.py --cayley # CayleyBlock before last convolution in VGG16
```
```bash
python DBCNN.py --cayley2 # CayleyBlock before last convolutions in S-CNN and VGG16
```
```bash
python DBCNN.py --cayley3 # CayleyBlock before last convolution block in VGG16
```

# Build
Load dataset [KonIQ-10k](https://database.mmsp-kn.de/koniq-10k-database.html).
Clone repository
```bash
git clone -b DBCNN https://github.com/abhhfcgjk/VG_Architectural_Adversarial_Robustness.git
```
Make the softlink to KonIQ-10k folder
```bash
cd VG_Architectural_Adversarial_Robustness
ln -s <KonIQ-10k folder> ./dataset/KonIQ-10k
```
Build environment and run train script
```bash
nohup uv run DBCNN.py > DBCNN.out&
nohup uv run DBCNN.py --cayley > DBCNN_cayley.out&
nohup uv run DBCNN.py --cayley2 > DBCNN_cayley2.out&
nohup uv run DBCNN.py --cayley3 > DBCNN_cayley3.out&
```