This repo provides a collection of baselines for DGraphFin dataset. Please download the dataset from the [DGraph](http://dgraph.xinye.com) web and place it under the folder './dataset/DGraphFin/raw'.  

## Environments
Implementing environment:  
- numpy = 1.23.1  
- pytorch = 1.10.1  
- torch_geometric = 2.0.4  
- torch_scatter = 2.0.9  
- torch_sparse = 0.6.13  

- GPU: Tesla A100 40G  


## Training

- **GEARSage**
```bash
python main.py
```

## Results:
Performance on **DGraphFin**(10 runs):
0.8387553562234347 0.00019525288374594048
| Methods  | Params | Valid AUC       | Test AUC        |
| :------- | ------ | --------------- | --------------- |
| GEARSage | 50544  | 0.8388 ± 0.0002 | 0.8460 ± 0.0002 |