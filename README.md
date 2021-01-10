# MAGCN
MAGCN: A Multi-Adaptive Graph Convolutional Network for Traffic Forecasting

# Prerequisites
- Python3( >3.5)
- [Pytorch](http://pytorch.org/) (>1.0)
- Other Python libraries can be installed by 
`pip install -r requirements.txt`

# Data Preparation
- Uncompress data file using `tar -zxvf data.tar.gz`

        -data\  
          -PEMS03\  
            -pems03.csv
            -pems03.npz
            -pems03.txt
            -pems03_data.csv
          -PEMS07\  
            -pems07.csv
            -pems07.npz
    
 - Preprocess the data with
  
    `python data_gen/pems_gendata.py --data_path ./data/PEMS03 --out_folder ./data/PEMS03 --config ./config/pems03/pems03_train.yaml`
    
    `python data_gen/pems_gendata.py --data_path ./data/PEMS07 --out_folder ./data/PEMS07 --config ./config/pems07/pems07_train.yaml`
    
# Training & Testing
## Training
Change the config file depending on what you want.


    `python main.py --config ./config/pems03/pems03_train.yaml`
    
    `python main.py --config ./config/pems07/pems07_train.yaml`
## Testing
To evaluate model on PEMS03, run 

    `python main.py --config ./config/pems03/pems03_test.yaml`
To evaluate model on PEMS07, run 

    `python main.py --config ./config/pems07/pems07_test.yaml`