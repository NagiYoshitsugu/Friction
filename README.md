# Friction
To analyze friction data

## Preparation
1. Anacondaをインストール  
https://www.anaconda.com/products/distribution
1. 仮想環境を再構築  
```
conda env create -f=env_friction_2.yml
```

## Usage
1. ディレクトリ構造
```
  WD  
  |__Analysis  
  |__|__*.ipynb
  |__Data  
  |  |__*Raw data directory  
  |  |__*conditions.csv  
  |__Questionnaire  
```
1. データを格納  
  1.1 トリニティデータ  
    * Dataディレクトリ内にデータを格納
    * [date]_conditions.csvを作成→例
1. 仮想環境をアクティベート
```
conda activate friction_2
```
1. notebookにパラメータ入力
