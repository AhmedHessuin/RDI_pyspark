# RDI pyspark repo
## enviorment
setup can be done as
```
conda create -n pyspark python=3.8.0
conda activate pyspark
pip install pyspark
pip install pandas
```

## Train
train is made with logestic regrssion, you can start train with 
```
python main.py 1 1 1 "<data-base-path-as-sqlite>"
```
flags explaination
```

1 means we will create the .csv file from pyspark to load the data, 0 means pyspark load the .csv file directly (note the file .csv is static named in the script )
1 means we will train on the 90% of the data generating 5 models with different itterations  (  saved_model_iteration_{i} i range is from 1 to 5), 0 means no train on this data set 
1 means we will evalute on the data using the models we generated in the train stage 
<data-base-path-as-sqlite>  the data base path if we will generate the .csv file

train clean the data by 
* drop any row with Score values larger than 5
* drop any Non value in Score colum by setting the Score colum data type as int 
* drop any text feed back taht is larger than 2000 word 

train tokenize the text feed back from customer 

train is made with logestic regression model using only the text column features 

## Test 
test is made by loading the model and inference on the data 
```
python inf.py 1 "<data-base-path-as-sqlite>" "<model-path>"
```
1 means we will create the .csv file based on the <data-base-path-as-sqlite>
<data-base-path-as-sqlite> the data base path 
<model-path> the model path 

inference is made by making all data as test data then evaluate on it 

#Data base file and re-run experiment 
[link](https://drive.google.com/file/d/1aN1Ln4exHJzI7vL9xN8Pg4YqYer7_ZdZ/view?usp=sharing)
you can download it in the same directory 
for train
```
python main.py 1 1 1 database.sqlite 
```
for test
```
python inf.py 1 database.sqlite saved_model_iteration_3
```
