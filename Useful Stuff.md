```python
train=pd.read_csv('Admission_Predict.csv') #Reading
train.drop(columns=['Serial No.'],axis=1,inplace=True) #Dropping
test=pd.read_csv('drugsComTest_raw.csv',usecols = ['drugName','condition','rating','usefulCount']) #Using col.
```
