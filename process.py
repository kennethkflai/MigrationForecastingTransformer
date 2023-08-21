import pandas as pd
import numpy as np
# from datasets import Dataset, DatasetDict

path = r"EN_ODP-PR-ProvImmCat.xlsx"

data = np.array(pd.read_excel(path))

firstColumn = data[:,0]

prov = []

for p in firstColumn:
    if type(p) == float:
        continue
    else:
        prov.append(p[:-8])

province = prov[1:14]


dataProvince = [[] for i in range(len(province))]

monthRow = data[3,4:]
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
m = [ monthRow[i] in months for i in range(len(monthRow))]

for r in range(len(firstColumn)):
    for pIndex, p in enumerate(province):
        p = p + " - Total"
        if firstColumn[r] == p:
            seriesData = data[r,4:]
            
            tempData = seriesData[m]
            for i in range(len(tempData)):
                if len(tempData[i])>3:
                    tempData[i] = tempData[i].replace(',','')
                    
            tempData[tempData=='--']=0  
            dataProvince[pIndex] = np.int32(tempData)
            
            
d = pd.DataFrame(dataProvince, index=province)

t = {}
month = 1
year = 2015
for i in range(100):
    t[i] = f"{year}-{month}-01 00:00:00"
    month +=1
    if month > 12:
        year += 1
        month = 1
        
d["item_id"] = province
import datetime
startTime = [datetime.datetime(2015,1,1,0,0,0) for j in range(len(province))]
d["start"] = startTime[0]
d["target"]= d[[i for i in range(100)]].values.tolist()
d["feat_static_cat"] = [[j] for j in range(len(province))]
d["feat_dynamic_cat"] = [None for j in range(len(province))]

d2 = d.iloc[:,100:]
d2 = d2.reset_index(drop=True)

d1 = d2.copy()
d1["target"] = d[[i for i in range(100-12)]].values.tolist()
# test = Dataset.from_pandas(d1)
# train = Dataset.from_pandas(d2)
# tds.save_to_disk("data")

