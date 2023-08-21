#process data to convert from xlsx to csv for reading to a Bayesian Network

import pandas as pd
import numpy as np

#list of category interested in collecting
dataList = ["Worker Program", 
            "Business", 
            "Economic - Total", 
            "Sponsored Family - Total", 
            "Resettled Refugee & Protected Person in Canada - Total"]

#convertion list to match between csv and dataframe
convertName = {"Worker Program":"Worker", 
            "Business":"Business", 
            "Economic - Total":"Economic", 
            "Sponsored Family - Total":"Sponsor", 
            "Resettled Refugee & Protected Person in Canada - Total":"Refugee"}
    

#convert numerical data containing "," to numbers without
def convertData(tempData):
    for i in range(len(tempData)):
        if len(tempData[i])>3:
            tempData[i] = tempData[i].replace(',','')
            
    tempData[tempData=='--'] = 0
    
    return np.int32(tempData)
    
#get the list of provinces/territories from the data file
def getProvince(column):
    prov = []
    for p in column:
        if type(p) == float:
            continue
        else:
            prov.append(p[:-8])
    
    province = prov[1:14]
    
    return province
    

#get the categories from the data file
def getCategories(columnCategories):
    categories = [[] for i in range(len(columnCategories))]
    
    for i in range(len(columnCategories)):
        for j in range(len(columnCategories[i])):
            if type(columnCategories[i][j]) == float:
                continue
            categories[i] = columnCategories[i][j]  
            
    return categories

#get the months from the data file
def getMonths(monthRow):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    m = [ monthRow[i] in months for i in range(len(monthRow))]
    
    return m

#extract the data from the data file
def extractData(path=r"EN_ODP-PR-ProvImmCat.xlsx"):
    data = np.array(pd.read_excel(path))
        
    provinceList = getProvince(data[:,0])
    categories = getCategories(data[:363,:4])
    
    
    month = getMonths(data[3,4:])
    
    #create empty dictionary to hold data
    provinceData = {p:{convertName[d]:0 for d in dataList} for p in provinceList}
    
    #iterate through each row in matrix and match it to the corresponding province/category
    provinceIndex = 0 
    for r in range(len(categories)): #loop through each category
        if provinceIndex >= len(provinceList):
            break
        
        #find the current province
        currentProvince = provinceList[provinceIndex]
        for pIndex, p in enumerate(dataList):           
            if categories[r] == p:
                seriesData = data[r,4:]
                
                tempData = convertData(seriesData[month])
                #save data to the province and category
                provinceData[currentProvince][convertName[p]] = tempData
           
        #find the final total value for current province
        provinceTotalStr = currentProvince + " - Total"
        if provinceTotalStr == categories[r]:
            seriesData = data[r,4:]
            
            tempData = convertData(seriesData[month])
            provinceData[currentProvince]["Total"] = tempData
            provinceIndex += 1
     
    #find the final value for all provinces
    provinceData["Total"] = convertData(data[362,4:][month])
    
    return provinceData

            
#save the extracted data in .csv files
def saveData(provinceData):
    
    root = r"data_csv2//"
    for p in provinceData.keys():
        df = pd.DataFrame(provinceData[p])
        
        df.to_csv(root + p + ".csv")
        
if __name__ == "__main__":
    
    provinceData = extractData(r"EN_ODP-PR-ProvImmCat.xlsx")
    saveData(provinceData)
    # f = pd.read_csv(root+"Alberta.csv")
    

