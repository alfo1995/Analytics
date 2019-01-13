#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 09:28:02 2018

@author: alfonsodamelio
"""


################################ GIUGNO 1 PARTE  ########################################################

with open('/Users/alfonsodamelio/Downloads/orders-test.csv') as f:
    tran_id = []
    prod_id = []
    name_prod=[]
    prod_name=[]
    
    for line in f:
        c=line.rstrip()
        words = c.split(';')
        #print(words)
        tran_id.append(words[0])
        prod_id.append(words[1])
        name_prod.append(words[2])






#################### primo esercizio
dictio=dict()
cont=0
for i in tran_id:
    if i not in dictio.keys():
        dictio[i]=[(prod_id[cont],name_prod[cont])]
    else:
        dictio[i].append((prod_id[cont],name_prod[cont]))
    cont+=1
     


def find_prod_id(transaction):
    print('These are the products ids refers to transaction: '+transaction)
    print()
    print("\n".join([str(i) for i in (dictio[transaction])]))
  

find_prod_id(input())



#################### secondo esercizio
dictio2=dict()
cont=0
for i in tran_id:
    if i not in dictio2.keys():
        dictio2[i]=[prod_id[cont]]
    else:
        dictio2[i].append(prod_id[cont])
    cont+=1   
    

product=input()

for i in dictio2.keys():
    if product in dictio2[i]:
        print('product id is: '+product)
        print()
        print('transaction id is: '+i)
        print()
        other=[i for i in dictio2[i]]
        other.remove(product)
        print('Other product id appears in the same transaction of \ninput product id: ')
        print("\n".join(other))
        
       
        
        
        
        
        
        

################################ GIUGNO 1 PARTE  ########################################################


with open('/Users/alfonsodamelio/Downloads/orders1.csv') as f:
    tran_id=[]
    prod_id=[]
    name=[]
    for line in f:
        c=line.rstrip()
        word=c.split(';')
        tran_id.append(word[0])
        prod_id.append(word[1])
        name.append(word[2])
        


with open('/Users/alfonsodamelio/Downloads/orders2.csv') as f2:
    tran_id2=[]
    prod_id2=[]
    name2=[]
    for line in f2:
        c2=line.rstrip()
        word2=c2.split(';')
        tran_id2.append(word2[0])
        prod_id2.append(word2[1])
        name2.append(word2[2])




dict1=dict()
dict2=dict()

cont=0
for i in tran_id:
    if i not in dict1.keys():
        dict1[i]=[prod_id[cont]]
    else:
        dict1[i].append(prod_id[cont])
    cont+=1
        



cont2=0
for i in tran_id2:
    if i not in dict2.keys():
        dict2[i]=[prod_id2[cont2]]
    else:
        dict2[i].append(prod_id2[cont2])
    cont2+=1
        



#dizionari transazioni a coppie

dict1_couple=dict()
couple=[]
cont=0
for i in tran_id:
    for j in range(0,len(dict1[i])):
        for k in range(j+1,len(dict1[i])):
            couple.append((dict1[i][j],dict1[i][k]))
    dict1_couple[i]=couple
    couple=[]
    
    
    
dict2_couple=dict()
couple=[]
cont=0
for i in tran_id2:
    for j in range(0,len(dict2[i])):
        for k in range(j+1,len(dict2[i])):
            couple.append((dict2[i][j],dict2[i][k]))
    dict2_couple[i]=couple
    couple=[]


def reverse_tuple(tup):
    j=(tup[1],tup[0])
    return j


#now last step...find out couple
values=[i for i in dict2_couple.values()] 
 
couple=[]          
for i in tran_id:
    for j in dict1_couple[i]:
        for k in values:
            if j in k:
                couple.append(j)
            elif reverse_tuple(j) in k:
                couple.append(j)


couple=[i for i in set(couple)] 

prod_fin=prod_id+prod_id2
name_fin=name+name2
name_product=dict(zip(prod_fin,name_fin))

name_couple=[]
for i in couple:
    name=(name_product[i[0]],name_product[i[1]])
    name_couple.append(name)
    
    
    
    


################################################
def applicazione(path): 

    with open(path) as f:
        id_=[]
        importo=[]
        soggetto=[]
        benefi=[]
        tipo=[]
        for line in f:
            c=line.rstrip()
            word=c.split(';')
            id_.append(word[0])
            importo.append(word[1])
            soggetto.append(word[2])
            benefi.append(word[3])
            tipo.append(word[4])
            
    
    #creo dizionario con i beneficiari ----> id benificiario: importo #tutti i soldi presi
    beneficiario=dict()
    count=0
    for i in benefi:
        if i not in beneficiario.keys():
            beneficiario[i]=[importo[count]]
        else:
            beneficiario[i].append(importo[count])
        count+=1
            
    # dizionario soggetto ----> id: importo soldi da sottrarre
    soggetto_trans=dict()
    count=0
    for i in soggetto:
        if i not in soggetto_trans.keys():
            soggetto_trans[i]=[importo[count]]
        else:
            soggetto_trans[i].append(importo[count])
        count+=1
    
    del beneficiario['BENEFICIARIO']
    del soggetto_trans['SOGGETTO']
    
    
    
    all_account=soggetto+benefi
    all_account=[i for i in set(all_account)]
    
    
    transaction=dict()
    for i in all_account:
        if i in beneficiario.keys() :
            transaction[i]=sum(list(map(int,beneficiario[i])))
    
    for i in all_account:
        if i in soggetto_trans.keys():
            if i not in transaction.keys():
                transaction[i]=sum(list(map(int,soggetto_trans[i])))
            else:
                transaction[i]-=sum(list(map(int,soggetto_trans[i])))
    del transaction['']
    print('Application resume:')
    print()
    for key, value in transaction.items():
        print('utente: '+str(key)+' importo: '+str(value)+' £')
    
    #check max transaction and return id
    max=0
    _id=''
    for i in transaction.keys():
        if transaction[i]>max:
            max=transaction[i]
            _id=i
    print()
    print("l'utente che ha movimentato piu soldi è: "+_id+" con importo: "+str(max))
    print()
        
    with open(path) as f:
        for line in f:
            c=line.rstrip()
            word=c.split(';')
            if (word[2])==_id:
                print('id transazione: '+word[0]+' ----> importo: '+word[1]+'£ soggetto: '+word[2]+' con beneficiario: '+word[3]+' tipo: '+word[4] )
                print()
            if (word[3])==_id:
                print('id transazione: '+word[0]+' ----> importo: '+word[1]+'£ soggetto: '+word[2]+' con beneficiario: '+word[3]+' tipo: '+word[4] )
                print()
        
        
  
    
path='/Users/alfonsodamelio/Downloads/transazioni.csv'  
applicazione(path)

    

    


