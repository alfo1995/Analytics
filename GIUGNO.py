#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 18:16:50 2019

@author: alfonsodamelio
"""

###### PRIMA PARTE ESAME GIUGNO



#1 parte

path='./orders.csv'  
      
def restituzione_id_prodotti_stessa_trans(path):
    
    #importo il dataset senza l'utilizzo di librerie di alto livello 
    with open(path,encoding='utf-8') as data:
        id_transazione=[]
        id_prodotto=[]
        name_prodotto=[]
        for i in data:
            doc=i.rstrip('\n')
            doc=doc.split(';')
            id_transazione.append(doc[0])
            id_prodotto.append(doc[1])
            name_prodotto.append(doc[2])
    
    #creo dizionario con chiavi uguali agli ID delle transazioni e valori =lista di prodotti acquistati in quella transazione
    dizionario_transazioni=dict()
    cont=0
    for i in id_transazione:
        if i not in dizionario_transazioni.keys():
            dizionario_transazioni[i]=[id_prodotto[cont]]
        else:
            dizionario_transazioni[i].append(id_prodotto[cont])
        cont+=1
    return(dizionario_transazioni)
  
    
#input id transazione   
id_input=input()

def check(id_input):        
    #restituzione
    if id_input in restituzione_id_prodotti_stessa_trans(path).keys():
        out=restituzione_id_prodotti_stessa_trans(path)[id_input]
        print()
        print('Questi sono i prodotti acquistati nella transazione %s:'%id_input)
        print('')
        print('\n'.join(out))
    
check(id_input)





#2 parte
    
#prodotto id input
input_prodotto_id=input()

output=[]
transazioni=[]
#utilizzo la funzione della prima parte
for i in restituzione_id_prodotti_stessa_trans(path):
    if input_prodotto_id in restituzione_id_prodotti_stessa_trans(path)[i]:
        transazioni.append(i)
        for k in restituzione_id_prodotti_stessa_trans(path)[i]:
            output.append(k)
        output.remove(input_prodotto_id)
print()
print('Gli id dei prodotti acquistati insieme al prodotto con id '+input_prodotto_id+' sono:')
print()
print('\n'.join(output))
print()
print('Trovati nelle transazioni:')
print('\n'.join(transazioni))

    
    
    
    
    
    
    