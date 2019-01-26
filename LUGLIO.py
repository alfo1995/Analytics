#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 11:26:31 2019

@author: alfonsodamelio
"""
def creazione_dizionario(path):
    with open(path,encoding='utf-8') as data:
        id_transazione=[]
        id_prodotto=[]
        nome=[]
        for i in data:
            strip=i.strip('\n')
            split=strip.split(';')
            id_transazione.append(split[0])
            id_prodotto.append(split[1])
            nome.append(split[2])
  
            
    #creo dizionario con chiave uguale alla transazione e valore uguale alla lista di id prodotti.
            
    dizionario=dict()
    cont=0
    for i in id_transazione:
        if i not in dizionario.keys():
            dizionario[i]=[id_prodotto[cont]]
        else:
            dizionario[i].append(id_prodotto[cont])
        cont+=1
        
    #creo le coppie in ciascuna lista  
    value=[]
    for k in dizionario.keys():
        couple=[]
        for i in range(0,len(dizionario[k])):
            for j in range(i+1,len(dizionario[k])):
                couple.append((dizionario[k][i],dizionario[k][j]))
                if path=='./orders1.csv':
                    couple.append((dizionario[k][j],dizionario[k][i]))

        value.append(couple)
      
    chiavi=[i for i in dizionario.keys()]
    
    #creo nuovamente il dizionario adesso con le coppie come valori
    dizionario_coppie=dict(zip(chiavi,value))
    return id_prodotto,nome,dizionario_coppie





prodotto1,nome,dizionario_orders1=creazione_dizionario('./orders1.csv')

prodotto2,nome2,dizionario_orders2=creazione_dizionario('./orders2.csv')



#creo dizionario dove ad ogni id prodotto corrisponde il suo nome
prodotto_final=prodotto1+prodotto2
nome_final=nome+nome2

dizionario_id_nome=dict(zip(prodotto_final,nome_final))

######################### PROGRAMMA ROUTINE  #########################
prodotti_dizio2=[k for k in dizionario_orders2.values()]

out=[]
for transazione in dizionario_orders1.keys():
    for prodotto in  dizionario_orders1[transazione]:
        for k in prodotti_dizio2:
            if prodotto in k:
                out.append(prodotto)
nomi_out=[]
for i,j in out:
    nomi_out.append((dizionario_id_nome[i],dizionario_id_nome[j]))

print('Le coppie di prodotti trovati sono: ')
print(*nomi_out)