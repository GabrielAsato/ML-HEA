'''
Calculate heat of mixing [kJ/mole] of binary metal system based on Miedema scheme
Writen by Jun Ou
Adapted version
GitHub: https://github.com/JunOu/HmixCalculator/tree/master
'''
import numpy as np

# Variables
A_phiStar = 0.0
A_nws13 = 0.0
A_Vm23 = 0.0
aA = 0.0
A_name = ''
B_phiStar = 0.0
B_nws13 = 0.0
B_Vm23 = 0.0
aB = 0.0
B_name = ''
deH_A_partial_infDilute = 0.0
P = 0.0
RP = 0.0
QP = 9.4
e = 1.0
elementName = {}
elementPhiStar = {}
elementNWS13 = {}
elementVM23 = {}
elementRP = {}
elementTRAN = {}
Avogardro = 6.02E23 # unit /mole
xA = np.linspace(0.001,0.999,200)
xAs = np.empty(len(xA))
fxs = np.empty(len(xA))
g = np.empty(len(xA))
deHmix = np.empty(len(xA))
xB = 1.0 - xA        

# Type the symbol of the element
def calHmix(A_name, B_name):
    try:
        fReadDatabase = open('./database.dat','r')
        for line in fReadDatabase:
            elementTmp = line.replace('\n',' ').replace(' ','').split(',')
            
            #print elementTmp
            nameTmp = elementTmp[0]
            elementName[nameTmp] = elementTmp[0]                
            elementPhiStar[nameTmp] = elementTmp[1]                
            elementNWS13[nameTmp] = elementTmp[2]
            elementVM23[nameTmp] = elementTmp[3]
            elementRP[nameTmp] = elementTmp[4]
            elementTRAN[nameTmp] = elementTmp[5]
        print('succesful initialization of the database.')    
    except ValueError:
        print('database Error')

    #####
    if elementTRAN[A_name]=='T' and elementTRAN[B_name]=='T':
        RP = 0.0
    elif elementTRAN[A_name]=='N' and elementTRAN[B_name]=='N':
        RP = 0.0
    else:
        RP = float(elementRP[A_name])*float(elementRP[B_name])*0.73
            
    #####
    if elementTRAN[A_name]=='T' and elementTRAN[B_name]=='T':
        P = 0.147
    elif elementTRAN[A_name]=='N' and elementTRAN[B_name]=='N':
        P = 0.111
    else:
        P = 0.128            

    #####
    A_phiStar = float(elementPhiStar[A_name])
    B_phiStar = float(elementPhiStar[B_name])
    A_nws13 = float(elementNWS13[A_name])
    B_nws13 = float(elementNWS13[B_name])
    A_Vm23 = float(elementVM23[A_name])
    dePhi = A_phiStar-B_phiStar
    deNws13 = A_nws13-B_nws13
    deH_A_partial_infDilute = 2.0*A_Vm23/(1.0/A_nws13+1.0/B_nws13)*Avogardro*P*(-e*(dePhi)**2+QP*(deNws13)**2-RP)*1.60217657E-22

    return deH_A_partial_infDilute