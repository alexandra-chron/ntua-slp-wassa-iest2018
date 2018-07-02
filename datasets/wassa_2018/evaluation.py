#!/usr/bin/env python
# Author: roman.klinger@ims.uni-stuttgart.de
# Evaluation script for IEST at WASSA 2018
from __future__ import print_function
import sys
import itertools

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def welcome():
    eprint("====================================")
    eprint("Evaluation script v0.2 for the Implicit Emotions Shared Task 2018.")
    eprint("Please call it via")
    eprint("./evaluate-iest.py <gold> <prediction>")
    eprint("where each csv file has labels in its first column.")
    eprint("The rows correspond to each other (1st row in <gold>")
    eprint("is the gold label for the 1st column in <prediction>).")
    eprint("")
    eprint("If you have questions, please contact klinger@wassa2018.com")
    eprint("====================================\n\n")

def checkParameters():
    if ((len(sys.argv) < 3 or len(sys.argv) > 3)):
        eprint("Please call the script with two files as parameters.")
        sys.exit(1)

def readFileToList(filename):
    eprint("Reading data from",filename)
    f=open(filename,"r")
    lines=f.readlines()
    result=[]
    for x in lines:
        result.append(x.split('\t')[0].rstrip())
    f.close()
    eprint("Read",len(result),"labels.")
    return result

def calculatePRF(gold,prediction):
    # initialize counters
    labels = set(gold+prediction)
    print("Labels: "+';'.join(labels))
    tp = dict.fromkeys(labels, 0.0)
    fp = dict.fromkeys(labels, 0.0)
    fn = dict.fromkeys(labels, 0.0)
    precision = dict.fromkeys(labels, 0.0)
    recall = dict.fromkeys(labels, 0.0)
    f = dict.fromkeys(labels, 0.0)
    # check every element
    for g,p in itertools.izip(gold,prediction):
        #        print(g,p)
        # TP 
        if (g == p):
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    # print stats
    print("Label\tTP\tFP\tFN\tP\tR\tF")
    for label in labels:
        recall[label] = 0.0 if (tp[label]+fn[label]) == 0.0 else (tp[label])/(tp[label]+fn[label])
        precision[label] = 1.0 if (tp[label]+fp[label]) == 0.0 else (tp[label])/(tp[label]+fp[label])
        f[label] = 0.0 if (precision[label]+recall[label])==0 else (2*precision[label]*recall[label])/(precision[label]+recall[label])
        print(label+
            "\t"+str(int(tp[label]))+
            "\t"+str(int(fp[label]))+
            "\t"+str(int(fn[label]))+
            "\t"+str(round(precision[label],3))+
            "\t"+str(round(recall[label],3))+
            "\t"+str(round(f[label],3))
            )
        # micro average
        microrecall = (sum(tp.values()))/(sum(tp.values())+sum(fn.values()))
        microprecision = (sum(tp.values()))/(sum(tp.values())+sum(fp.values()))
        microf = 0.0 if (microprecision+microrecall)==0 else (2*microprecision*microrecall)/(microprecision+microrecall)
    # Micro average
    print("MicAvg"+
        "\t"+str(int(sum(tp.values())))+
        "\t"+str(int(sum(fp.values())))+
        "\t"+str(int(sum(fn.values())))+
        "\t"+str(round(microprecision,3))+
        "\t"+str(round(microrecall,3))+
        "\t"+str(round(microf,3))
        )
    # Macro average
    macrorecall = sum(recall.values())/len(recall)
    macroprecision = sum(precision.values())/len(precision)
    macroF = sum(f.values())/len(f)
    print("MacAvg"+
        "\t"+str( )+
        "\t"+str( )+
        "\t"+str( )+
        "\t"+str(round(macroprecision,3))+
        "\t"+str(round(macrorecall,3))+
        "\t"+str(round(macroF,3))
        )
    print("Official result:",macroF)
        
if __name__ == '__main__':
    welcome()
    checkParameters()
    goldFile = sys.argv[1]
    predictedFile = sys.argv[2]
    goldList = readFileToList(goldFile)
    predictedList = readFileToList(predictedFile)
    if (len(goldList) != len(predictedList)):
        eprint("Number of labels is not aligned!")
        sys.exit(1)
    calculatePRF(goldList,predictedList)
