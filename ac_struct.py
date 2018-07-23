import numpy as np
def struct(strpath,lmin):
    f=open(strpath,'r')
    line=f.read()
    line=line.split(")")
    line1=line[1].split("\n")
    line2=line1[2:-1]
    col1=[]
    col2=[]
    col3=[]
    for i in range(len(line2)):
        temp=line2[i].split(" ")
#         print temp
        col1.append(temp[-5])
        col2.append(temp[-3])
        col3.append(temp[-1])
    col1=np.array(col1)
    col2=np.array(col2)
    col3=np.array(col3)
    col1 = map(eval,col1) 
    avgcol1=np.mean(col1)
    col2 = map(eval,col2) 
    avgcol2=np.mean(col2)
    col3 = map(eval,col3) 
    avgcol3=np.mean(col3)
    lamuda=1
    feature=[]
    while lamuda<lmin:
        sum=0
        for i in range(len(col1)-lamuda):
            sum=(col1[i]-avgcol1)*(col1[i+lamuda]-avgcol1)+sum
        sum=sum/(len(col1)-lamuda)
        feature.append(sum)
        lamuda=lamuda+1
    feature.append(avgcol1)
    feature.append(np.std(col1))
    lamuda=1
    while lamuda<lmin:
        sum=0
        for i in range(len(col2)-lamuda):
            sum=(col2[i]-avgcol2)*(col2[i+lamuda]-avgcol2)+sum
        sum=sum/(len(col2)-lamuda)
        feature.append(sum)
        lamuda=lamuda+1
    feature.append(avgcol2)
    feature.append(np.std(col2))
    lamuda=1
    while lamuda<lmin:
        sum=0
        for i in range(len(col3)-lamuda):
            sum=(col3[i]-avgcol3)*(col3[i+lamuda]-avgcol3)+sum
        sum=sum/(len(col3)-lamuda)
        feature.append(sum)
        lamuda=lamuda+1
    feature.append(avgcol3)
    feature.append(np.std(col3))
    print len(feature)
    return feature
def transtofeature(length,stratpositive,endpositive,countpositive,startnege,endnege,countnege):
    featurepositive=[]
	#length=50
    #i=1075
    i=startpositive
    #while i<1168:
    while i<endpositive+1:
        print i
        strpath="F:/DNAbindingprotein/structal/"+str(i)+".ss2"
        featurepositive.append(struct(strpath,length))
        i=i+1
    featurenegative=[]
    #i=1168
    i=startnege
    #while i<1261:
    while i < endnege+1:
        print i
        strpath="F:/DNAbindingprotein/structal/"+str(i)+".ss2"
        featurenegative.append(struct(strpath,length))
        i=i+1
    labelnegtive=np.zeros(countnege,dtype='int')
    labelnpositive=np.ones(countpositive,dtype='int')
    feature=[]
    feature.extend(featurenegative)
    feature.extend(featurepositive)
    np.savetxt('./featuredata/g_feature_gai1221_structual_'+str(countnege+countpositive)+'.csv', feature, delimiter=',')
    print len(feature)
    label=[]
    label.extend(labelnegtive)
    label.extend(labelnpositive)
    np.savetxt('./featuredata/g_label_gai1221_structual__'+str(countnege+countpositive)+'.csv', label, delimiter=',')
#transtofeature()
import sys
print "jiaobenmin", sys.argv[0]
length=int(sys.argv[1])
startpositive=int(sys.argv[2])
endpositive=int(sys.argv[3])
countpositive=int(sys.argv[4])
startnege=int(sys.argv[5])
endnege=int(sys.argv[6])
countnege=int(sys.argv[7])
transtofeature(length,startpositive,endpositive,countpositive,startnege,endnege,countnege)
