def local_DPP(line):
    avg=np.mean(line,axis=1)
    var=np.var(line,axis=1)
    var=np.sqrt(var)
    f=np.empty(line.shape)
    for i in range(len(line)):
        for j in range (len(line[0])):
            if var[i]==0:
                f[i][j]=0
            else:
                f[i][j]=(line[i][j]-avg[i])*1.0/var[i]
    p=[]
    n=3
    p.append(f[:(len(f)/3)])
    p.append(f[(len(f)/3):(2*len(f)/3)])
    p.append(f[(2*len(f)/3):])
    feature=[]
    lamudamax=2
    avgcol=np.mean(line,axis=0)
    for k in range(len(p)):
        for j in range(20):
            sum=0
            for i in range(len(p[k])):
                    sum=sum+p[k][i][j]
            if(len(p[k]) !=0):
                sum=sum*1.0/(len(p[k]))
            feature.append(sum)
        for j in range(20):
            sum=0
            lamuda=1
            while lamuda<lamudamax:
                for i in range(len(p[k])-lamuda):
                        sum=pow((p[k][i][j])-(p[k][i+lamuda][j]),2)+sum
                if((len(p[k])-lamuda)!=0):
                    sum=sum*1.0/(len(p[k])-lamuda)  
                feature.append(sum)
                lamuda=lamuda+1
    print len(feature)
    return feature


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