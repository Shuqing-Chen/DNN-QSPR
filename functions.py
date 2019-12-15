import numpy as np
import math
import matplotlib.pyplot as plt

def data_select(x,y,sub=100):
    a=(max(x)-min(x))/sub
    x2=[min(x)+a*i for i in range(sub+1)]
    y2=[0 for i in range(sub+1)]
    c2=[0 for i in range(sub+1)]
    for i in range(len(x)):
        b=int((x[i]-min(x))//a)
        y2[b]+=y[i]
        c2[b]+=1
    for i in range(len(y2)):
        if y2[i]>0:
            y2[i]=y2[i]/c2[i]
    return x2,y2
def replicate(l,n):
    group=int(len(l)/n)
    l2=[]
    for i in range(group):
        tem=0
        for j in range(n):
            tem+=l[i*n+j]
        l2.append(tem/n)
    return l2
def data_select2(x,y):
    x = x.tolist()
    y = y.tolist()
    for i in range(len(x)):
        x[i]=float('%.5f'%(x[i]))
    x_copy=x.copy()
    x2=list(set(x_copy))
    x2.sort()
    y2=[0 for i in x2]
    c=[0 for i in x2]
    for i in range(len(x)):
        index=x2.index(x[i])
        y2[index]+=y[i]
        c[index]+=1
    for i in range(len(y2)):
        y2[i]=y2[i]/c[i]
    rate=len(x2)/len(x)
#     if rate>0.5:
#         x2=replicate(x2,7)
#         y2=replicate(y2,7)
    return x2,y2
def model0(x,para,best=0):
    if isinstance(x,np.ndarray) or isinstance(x,list):
        result=[]
        for i in x:
            i-=best
            if i<=0:
                result.append(math.log(1+para[0]*math.exp(para[1]*i)))
            else:
                result.append(math.log(1+para[0]*math.exp(-para[2]*i)))
        return result
    else:
        x-=best
        if x<=0:
            return math.log(1+para[0]*math.exp(para[1]*x))
        else:
            return math.log(1+para[0]*math.exp(-para[2]*x))
def loss_coff(const,x,y,model):
    pre_y=[]
    for i in x:
        pre_y.append(model(i,const))
    pre_y=np.array(pre_y)
    error=np.absolute(pre_y-y)
    return np.sum(error)/len(y)
def draw_plot(x,y,style=['.g','.r','.b']):
    plt.figure(dpi=300, figsize=(10, 6))
    fig=plt.gcf()
    ax=plt.gca()
    # fig.set_facecolor('None')
    # ax.set_facecolor('None')
    for k in range(len(x)):
        plt.plot(x[k], y[k], style[k], alpha=0.3)
    plt.show()
def di(a,b):
    return (a-b)/a
def di2(a,b):
    return float('%.5f'%(a-b))
def di3(a,b):
    return (a-b)/(a+b)
def w2a(m1,m2,w):
    if w!=0:
        w=w/100
        a=1/(1+m2*(1-w)/(m1*w))
    else:
        a=0
    return a*100
def a2w(m1,m2,a):
    if a!=0:
        a=a/100
        w=1/(1+m1*(1-a)/(m2*a))
    else:
        w=0
    return w*100
