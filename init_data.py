with open('data_all_180814.json', 'r',encoding="UTF-8") as f:
    data=json.load(f)
x=[]
y=[]
x0=[]
y0=[]
for i in data['ss']:
    y.append(i[2]/100) #
    tem=[]
    for k in range(10):
        tem.append(di2(data['base'][i[0]][k],data['base'][i[1]][k]))
    x.append(tem)
    # if y[-1]>0.01 and y[-1]<1:
    #     y0.append(y[-1])
    #     x0.append(tem)
attrbutes=data['desc']
