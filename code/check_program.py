import math
def torad(d):
    return d*math.pi/180.0
def angle(x1,y1,x2,y2):
    x1=torad(x1)
    y1=torad(y1)
    x2=torad(x2)
    y2=torad(y2)
    if x1>x2:
        return math.atan((x2-x1)/(y2-y1))
    else:
        return math.pi+math.atan((x2-x1)/(y2-y1))

def transfer(a1,b1,a2,b2):
    a1=torad(a1)
    b1=torad(b1)
    a2=torad(a2)
    b2=torad(b2)
    s=2.0*math.asin(math.sqrt((math.sin((b2-b1)/2.0)**2)+math.cos(b1)*math.cos(b2)*(math.sin((a2-a1)/2.0)**2)))*6378.137
    return s

new_x=[]
new_y=[]
for i in range(len(latitude)):
    dis=transfer(magtitude[i],latitude[i],114.0,23.0)
    ang=angle(magtitude[i],latitude[i],114.0,23.0)
    x=dis*math.cos(ang)
    y=dis*math.sin(ang)
    new_x.append(x)
    new_y.append(y)