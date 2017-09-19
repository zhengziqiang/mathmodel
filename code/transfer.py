import math
import numpy as np
def torad(d):
    return d*math.pi/180.0

def transfer(a1,b1,a2,b2):
    a1=torad(a1)
    b1=torad(b1)
    a2=torad(a2)
    b2=torad(b2)
    s=2.0*math.asin(math.sqrt((math.sin((b2-b1)/2.0)**2)+math.cos(b1)*math.cos(b2)*(math.sin((a2-a1)/2.0)**2)))*6378.137
    return s

def angle(x1,y1,x2,y2):
    x1=torad(x1)
    y1=torad(y1)
    x2=torad(x2)
    y2=torad(y2)
    d=math.sin(y1)*math.sin(y2)+math.cos(y1)*math.cos(y2)*math.cos(x2-x1)
    d=np.sqrt(1.0-d*d)
    d=math.cos(y2)*math.sin(x2-x1)/d
    d=math.asin(d)*180/math.pi
    return -d