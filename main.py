# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 14:49:57 2019

@author: Loo Ting
"""
import numpy as np
import math
from astroquery.jplhorizons import Horizons
import matplotlib.pyplot as plt

def mag_vec(x):
    # define magnitude of vec
    mag = np.linalg.norm(x)
    return mag.item()

def tof_min(s,c,x,grav_sun):
    # find tof_min
    alf_min = math.pi
    if x == 'large':
        bet_min = -2*math.asin(math.sqrt((s-c)/s))
        
    else:
        bet_min = 2*math.asin(math.sqrt((s-c)/s))

    a_min = s/2
    
    n_min = math.sqrt(grav_sun/(a_min**3))

    tof_min = (alf_min - bet_min -(math.sin(alf_min)-math.sin(bet_min)))/n_min
    return tof_min

def tof_g(s,c,d,x,tof,grav_sun):
    
    alf_0 = 2*math.asin(math.sqrt(s/(2*d)))
    bet_0 = 2*math.asin(math.sqrt((s-c)/(2*d)))
    n = math.sqrt(grav_sun/(d**3))
    
    if x == 'large':
        if tof > tof_min(s,c,x,grav_sun):
            alf = 2*math.pi - alf_0
            bet = -bet_0
        else:
            alf = alf_0
            bet = -bet_0
    else:
        if tof > tof_min(s,c,x,grav_sun):
            alf = 2*math.pi - alf_0
            bet = bet_0
        else:
            alf = alf_0
            bet = bet_0
    tof_g= (alf - bet - (math.sin(alf) - math.sin(bet)))*(1/n)  # calculate residual
    
    return tof_g, alf, bet


def lamb_sol(r1v,v1iv,r2v,v2fv,t1,t2,x,a2):
    grav_sun =  1.32712e11 #3.986004415e5
    
    tof = (t2-t1)*24*60*60 #seconds
    tol = 0.000001
    if tof<=0:
        return None
        
    if x =='large': #larger transfer angle
        d_ta= 2 *math.pi - math.acos(np.dot(r1v,r2v)/(mag_vec(r1v)*mag_vec(r2v)))
        c = (math.sqrt(mag_vec(r1v)**2 +mag_vec(r2v)**2 - 2* mag_vec(r1v)*mag_vec(r2v)*math.cos(d_ta)))
        s = 0.5 * (mag_vec(r1v) + mag_vec(r2v) + c) #convert to float
       
        a1 = s/2
        d= (a1+a2)/2
        while abs(a2-a1) >=tol or abs(tof_g(s,c,d,x,tof,grav_sun)[0]-tof) >=tol:
        #while abs(a2-a1) >=tol:
            d= (a1+a2)/2
            
            if tof_g(s,c,d,x,tof,grav_sun)[0]-tof == 0.0:    
                return d
            if (tof_g(s,c,d,x,tof,grav_sun)[0]-tof)*(tof_g(s,c,a1,x,tof,grav_sun)[0]-tof)> 0:
                a1 = d
            else:
                a2 = d


    else: #smaller transfer angle
        d_ta = math.acos((np.dot(r1v,r2v))/(mag_vec(r1v)*mag_vec(r2v)))
        c = math.sqrt(mag_vec(r1v)**2 +mag_vec(r2v)**2 - 2* mag_vec(r1v)*mag_vec(r2v)*math.cos(d_ta))
        s = 0.5* (mag_vec(r1v)+mag_vec(r2v) + c)
        
        a1 = s/2
        d= (a1+a2)/2
        while abs(a2-a1) >=tol or abs(tof_g(s,c,d,x,tof,grav_sun)[0]-tof) >=tol:
        #while abs(a2-a1) >=tol:
            d= (a1+a2)/2
            if tof_g(s,c,d,x,tof,grav_sun)[0]-tof == 0.0:    
                return d
            if (tof_g(s,c,d,x,tof,grav_sun)[0]-tof)*(tof_g(s,c,a1,x,tof,grav_sun)[0]-tof)> 0:
                a1 = d
            else:
                a2 = d
        
    #calculate semi-perimeter
    p=4*d*(s-mag_vec(r1v))*(s-mag_vec(r2v))*(math.sin((tof_g(s,c,d,x,tof,grav_sun)[1]+tof_g(s,c,d,x,tof,grav_sun)[2])/2))**2/(c**2)
    e=math.sqrt(1-p/d)
    #print(p,e)
    
    #find true anomaly
    ta1=math.acos((p/mag_vec(r1v)-1)/e)
    ta2=math.acos((p/mag_vec(r2v)-1)/e)
    if abs(d_ta-(ta2-ta1))<0.000001:
        ta2=ta2
        ta1=ta1
    elif abs(d_ta-(-ta2-ta1))<0.000001:
        ta2 =-ta2
        ta1 =ta1
    elif abs(d_ta-(-ta2+ta1))<0.000001:
        ta2=-ta2
        ta1=-ta1
    else:
        ta2=ta2
        ta1=-ta1
    print("true anomaly 1 and 2:", ta1,",",ta2)
    # use fg functions and return vd, va
    f1=1-mag_vec(r2v)*(1-math.cos(d_ta))/p
    g1=mag_vec(r1v)*mag_vec(r2v)*math.sin(d_ta)/math.sqrt(grav_sun*p)
    df1=math.sqrt(grav_sun/p)*(math.tan(d_ta/2))*((1-math.cos(d_ta))/p-1/mag_vec(r1v)-1/mag_vec(r2v))
    dg1=1-mag_vec(r1v)*(1-math.cos(d_ta))/p
    print("Lagrange coefficients (f,g,f'g'): ",f1,g1,df1,dg1)
    print("Planet velocity 1 and 2:",v1iv,v2fv)
    #find departure and arrival velocity within the transfer arc
    v_d=(r2v-f1*r1v)/g1
    v_a=df1*r1v+dg1*v_d
    print("departure and arrival velocity of transfer:",v_d,v_a)
    v_inf_d=v_d-v1iv
    v_inf_a=v_a-v2fv
    print("vinf departure and arrival:",v_inf_d,v_inf_a)
    print(mag_vec(v_inf_d),mag_vec(v_inf_a))
    c3d=mag_vec(v_inf_d)**2
    c3a=mag_vec(v_inf_a)**2
    print("c3d and c3a:",c3d,c3a)

    return c3d



au2km=149597870.7
d2s =24*60*60 #days to seconds

# inputs in sun-centered inertial coordinate
t1 = Time("2030-01-20 00:00", scale="utc").jd
t2 = Time("2032-07-01 00:00", scale="utc").jd

r1v=np.array([-72576391.16328001, 128061475.5880728, -8055.475574925542])
v1iv=np.array([-2.639048456109558E+01,-1.479176337239575E+01,2.010693239588690E-03])
r2v=np.array([3.278496985125721E+08,-6.961872482800779E+08 ,-4.440828489689320E+06])
v2fv=np.array([1.167433018601784E+01, 6.182818012521681E+00,-2.868375221595132E-01])
       
lamb_sol(r1v,v1iv,r2v,v2fv,t1,t2,'small',1e9)
