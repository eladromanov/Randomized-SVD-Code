# -*- coding: utf-8 -*-


import numpy
import numpy as np
from numpy import sqrt
from scipy import integrate

def SanitizeGammaBeta(gamma,beta):
    
    assert(gamma > 0 and beta > 0)
    if beta > 1:
        beta = 1
    if gamma*beta > 1:
        beta = 1/gamma
    return gamma, beta
    

def KFunc(gamma, beta, y):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    assert( gamma*beta < 1)
    assert( y >= sqrt(NoiseEdge(gamma, beta)) )
    
    return numpy.array( [numpy.array( [1/2 * ( y )**( -1 ) * ( 1 + ( ( y )**( 2 \
) * ( gamma )**( 1/2 ) + ( -1 * gamma + -1 * ( ( ( ( -1 + ( ( y )**( \
2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * \
gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( 1/2 ) ) ) ),( y \
)**( -1 ) * ( 1 + -1 * beta * gamma ),0,0,0,0,] ),numpy.array( [( y \
)**( -1 ) * ( 1 + -1 * beta * gamma ),( y )**( -1 ) * ( 1 + -1 * beta \
* gamma ),0,0,0,( -1 + beta * gamma ),] ),numpy.array( [0,0,1/2 * ( y \
)**( -3 ) * ( gamma )**( -1 ) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) \
* ( ( y )**( 2 ) * ( gamma )**( 1/2 ) * ( 1 + gamma ) + ( -1 * ( ( 1 \
+ ( gamma + -2 * beta * gamma ) ) )**( 2 ) + -1 * ( 1 + ( gamma + -2 \
* beta * gamma ) ) * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) \
+ -1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * \
beta * gamma ) ) ) )**( 1/2 ) ) ),1/2 * ( y )**( -2 ) * ( gamma )**( \
-1 ) * ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + ( -1 * gamma + ( \
2 * beta * gamma + -1 * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( \
1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + \
-1 * beta * gamma ) ) ) )**( 1/2 ) ) ) ) ),( -1 * beta * ( gamma )**( \
1/2 ) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) + 1/2 * ( y )**( -2 ) * \
( 1 + -1 * beta ) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) * ( 1 + ( \
-1 * ( y )**( 2 ) * ( gamma )**( 1/2 ) + ( gamma + ( -2 * beta * \
gamma + ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma \
) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * beta * gamma \
) ) ) )**( 1/2 ) ) ) ) ) ),0,] ),numpy.array( [0,0,1/2 * ( y )**( -2 \
) * ( gamma )**( -1 ) * ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + \
( -1 * gamma + ( 2 * beta * gamma + -1 * ( ( ( ( -1 + ( ( y )**( 2 ) \
* ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * \
( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( 1/2 ) ) ) ) ),1/2 * ( y \
)**( -1 ) * ( gamma )**( -1 ) * ( -1 + ( ( y )**( 2 ) * ( gamma )**( \
1/2 ) + ( gamma + -1 * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 \
) + -1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 \
* beta * gamma ) ) ) )**( 1/2 ) ) ) ),( y )**( -1 ) * ( 1 + -1 * beta \
),0,] ),numpy.array( [0,0,( -1 * beta * ( gamma )**( 1/2 ) * ( ( 1 + \
-1 * beta * gamma ) )**( -1 ) + 1/2 * ( y )**( -2 ) * ( 1 + -1 * beta \
) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) * ( 1 + ( -1 * ( y )**( 2 ) \
* ( gamma )**( 1/2 ) + ( gamma + ( -2 * beta * gamma + ( ( ( ( -1 + ( \
( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * \
beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( 1/2 ) ) ) \
) ) ),( y )**( -1 ) * ( 1 + -1 * beta ),1/2 * ( y )**( -1 ) * ( 1 + \
-1 * beta ) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) * ( 1 + ( ( y \
)**( 2 ) * ( gamma )**( 1/2 ) + ( -1 * gamma + -1 * ( ( ( ( -1 + ( ( \
y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta \
* gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( 1/2 ) ) ) \
),0,] ),numpy.array( [0,( -1 + beta * gamma ),0,0,0,1/2 * ( y )**( -1 \
) * ( 1 + -1 * beta ) * ( gamma )**( 1/2 ) * ( 1 + -1 * beta * gamma \
) * ( ( 1 + ( gamma + -1 * beta * gamma ) ) )**( -1 ) * ( 1 + ( ( y \
)**( 2 ) * ( gamma )**( 1/2 ) + ( gamma + ( -2 * beta * gamma + -1 * \
( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) \
)**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) \
) )**( 1/2 ) ) ) ) ),] ),] )
                                           
def TFunc(gamma,beta,y):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    assert( gamma*beta < 1)
    assert( y >= sqrt(NoiseEdge(gamma, beta)) )
    
    return numpy.array( [numpy.array( [-1/2 * y * ( 1/2 * ( y )**( -2 ) * ( 2 * \
y * ( gamma )**( 1/2 ) + -2 * y * ( -1 + ( ( y )**( 2 ) * ( gamma \
)**( 1/2 ) + -1 * gamma ) ) * ( gamma )**( 1/2 ) * ( ( ( ( -1 + ( ( y \
)**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * \
gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( -1/2 ) ) + -1 * \
( y )**( -3 ) * ( 1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + ( -1 * \
gamma + -1 * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * \
gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * beta * \
gamma ) ) ) )**( 1/2 ) ) ) ) ),( y )**( -2 ) * ( 1 + -1 * beta * \
gamma ),0,0,0,0,] ),numpy.array( [( y )**( -2 ) * ( 1 + -1 * beta * \
gamma ),( y )**( -2 ) * ( 1 + -1 * beta * gamma ),0,0,0,0,] \
),numpy.array( [0,0,-1/2 * y * ( 1/2 * ( y )**( -4 ) * ( gamma )**( \
-1 ) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) * ( 2 * y * ( gamma )**( \
1/2 ) * ( 1 + gamma ) + -2 * y * ( -1 + ( ( y )**( 2 ) * ( gamma )**( \
1/2 ) + -1 * gamma ) ) * ( gamma )**( 1/2 ) * ( 1 + ( gamma + -2 * \
beta * gamma ) ) * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + \
-1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * \
beta * gamma ) ) ) )**( -1/2 ) ) + -2 * ( y )**( -5 ) * ( gamma )**( \
-1 ) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) * ( ( y )**( 2 ) * ( \
gamma )**( 1/2 ) * ( 1 + gamma ) + ( -1 * ( ( 1 + ( gamma + -2 * beta \
* gamma ) ) )**( 2 ) + -1 * ( 1 + ( gamma + -2 * beta * gamma ) ) * ( \
( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( \
2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) \
)**( 1/2 ) ) ) ),1/2 * ( -1/2 * ( y )**( -2 ) * ( gamma )**( -1 ) * ( \
2 * y * ( gamma )**( 1/2 ) + -2 * y * ( -1 + ( ( y )**( 2 ) * ( gamma \
)**( 1/2 ) + -1 * gamma ) ) * ( gamma )**( 1/2 ) * ( ( ( ( -1 + ( ( y \
)**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * \
gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( -1/2 ) ) + ( y \
)**( -3 ) * ( gamma )**( -1 ) * ( -1 + ( ( y )**( 2 ) * ( gamma )**( \
1/2 ) + ( -1 * gamma + ( 2 * beta * gamma + -1 * ( ( ( ( -1 + ( ( y \
)**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * \
gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( 1/2 ) ) ) ) ) \
),1/2 * ( -1/2 * ( y )**( -2 ) * ( 1 + -1 * beta ) * ( ( 1 + -1 * \
beta * gamma ) )**( -1 ) * ( -2 * y * ( gamma )**( 1/2 ) + 2 * y * ( \
-1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) * ( gamma \
)**( 1/2 ) * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * \
gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * beta * \
gamma ) ) ) )**( -1/2 ) ) + ( y )**( -3 ) * ( 1 + -1 * beta ) * ( ( 1 \
+ -1 * beta * gamma ) )**( -1 ) * ( 1 + ( -1 * ( y )**( 2 ) * ( gamma \
)**( 1/2 ) + ( gamma + ( -2 * beta * gamma + ( ( ( ( -1 + ( ( y )**( \
2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * \
gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( 1/2 ) ) ) ) ) \
),0,] ),numpy.array( [0,0,1/2 * ( -1/2 * ( y )**( -2 ) * ( gamma )**( \
-1 ) * ( 2 * y * ( gamma )**( 1/2 ) + -2 * y * ( -1 + ( ( y )**( 2 ) \
* ( gamma )**( 1/2 ) + -1 * gamma ) ) * ( gamma )**( 1/2 ) * ( ( ( ( \
-1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + \
-4 * beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( -1/2 \
) ) + ( y )**( -3 ) * ( gamma )**( -1 ) * ( -1 + ( ( y )**( 2 ) * ( \
gamma )**( 1/2 ) + ( -1 * gamma + ( 2 * beta * gamma + -1 * ( ( ( ( \
-1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + \
-4 * beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( 1/2 \
) ) ) ) ) ),-1/4 * ( y )**( -1 ) * ( gamma )**( -1 ) * ( 2 * y * ( \
gamma )**( 1/2 ) + -2 * y * ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 \
) + -1 * gamma ) ) * ( gamma )**( 1/2 ) * ( ( ( ( -1 + ( ( y )**( 2 ) \
* ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * \
( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( -1/2 ) ),0,0,] \
),numpy.array( [0,0,1/2 * ( -1/2 * ( y )**( -2 ) * ( 1 + -1 * beta ) \
* ( ( 1 + -1 * beta * gamma ) )**( -1 ) * ( -2 * y * ( gamma )**( 1/2 \
) + 2 * y * ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) \
) * ( gamma )**( 1/2 ) * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( \
1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + \
-1 * beta * gamma ) ) ) )**( -1/2 ) ) + ( y )**( -3 ) * ( 1 + -1 * \
beta ) * ( ( 1 + -1 * beta * gamma ) )**( -1 ) * ( 1 + ( -1 * ( y \
)**( 2 ) * ( gamma )**( 1/2 ) + ( gamma + ( -2 * beta * gamma + ( ( ( \
( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) \
+ -4 * beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( \
1/2 ) ) ) ) ) ),0,-1/4 * ( y )**( -1 ) * ( 1 + -1 * beta ) * ( ( 1 + \
-1 * beta * gamma ) )**( -1 ) * ( 2 * y * ( gamma )**( 1/2 ) + -2 * y \
* ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) * ( \
gamma )**( 1/2 ) * ( ( ( ( -1 + ( ( y )**( 2 ) * ( gamma )**( 1/2 ) + \
-1 * gamma ) ) )**( 2 ) + -4 * beta * gamma * ( 1 + ( gamma + -1 * \
beta * gamma ) ) ) )**( -1/2 ) ),0,] ),numpy.array( [0,0,0,0,0,-1/4 * \
( y )**( -1 ) * ( 1 + -1 * beta ) * ( gamma )**( 1/2 ) * ( 1 + -1 * \
beta * gamma ) * ( ( 1 + ( gamma + -1 * beta * gamma ) ) )**( -1 ) * \
( 2 * y * ( gamma )**( 1/2 ) + -2 * y * ( -1 + ( ( y )**( 2 ) * ( \
gamma )**( 1/2 ) + -1 * gamma ) ) * ( gamma )**( 1/2 ) * ( ( ( ( -1 + \
( ( y )**( 2 ) * ( gamma )**( 1/2 ) + -1 * gamma ) ) )**( 2 ) + -4 * \
beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) ) )**( -1/2 ) \
),] ),] )
                                                                
                                                                
# def BBP(gamma,beta):
    
#     gamma, beta = SanitizeGammaBeta(gamma, beta)
    
#     return ( 2 )**( -1/2 ) * ( ( gamma )**( -1/2 ) * ( -1 + ( -1 * gamma + ( \
# beta * gamma + ( ( beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) \
# ) )**( 1/2 ) + ( ( ( 1 + gamma ) * ( 1 + ( gamma + -1 * beta * gamma \
# ) ) + ( beta * gamma * ( 1 + ( gamma + -1 * beta * gamma ) ) )**( 1/2 \
# ) * ( 4 * ( beta )**( -1 ) + -2 * ( 1 + ( gamma + -1 * beta * gamma ) \
# ) ) ) )**( 1/2 ) ) ) ) ) )**( 1/2 )
                                   
                                   
def BBP(gamma,beta):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    rho = 1 + gamma - beta*gamma
    v = (gamma**(1/2)+gamma**(-1/2)-beta*gamma**(1/2)-sqrt(beta*rho))/2
    return sqrt( sqrt( v**2 + (rho/gamma)/sqrt(beta) ) - v )
                                   
def SpikeFwd(gamma,beta,sigma):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    if sigma <= BBP(gamma,beta):
        return sqrt(NoiseEdge(gamma,beta))
    else:
        return ( ( gamma )**( -1/2 ) * ( sigma )**( -2 ) * ( ( gamma )**( 1/2 ) + ( \
    sigma )**( 2 ) ) * ( 1 + ( gamma )**( 1/2 ) * ( sigma )**( 2 ) ) * ( \
    ( 1 + ( gamma + ( -1 * beta * gamma + ( gamma )**( 1/2 ) * ( sigma \
    )**( 2 ) ) ) ) )**( -1 ) * ( ( 1 + beta * ( gamma )**( 1/2 ) * ( \
    sigma )**( 2 ) ) )**( -1 ) * ( 1 + ( gamma + ( -1 * beta * gamma + ( \
    2 * beta * ( gamma )**( 1/2 ) * ( 1 + ( gamma + -1 * beta * gamma ) ) \
    * ( sigma )**( 2 ) + beta * gamma * ( sigma )**( 4 ) ) ) ) ) )**( 1/2 \
    )
                                                                 
def NoiseEdge(gamma,beta):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    return ( ( gamma )**( -1/2 ) + ( ( gamma )**( 1/2 ) + 2 * ( beta * ( 1 + ( \
gamma + -1 * beta * gamma ) ) )**( 1/2 ) ) )
   
def NoiseEdgeLower(gamma,beta):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    return ( ( gamma )**( -1/2 ) + ( ( gamma )**( 1/2 ) - 2 * ( beta * ( 1 + ( \
gamma + -1 * beta * gamma ) ) )**( 1/2 ) ) )
        
def SpikeInv(gamma,beta,y):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    assert(y >= np.sqrt(NoiseEdge(gamma,beta)))
    sigmaLow = BBP(gamma,beta)
    sigmaHigh = 2*sigmaLow
    prec=1e-9
    while SpikeFwd(gamma,beta,sigmaHigh) < y:
        sigmaLow = sigmaHigh
        sigmaHigh = 2*sigmaHigh
    while sigmaHigh-sigmaLow > prec:  # Search up to machine precision   
        sigmaMid = (sigmaLow+sigmaHigh)/2
        value = SpikeFwd(gamma,beta,sigmaMid)
        if value < y:
            sigmaLow = sigmaMid
        else:
            sigmaHigh = sigmaMid
    return (sigmaLow+sigmaHigh)/2
        
def LimitingAngles(gamma,beta,y):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    if y**2 <= NoiseEdge(gamma,beta):
        return [0, 0]
            
    sigma = SpikeInv(gamma,beta,y)
    
    # When gamma*beta = 1, use the known closed-form expression
    if gamma*beta == 1:
        
        VAngle = sqrt( (sigma**4-1) / (sigma**4 + sqrt(gamma)*sigma**2) )
        UAngle = sqrt( (sigma**4-1) / (sigma**4 + sigma**2/sqrt(gamma)) )
        
        return UAngle, VAngle
    
    # When gamma*beta < 1, use our new results. To compute the angles, need to solve
    # system of equations in 6 variables
        
    H = numpy.array( [numpy.array( [0,0,0,1,0,0,] ),numpy.array( \
[0,0,0,0,-1,0,] ),numpy.array( [0,0,0,0,0,1,] ),numpy.array( \
[1,0,0,0,0,0,] ),numpy.array( [0,-1,0,0,0,0,] ),numpy.array( \
[0,0,1,0,0,0,] ),] )
                                                            
    eig_vals, eig_vecs = np.linalg.eigh(KFunc(gamma, beta, y) - (1/sigma)*H)
    indx = np.argmin(np.abs(eig_vals))
    d_pre = eig_vecs[:,indx]
    normalization = np.sqrt( d_pre.T @ TFunc(gamma,beta,y) @ d_pre )
    d = d_pre/normalization
    
    return np.abs([d[3]/sigma, d[0]/sigma])

def MarchenkoPasturDensity(gamma,beta,x):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    UpperEdge = gamma**(1/2) + gamma**(-1/2) + 2*sqrt(beta*(1+gamma-gamma*beta))
    LowerEdge = gamma**(1/2) + gamma**(-1/2) - 2*sqrt(beta*(1+gamma-gamma*beta))
    Delta = (UpperEdge-x)*(x-LowerEdge)
    return sqrt(Delta) /  (2*np.pi*beta*sqrt(gamma) * x)

    
def OptimalShrinker(gamma,beta,y):
     
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    if y**2 <= NoiseEdge(gamma,beta):
        return 0
    
    if (gamma*beta == 1):
        # The case gamma*beta=1 can be obtained as the limit of our shrinker. However, the the numerics are not applicable in this case (0/0 etc).
        # Instead, we use the well-known closed-form formula:
        return sqrt( (y**2 - NoiseEdge(gamma,1)) * (y**2 - NoiseEdgeLower(gamma,1))  ) / y
    
    spikeInv = SpikeInv(gamma,beta,y)
    AngleU, AngleV = LimitingAngles(gamma, beta, y)
    return spikeInv * AngleU * AngleV


# This is a naive generalization of the beta=1 optimal shrinker
def ConjecturedShrinker(gamma,beta,y):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    if y**2 <= NoiseEdge(gamma,beta):
        return 0
    
    return sqrt( (y**2 - NoiseEdge(gamma,beta)) * (y**2 - NoiseEdgeLower(gamma,beta))  ) / y

def ShrinkerMSE(gamma,beta,sigma,w):
    
    y = SpikeFwd(gamma,beta,sigma)
    UAngle, VAngle = LimitingAngles(gamma, beta, y)
    
    return sigma**2 + w**2 - 2*sigma*w*UAngle*VAngle

def MPMedian(gamma,beta):
    
    gamma, beta = SanitizeGammaBeta(gamma, beta)
    
    UpperEdge = gamma**(1/2) + gamma**(-1/2) + 2*sqrt(beta*(1+gamma-gamma*beta))
    LowerEdge = gamma**(1/2) + gamma**(-1/2) - 2*sqrt(beta*(1+gamma-gamma*beta))
    
    high=UpperEdge
    low=LowerEdge
    prec=1e-8
    while high-low>prec:
       mid = (high+low)/2
       val, _ = integrate.quad(lambda x: MarchenkoPasturDensity(gamma, beta, x), LowerEdge, mid)
       if val < 1/2:
           low=mid
       else:
           high=mid
           
    mid = (high+low)/2
    return mid
           
def UStar(gamma,A):
    
    eps=1e-4
    
    def decrease_beta(beta):
        beta=beta/10
        sigma=A*beta**(-1/2)
        while sigma <= BBP(gamma,beta):
            beta=beta/10
            sigma=A*beta**(-1/2)
        y=SpikeFwd(gamma,beta,sigma)
        return beta,sigma,y
    
    beta, sigma, y = decrease_beta(1/1000)   
    UAngle, _ = LimitingAngles(gamma, beta, y)
    
    beta, sigma, y = decrease_beta(beta/10)
    UAngle_New, _ = LimitingAngles(gamma, beta, y)
    
    while (np.abs(UAngle_New-UAngle)) > eps:
        
        UAngle=UAngle_New
        beta,sigma,y = decrease_beta(beta/10)
        UAngle_New, _ = LimitingAngles(gamma, beta, y)
        
    return UAngle_New
    
def VStar(gamma,A):
    
    eps=1e-4
    
    def decrease_beta(beta):
        beta=beta/10
        sigma=A*beta**(-1/4)
        while sigma <= BBP(gamma,beta):
            beta=beta/10
            sigma=A*beta**(-1/4)
        y=SpikeFwd(gamma,beta,sigma)
        return beta,sigma,y
    
    beta, sigma, y = decrease_beta(1/1000)   
    _, VAngle = LimitingAngles(gamma, beta, y)
    
    beta, sigma, y = decrease_beta(beta/10)
    _, VAngle_New = LimitingAngles(gamma, beta, y)
    
    while (np.abs(VAngle_New-VAngle)) > eps:
        
        VAngle=VAngle_New
        beta,sigma,y = decrease_beta(beta/10)
        _, VAngle_New = LimitingAngles(gamma, beta, y)
        
    return VAngle_New