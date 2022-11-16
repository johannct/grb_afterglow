# cython: infer_types=True
#cython: language_level=3
from math import sin,cos,sqrt,exp
import numpy as np
from scipy import integrate as sciint
from scipy import optimize as sciopt
from scipy.special import kn
from scipy.special import gamma as gfunc
from math import pi, sqrt, exp, cos, sin

cimport cython
cimport numpy as np
np.import_array()

cdef double mev2erg = 1.0218e-6
cdef double mpc2 = 938.72*mev2erg # erg
cdef double c=2.9792458e10#cm s-1
#cdef double u = 1e50 *1e-24*1.e3/mpc2/c**3
cdef double u = 1/mpc2/c**3
cdef double bm = 17/8/pi
cdef double st = (2*1.15/5)**2
cdef mpratio = 1836.15267
#cdef qe = 1.60217663e-19 #C
cdef qe = 4.8032e-10 # esu ou statC
cdef mec2 = mpc2/mpratio
cdef mec = mec2/c
cdef sigmaT = 6.6524e-25#cm2 Thomson cross section

# time before which the shell is coasting with constant
# Gamma^2beta^2~Gamma^2=Gshmax^2=BM(U(tmin,x))
cdef double _tmin(double x, double Gshmax):
    return (bm*u*x/Gshmax**2)**(1/3)

cdef double _mu(double theta, double phi, double theta_v):
    return sin(theta)*sin(theta_v)*cos(phi) + cos(theta)*cos(theta_v)

    
# U=E*t^-3/(n0*mp*c5) = x*t^-3/(mpc5)
# t, x=E/n in units of 1e8 s and 1e53 ergs cm3
cdef double _U(double t, double x):
    return u * x * t**(-3)
# _BM = C_BM^2*t^-3 (eq.1 & 2)
cdef double _BM(double t, double x):
    return bm * _U(t, x)
# _ST = C_ST^2*t^-6/5
cdef double _ST(double t, double x):
    return st * _U(t, x)**(2/5)
cdef double _A(double t, double x):
    return _BM(t, x) + _ST(t, x)

#Gamma_shell^2 * beta_shell^2
cdef double _GB2(double t, double x, double Gshmax) :
    cdef double uval = _U(t,x)
    cdef double val = bm * uval + st * uval**(2/5)
    val = min(val, Gshmax**2)
    return val

#Gamma_shell^2
cdef double _G2(double t, double x, double Gshmax):
    return _GB2(t, x, Gshmax) + 1
#beta_shell^2
cdef double _B2(double t, double x, double Gshmax):
    cdef double gb2 = _GB2(t, x, Gshmax)
    return gb2 / (gb2 + 1)
cdef double _B(double t, double x, double Gshmax):
    return sqrt(_B2(t, x, Gshmax))


def U(t, x):
    return u * x * t**(-3)

cdef double _U2t(double U, double x):
    cdef double res = U/u/x
    return res**(-1/3)

# _BM = C_BM^2*t^-3 (eq.1 & 2)
def BM(t, x):
    return bm * U(t, x)

# _ST = C_ST^2*t^-6/5
def ST(t, x):
    st = (2*1.15/5)**2
    return st * U(t, x)**(2/5)
def A(t, x):
    return BM(t, x) + ST(t, x)
#Gamma_shell^2 * beta_shell^2
def GB2(t, x, Gshmax) :
    val = A(t, x)
    val[val>=Gshmax**2] = Gshmax**2
    return val
#Gamma_shell^2
def G2(t, x, Gshmax):
    return GB2(t, x, Gshmax) + 1
#beta_shell^2
def B2(t, x, Gshmax):
    return GB2(t, x, Gshmax) / (GB2(t, x, Gshmax)+1)
def B(t, x, Gshmax):
    return np.sqrt(B2(t, x, Gshmax))

cdef double _gaussian_profile_scalar(double theta, double theta_G, double theta_c, double E_g):
    cdef double res = 0 
    if theta<=theta_c:
        res =  E_g * exp(-0.5*(theta/theta_G)**2)
    return res

#R/c eq.4
cdef double _R_over_c(double t, double x, double Gshmax):
    cdef double tmin = (bm*u*x/Gshmax**2)**(1/3)
    cdef double bmax = sqrt(_GB2(tmin,x,Gshmax)/(_GB2(tmin,x,Gshmax)+1))
    cdef double res = sciint.quad(_B,tmin,t,args=(x, Gshmax))[0]
    return tmin*bmax + res

def approxR(double t, double x, double Gshmax, double bmax, double tmin, double t1, double t2):
    if t<=tmin:
        return bmax * t
    if t>tmin and t<=t1:
        return bmax*tmin + t - tmin + 1/(8*bm*u*x)*(t**4 - tmin**4)
    if t>t1 and t<=t2:
        return bmax*tmin + t1 - tmin + 1/(8*bm*u*x)*(t1**4 - tmin**4) + sciint.quad(_B, args=(x,Gshmax),a=t1, b=t)[0]
    if t>t2 :
        return bmax*tmin + t1 - tmin + 1/(8*bm*u*x)*(t1**4 - tmin**4) + sciint.quad(_B, args=(x,Gshmax),a=t1, b=t2)[0] \
            + sqrt(st)*(u*x)**(1/5) * 5/2*(t**(2/5)-t2**(2/5))


cdef double _B_deriv(double t, double x, double Gshmax, double tmin):
    if t<=tmin: return 0
    cdef double gb2 = _GB2(t,x,Gshmax)
    cdef double gb2_prime = -3/t*(_BM(t,x) + 2/5*_ST(t,x))
    return 0.5*(gb2/gb2_prime)**2*_B(t,x,Gshmax)

cdef double _func(double t, double x, double Gshmax, double T, double mu, double bmax, double tmin, double t1, double t2):
    return t - T - mu * approxR(t, x, Gshmax, bmax, tmin, t1, t2)

cdef double _func_deriv(double t, double x, double Gshmax, double T, double mu, double bmax, double tmin, double t1, double t2):
    return 1 - mu * _B(t, x, Gshmax)

cdef double _func_second_deriv(double t, double x, double Gshmax, double T, double mu, double bmax, double tmin, double t1, double t2):
    return - mu * _B_deriv(t, x, Gshmax, tmin)

cdef double _find_emission_time_scalar(double T, double x, double Gshmax, double mu, double tol=1, int maxiter=50, bint deriv2=False):
    cdef double res=0.
    cdef double bmax = sqrt(Gshmax**2/(Gshmax**2+1))
    cdef double tmin = (bm*u*x/Gshmax**2)**(1/3)
    cdef double t1 = _U2t(1.e3, x)
    cdef double t2 = _U2t(1.e-6, x)
    if deriv2:
        res = sciopt.newton(_func, x0=T, fprime=_func_deriv,\
                            args=(x, Gshmax, T, mu, bmax, tmin, t1, t2), maxiter=maxiter, tol=tol, fprime2=_func_second_deriv).real
    else:
        res = sciopt.newton(_func, x0=T, fprime=_func_deriv,\
                            args=(x, Gshmax, T, mu, bmax, tmin, t1, t2), maxiter=maxiter, tol=tol).real
#    print(res)
    return res

def find_emission_time_scalar(double T, double x, double Gshmax, double mu, double tol=1, int maxiter=50, bint deriv2=False):
    return _find_emission_time_scalar(T, x, Gshmax, mu, tol, maxiter, deriv2)

#Lorentz factor of the shock fluid (eq.12)
def _G(double x):
    cdef double val
    cdef g600 = 1.0025051996609338
    if x>=600:
        val = 1 + (g600-1)*600/x
    else:
        val = kn(3, x)/kn(2,x) - 1/x
    return val

#ratio of specific heats eq.11
def _g2(double x):
    cdef double val, k2
    #need an approximation at large x
    cdef double g600 = 1.665282968322527
    if x>=600:
        val = 5/3 + (g600 - 5/3)*600/x
    else:
        k2 = kn(2,x)
        val = 1 + 4*k2 / x / (3*kn(3,x)+kn(1,x)-4*k2)
    return val

#same but with a more direct equation.
def _g(double x):
    cdef double val, k2
    #need an approximation at large x
    g600 = 1.665282968322527
    if x>=600:
        val = 5/3 + (g600 - 5/3)*600/x
    else:
        k2 = kn(2,x)
        val = 1 + k2/(x*(kn(3,x) - k2) - k2 )
    return val

def _rootfunc(double x, double Gsh):
    Gamma = _G(x)
    gamma = _g(x)
    return (Gamma+1)*(1+gamma*(Gamma-1))**2 / (2+gamma*(2-gamma)*(Gamma-1)) - Gsh**2

#solve zeta buy solving eq.5 for a given Gamma_shell
cdef double _find_zeta(double Gsh):
    cdef double ba = 1/Gsh
    cdef double bb = 100*Gsh
    if Gsh<1.15:
        ba = 2/(Gsh**2-1)
        bb = 6/(Gsh**2-1)
    if Gsh>6:
        ba=3/Gsh
        bb=5/Gsh
    cdef double val = sciopt.root_scalar(_rootfunc, bracket=(ba,bb), method='bisect',\
                                         args=(Gsh,), maxiter=50).root
    return val

#eq.6
def _n_prime(double x, double n0) :
    return n0 * (_g(x) * _G(x) + 1) / (_g(x)-1)

#downstream internal energy
#eq 7
def _e_prime_i(double zeta, double n0):
    cdef double nprime = _n_prime(zeta, n0)
    cdef Gamma = _G(zeta)
    return nprime * mpc2 * (Gamma-1)

#eq.17-19, prescription from Keshet&Waxman 2005 for the electron index
cdef double _e_index(double beta, double bsh):
#    cdef Gamma = _G(zeta)
#    cdef double beta = sqrt(1 - 1/Gamma**2)
    cdef double bu = bsh
    cdef double bd = (bsh-beta)/(1-beta*bsh)
    return (3*bu - 2*bu*bd**2 + bd**3) / (bu - bd) - 2

def e_index(double beta, double bsh):
    return _e_index(beta, bsh)

#eq.15
cdef double _g_prime_m(double p, double Gamma, double bsh, double eps_e):
#    cdef double p = _e_index(zeta, bsh)
#    cdef double Gamma = _G(zeta)
    return max(1, (p-2)/(p-1) * mpratio * eps_e * (Gamma-1))

#eq.16
cdef double _n_prime_R(double nprime, double Gamma, double p, double eps_e):
#    cdef double nprime = _n_prime(zeta, n0)
#    cdef double Gamma = _G(zeta)
    return nprime * min(1, (p-2)/(p-1) * mpratio * eps_e * (Gamma-1))

def n_prime_R(double nprime, double Gamma, double p, double eps_e):
    return  _n_prime_R(nprime, Gamma, p, eps_e)

#eq22: characteristic Lorentz factor for synch cooling
cdef double _g_prime_c(double Gamma, double eprime, double t, double eps_b):
#    cdef double Gamma = _G(zeta)
#    cdef double eprime = _e_prime_i(zeta, n0)
    return 3*mec * Gamma / (4*sigmaT*eps_b * eprime * t)

#eq.23
cdef double _B_prime(double eprime, double eps_b):
#    cdef double eprime = _e_prime_i(zeta, n0)
    return sqrt(8*pi*eps_b*eprime)
def B_prime(double zeta, double eps_b):
    return _B_prime(zeta, eps_b)

#eq.26
cdef double _peak_emissivity(double p, double nprimeR, double Bprime):
    return 0.88*256/27*qe**3/mec2 * (p-1)/(3*p-1) * nprimeR * Bprime

#eq.27
cdef double _nu_prime_m(double g_prime_m, double Bprime):
#    cdef double g_prime_m = _g_prime_m(zeta, bsh, eps_e)
    return 3/16 * qe / mec * g_prime_m**2 * Bprime

#eq.28
cdef double _nu_prime_c(double g_prime_c, double Bprime):
#    cdef double g_prime_c = _g_prime_c(zeta, t, eps_b, n0)
    return 3/16 * qe / mec * g_prime_c**2 * Bprime

#eq.24 & 25
cdef double _emissivity(double nu, double nu_prime_m, double nu_prime_c, double p, double peak_emiss):
    cdef double res = peak_emiss
    if nu_prime_m < nu_prime_c:
        if nu<nu_prime_m:
            return res * (nu / nu_prime_m)**(1/3)
        if nu>=nu_prime_c:
            return res * (nu_prime_c/nu_prime_m)**(-(p-1)/2) * (nu/nu_prime_c)**(-p/2.)
        return res * (nu / nu_prime_m)**(-(p-1)/2)
    if nu_prime_m >= nu_prime_c:
        if nu<nu_prime_c:
            return res * (nu / nu_prime_c)**(1/3)
        if nu>=nu_prime_m:
            return res * (nu_prime_m/nu_prime_c)**(-1/2) * (nu/nu_prime_m)**(-p/2.)
        return res * (nu/nu_prime_c)**(-1/2)

 
#eq.C15 frequency for the intercept of the two asymptotic behaviors
cdef double _nu_prime_0(double p, double nu_prime_min):
    cdef double res = nu_prime_min
    cdef double p4 = p/4
    cdef double N = gfunc(3/2+p4) * gfunc(11/6+p4) * gfunc(1/6+p4) * gfunc(5/6)
    cdef double D = (p+2) * gfunc(2+p4)

    res *= (N/D)**(6/(3*p+2))

    return res * (2**(3*(3*p-4)) / 27*pi**(3*p+8))**(1/(3*p+2))


#eq.C17 and C11
def _self_abs_coeff(double p, double nu0, double nu_prime_min, double n_prime_R, double g_prime_min, double Bprime):
    cdef double res = 2**6*pi**(5/6)*qe/15/gfunc(5/6) #~1.57e-18
    res *= (p+2)*(p-1)/(3*p+2)
    res *= n_prime_R / g_prime_min**5 / Bprime
    return res * (nu0 / nu_prime_min)**(-5/3)
    
#eq.C16
cdef double _alpha_prime(nu, nu_prime_0, p_tilde, self_abs_coeff):
    cdef double res = (nu/nu_prime_0)
    if nu>=nu_prime_0:
        res = res**(-(p_tilde+4)/2)
    else:
        res = res**(-5/3)
    return res * self_abs_coeff
    
#emissivity
def emissivity(double nu, double bsh, double t, double mu, double g,
               double b, double p, double eprime, double nprime, double eps_e, double eps_b):
    cdef double delta = g*(1-b*mu)
    cdef double nu_prime = nu * delta
    cdef double Bprime = _B_prime(eprime, eps_b)
    cdef double n_prime_R = _n_prime_R(nprime, g, p, eps_e)
    cdef double peak_emiss = _peak_emissivity(p, n_prime_R, Bprime)
    cdef double g_prime_m = _g_prime_m(p, g, bsh, eps_e)
    cdef double nu_prime_m = _nu_prime_m(g_prime_m, Bprime)
    cdef double g_prime_c = _g_prime_c(g, eprime, t, eps_b)
    cdef double nu_prime_c = _nu_prime_c(g_prime_c, Bprime)    
    cdef double eps_prime = _emissivity(nu_prime, nu_prime_m, nu_prime_c, p, peak_emiss)
    #self absorption
    cdef double g_prime_min = min(g_prime_c, g_prime_m)
    
    cdef double p_tilde = 2
    if g_prime_m < g_prime_c:
        p_tilde = p
    
    cdef double nu_prime_min = min(nu_prime_c, nu_prime_m)
    cdef double nu_prime_0 = _nu_prime_0(p_tilde, nu_prime_min)
    cdef double self_abs_coeff = _self_abs_coeff(p_tilde, nu_prime_0, nu_prime_min, n_prime_R, g_prime_min, Bprime)
    cdef double alpha_prime = _alpha_prime(nu_prime, nu_prime_0, p_tilde, self_abs_coeff)

    return eps_prime, alpha_prime, delta


def  integrand_scalar(double theta, double phi, double T, double nu,
                      double eps_e, double eps_b, double Gshmax, double n0,
                      double theta_v, double theta_G, double theta_c, double E_g):
    if theta==0: return 0. # integrand is proportional to sin(theta)
    cdef double eg = _gaussian_profile_scalar(theta, theta_G, theta_c, E_g)
    cdef double mu = cos(theta)*cos(theta_v) + sin(theta)*sin(theta_v)*cos(phi)
    cdef double t = _find_emission_time_scalar(T, eg/n0, Gshmax, mu)
    cdef double gb2 = _GB2(t, eg/n0, Gshmax)
    cdef double gsh = sqrt(gb2+1)
    cdef double bsh = sqrt(1-1/gsh**2)
    cdef double Roc = (t-T)/mu #approxR(t, eg/n0, Gshmax)
    
    cdef double res = abs(mu-bsh)/(1-mu*bsh) * Roc**2

    #shocked medium properties
    cdef double zeta = _find_zeta(gsh)
    cdef double g = _G(zeta)
    cdef double b = sqrt(1-1/g**2)
    cdef double p = _e_index(b, bsh)
    cdef double eprime = _e_prime_i(zeta, n0)
    cdef double nprime = _n_prime(zeta, n0)

    eps_prime, alpha_prime, delta = \
        emissivity(nu, bsh, t, mu, g,
                   b, p, eprime, nprime, eps_e,eps_b)
    res *= eps_prime
    res /= alpha_prime
    res /= delta**3
    
    cdef double delta_s = Roc*c / 12. / g**2 / abs(mu - bsh)
    cdef double tau = alpha_prime * delta * delta_s
    cdef double extinction = 1-exp(-tau)
    if extinction<1.e-6:
        extinction = tau
    res *= extinction

    res *= sin(theta)

    return res

def integral_theta_scalar(phi, T, nu, eps_e, eps_b, Gshmax,n0, theta_v, theta_G, theta_c, E_g):
    return sciint.quad(integrand_scalar, 0,theta_c, args=(phi, T, nu, eps_e, eps_b, Gshmax, \
                    n0, theta_v, theta_G, theta_c, E_g), epsabs=0.1, epsrel=1.e-6)[0]

def integral_phi_scalar(T, nu, eps_e, eps_b, Gshmax,n0, theta_v, theta_G, theta_c, E_g):
    return sciint.quad(integral_theta_scalar, 0, pi, args=(T, nu, eps_e, eps_b, Gshmax, \
                    n0, theta_v, theta_G, theta_c, E_g), epsabs=0.1, epsrel=1.e-6)

