import numpy as np
import scipy.integrate
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate


## unit for each parameter is commented after it is declared (inline)
## if the suggested value for the parameter is different -
## The original default value from the original ADM1 report by Batstone et al (2002), is commented after each unit (inline)

##constant definition from the Rosen et al (2006) BSM2 report
R =  0.083145 #bar.M^-1.K^-1
T_base =  298.15 #K
p_atm =  1.013 #bar
T_op =  308.15 #k ##T_ad #=35 C

##parameter definition from the Rosen et al (2006) BSM2 report bmadm1_report
# Stoichiometric parameter
f_sI_xc =  0.1
f_xI_xc =  0.2
f_ch_xc =  0.2
f_pr_xc =  0.2
f_li_xc =  0.3
N_xc =  0.0376 / 14
N_I =  0.06 / 14 #kmole N.kg^-1COD
N_aa =  0.007 #kmole N.kg^-1COD
C_xc =  0.02786 #kmole C.kg^-1COD
C_sI =  0.03 #kmole C.kg^-1COD
C_ch =  0.0313 #kmole C.kg^-1COD
C_pr =  0.03 #kmole C.kg^-1COD
C_li =  0.022 #kmole C.kg^-1COD
C_xI =  0.03 #kmole C.kg^-1COD
C_su =  0.0313 #kmole C.kg^-1COD
C_aa =  0.03 #kmole C.kg^-1COD
f_fa_li =  0.95
C_fa =  0.0217 #kmole C.kg^-1COD
f_h2_su =  0.19
f_bu_su =  0.13
f_pro_su =  0.27
f_ac_su =  0.41
N_bac =  0.08 / 14 #kmole N.kg^-1COD
C_bu =  0.025 #kmole C.kg^-1COD
C_pro =  0.0268 #kmole C.kg^-1COD
C_ac =  0.0313 #kmole C.kg^-1COD
C_bac =  0.0313 #kmole C.kg^-1COD
Y_su =  0.1
f_h2_aa =  0.06
f_va_aa =  0.23
f_bu_aa =  0.26
f_pro_aa =  0.05
f_ac_aa =  0.40
C_va =  0.024 #kmole C.kg^-1COD
Y_aa =  0.08
Y_fa =  0.06
Y_c4 =  0.06
Y_pro =  0.04
C_ch4 =  0.0156 #kmole C.kg^-1COD
Y_ac =  0.05
Y_h2 =  0.06


# Biochemical parameter values from the Rosen et al (2006) BSM2 report
k_dis =  0.5 #d^-1
k_hyd_ch =  10 #d^-1
k_hyd_pr =  10 #d^-1
k_hyd_li =  10 #d^-1
K_S_IN =  10 ** -4 #M
k_m_su =  30 #d^-1
K_S_su =  0.5 #kgCOD.m^-3
pH_UL_aa =  5.5
pH_LL_aa =  4
k_m_aa =  50 #d^-1
K_S_aa =  0.3 ##kgCOD.m^-3
k_m_fa =  6 #d^-1
K_S_fa =  0.4 #kgCOD.m^-3
K_I_h2_fa =  5 * 10 ** -6 #kgCOD.m^-3
k_m_c4 =  20 #d^-1
K_S_c4 =  0.2 #kgCOD.m^-3
K_I_h2_c4 =  10 ** -5 #kgCOD.m^-3
k_m_pro =  13 #d^-1
K_S_pro =  0.1 #kgCOD.m^-3
K_I_h2_pro =  3.5 * 10 ** -6 #kgCOD.m^-3
k_m_ac =  8 #kgCOD.m^-3
K_S_ac =  0.15 #kgCOD.m^-3
K_I_nh3 =  0.0018 #M
pH_UL_ac =  7
pH_LL_ac =  6
k_m_h2 =  35 #d^-1
K_S_h2 =  7 * 10 ** -6 #kgCOD.m^-3
pH_UL_h2 =  6
pH_LL_h2 =  5
k_dec_X_su =  0.02 #d^-1
k_dec_X_aa =  0.02 #d^-1
k_dec_X_fa =  0.02 #d^-1
k_dec_X_c4 =  0.02 #d^-1
k_dec_X_pro =  0.02 #d^-1
k_dec_X_ac =  0.02 #d^-1
k_dec_X_h2 =  0.02 #d^-1
## M is kmole m^-3

# Physico-chemical parameter values from the Rosen et al (2006) BSM2 report
T_ad =  308.15 #K


K_w =  10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #2.08 * 10 ^ -14

K_a_va =  10 ** -4.86 #M  ADM1 value = 1.38 * 10 ^ -5
K_a_bu =  10 ** -4.82 #M #1.5 * 10 ^ -5
K_a_pro =  10 ** -4.88 #M #1.32 * 10 ^ -5
K_a_ac =  10 ** -4.76 #M #1.74 * 10 ^ -5


K_a_co2 =  10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #4.94 * 10 ^ -7
K_a_IN =  10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T_ad)) #M #1.11 * 10 ^ -9


k_A_B_va =  10 ** 10 #M^-1 * d^-1
k_A_B_bu =  10 ** 10 #M^-1 * d^-1
k_A_B_pro =  10 ** 10 #M^-1 * d^-1
k_A_B_ac =  10 ** 10 #M^-1 * d^-1
k_A_B_co2 =  10 ** 10 #M^-1 * d^-1
k_A_B_IN =  10 ** 10 #M^-1 * d^-1


p_gas_h2o =  0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad)) #bar #0.0557
k_p = 5 * 10 ** 4 #m^3.d^-1.bar^-1 #only for BSM2 AD conditions, recalibrate for other AD cases #gas outlet friction
k_L_a =  200.0 #d^-1
K_H_co2 =  0.035 * np.exp((-19410 / (100 * R))* (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #0.0271
K_H_ch4 =  0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #0.00116
K_H_h2 =  7.8 * 10 ** -4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_ad)) #Mliq.bar^-1 #7.38*10^-4

# Physical parameter values used in BSM2 from the Rosen et al (2006) BSM2 report
V_liq =  3400 #m^3
V_gas =  300 #m^3
V_ad = V_liq + V_gas #m^-3

# reading influent and initial condition data from csv files
influent_state = pd.read_csv("digester_influent.csv")
initial_state = pd.read_csv("digester_initial.csv")

# Function to set influent values for influent state variables at each simulation step
def setInfluent(i):
    global S_su_in, S_aa_in, S_fa_in, S_va_in, S_bu_in, S_pro_in, S_ac_in, S_h2_in,S_ch4_in, S_IC_in, S_IN_in, S_I_in,X_xc_in, X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,S_cation_in,S_anion_in
    ##variable definition
    # Input values (influent/feed) 
    S_su_in = influent_state['S_su'][i] #kg COD.m^-3
    S_aa_in = influent_state['S_aa'][i] #kg COD.m^-3
    S_fa_in = influent_state['S_fa'][i] #kg COD.m^-3
    S_va_in = influent_state['S_va'][i] #kg COD.m^-3
    S_bu_in = influent_state['S_bu'][i] #kg COD.m^-3
    S_pro_in = influent_state['S_pro'][i] #kg COD.m^-3
    S_ac_in = influent_state['S_ac'][i] #kg COD.m^-3
    S_h2_in = influent_state['S_h2'][i] #kg COD.m^-3
    S_ch4_in = influent_state['S_ch4'][i]  #kg COD.m^-3
    S_IC_in = influent_state['S_IC'][i] #kmole C.m^-3
    S_IN_in = influent_state['S_IN'][i] #kmole N.m^-3
    S_I_in = influent_state['S_I'][i] #kg COD.m^-3
    
    X_xc_in = influent_state['X_xc'][i] #kg COD.m^-3
    X_ch_in = influent_state['X_ch'][i] #kg COD.m^-3
    X_pr_in = influent_state['X_pr'][i] #kg COD.m^-3
    X_li_in = influent_state['X_li'][i] #kg COD.m^-3
    X_su_in = influent_state['X_su'][i] #kg COD.m^-3
    X_aa_in = influent_state['X_aa'][i] #kg COD.m^-3
    X_fa_in = influent_state['X_fa'][i] #kg COD.m^-3
    X_c4_in = influent_state['X_c4'][i] #kg COD.m^-3
    X_pro_in = influent_state['X_pro'][i] #kg COD.m^-3
    X_ac_in = influent_state['X_ac'][i] #kg COD.m^-3
    X_h2_in = influent_state['X_h2'][i] #kg COD.m^-3
    X_I_in = influent_state['X_I'][i] #kg COD.m^-3
    
    S_cation_in = influent_state['S_cation'][i] #kmole.m^-3
    S_anion_in = influent_state['S_anion'][i] #kmole.m^-3
    

# initiate variables (initial values for the reactor state at the initial time (t0)
S_su = initial_state['S_su'][0] #kg COD.m^-3 monosaccharides
S_aa = initial_state['S_aa'][0] #kg COD.m^-3 amino acids
S_fa = initial_state['S_fa'][0] #kg COD.m^-3 total long chain fatty acids 
S_va = initial_state['S_va'][0] #kg COD.m^-3 total valerate
S_bu = initial_state['S_bu'][0] #kg COD.m^-3 total butyrate
S_pro = initial_state['S_pro'][0] #kg COD.m^-3 total propionate
S_ac = initial_state['S_ac'][0] #kg COD.m^-3 total acetate
S_h2 = initial_state['S_h2'][0] #kg COD.m^-3 hydrogen gas 
S_ch4 = initial_state['S_ch4'][0] #kg COD.m^-3 methane gas
S_IC = initial_state['S_IC'][0] #kmole C.m^-3 inorganic carbon
S_IN = initial_state['S_IN'][0] #kmole N.m^-3 inorganic nitrogen
S_I = initial_state['S_I'][0] #kg COD.m^-3 soluble inerts

X_xc = initial_state['X_xc'][0] #kg COD.m^-3 composites
X_ch = initial_state['X_ch'][0] #kg COD.m^-3 carbohydrates
X_pr = initial_state['X_pr'][0] #kg COD.m^-3 proteins
X_li = initial_state['X_li'][0] #kg COD.m^-3 lipids
X_su = initial_state['X_su'][0] #kg COD.m^-3 sugar degraders
X_aa = initial_state['X_aa'][0] #kg COD.m^-3 amino acid degraders 
X_fa = initial_state['X_fa'][0] #kg COD.m^-3 LCFA degraders
X_c4 = initial_state['X_c4'][0] #kg COD.m^-3 valerate and butyrate degraders
X_pro = initial_state['X_pro'][0] #kg COD.m^-3 propionate degraders
X_ac = initial_state['X_ac'][0] #kg COD.m^-3 acetate degraders
X_h2 = initial_state['X_h2'][0] #kg COD.m^-3 hydrogen degraders
X_I = initial_state['X_I'][0] #kg COD.m^-3 particulate inerts

S_cation = initial_state['S_cation'][0] #kmole.m^-3 cations (metallic ions, strong base)
S_anion = initial_state['S_anion'][0] #kmole.m^-3 anions (metallic ions, strong acid)

pH = 7.4655377 # initial pH
S_H_ion = initial_state['S_H_ion'][0] #kmole H.m^-3
S_va_ion = initial_state['S_va_ion'][0] #kg COD.m^-3 valerate
S_bu_ion = initial_state['S_bu_ion'][0] #kg COD.m^-3 butyrate
S_pro_ion = initial_state['S_pro_ion'][0] #kg COD.m^-3 propionate
S_ac_ion = initial_state['S_ac_ion'][0] #kg COD.m^-3 acetate
S_hco3_ion = initial_state['S_hco3_ion'][0] #kmole C.m^-3 bicarbonate
S_nh3 = initial_state['S_nh3'][0] #kmole N.m^-3 ammonia
S_nh4_ion = 0.0041 #kmole N.m^-3 the initial value is from Rosen et al (2006) BSM2 report and it is calculated further down and does not need to be initiated
S_co2 = 0.14 #kmole C.m^-3 the initial value is from Rosen et al (2006) BSM2 report and it is calculated further down and does not need to be initiated
S_gas_h2 = initial_state['S_gas_h2'][0] #kg COD.m^-3 hydrogen concentration in gas phase
S_gas_ch4 = initial_state['S_gas_ch4'][0] #kg COD.m^-3 methane concentration in gas phase
S_gas_co2 = initial_state['S_gas_co2'][0]#kmole C.m^-3 carbon dioxide concentration in gas phas

# related to pH inhibition taken from BSM2 report, they are global variables to avoid repeating them in DAE part
K_pH_aa =  (10 ** (-1 * (pH_LL_aa + pH_UL_aa) / 2.0))
nn_aa =  (3.0 / (pH_UL_aa - pH_LL_aa)) #we need a differece between N_aa and n_aa to avoid typos and nn_aa refers to the n_aa in BSM2 report
K_pH_ac = (10 ** (-1 * (pH_LL_ac + pH_UL_ac) / 2.0))
n_ac =  (3.0 / (pH_UL_ac - pH_LL_ac))
K_pH_h2 =  (10 ** (-1 * (pH_LL_h2 + pH_UL_h2) / 2.0))
n_h2 =  (3.0 / (pH_UL_h2 - pH_LL_h2))

#pH equation
pH = - np.log10(S_H_ion)

setInfluent(0) #setting the influent for the initial time (t0) to be ready for the start of the simulation
q_ad =  178.4674 #m^3.d^-1 initial flow rate (can be modified during the simulation by the control algorithm)


state_zero = [S_su,
              S_aa,
              S_fa,
              S_va,
              S_bu,
              S_pro,
              S_ac,
              S_h2,
              S_ch4,
              S_IC,
              S_IN,
              S_I,
              X_xc,
              X_ch,
              X_pr,
              X_li,
              X_su,
              X_aa,
              X_fa,
              X_c4,
              X_pro,
              X_ac,
              X_h2,
              X_I,
              S_cation,
              S_anion,
              S_H_ion,
              S_va_ion,
              S_bu_ion,
              S_pro_ion,
              S_ac_ion,
              S_hco3_ion,
              S_co2,
              S_nh3,
              S_nh4_ion,
              S_gas_h2,
              S_gas_ch4,
              S_gas_co2]

state_input = [S_su_in,
              S_aa_in,
              S_fa_in,
              S_va_in,
              S_bu_in,
              S_pro_in,
              S_ac_in,
              S_h2_in,
              S_ch4_in,
              S_IC_in,
              S_IN_in,
              S_I_in,
              X_xc_in,
              X_ch_in,
              X_pr_in,
              X_li_in,
              X_su_in,
              X_aa_in,
              X_fa_in,
              X_c4_in,
              X_pro_in,
              X_ac_in,
              X_h2_in,
              X_I_in,
              S_cation_in,
              S_anion_in]


# Function for calulating the derivatives related to ADM1 system of equations from the Rosen et al (2006) BSM2 report
def ADM1_ODE(t, state_zero):
  global S_nh4_ion, S_co2, p_gas, q_gas, q_ch4
  S_su = state_zero[0]
  S_aa = state_zero[1]
  S_fa = state_zero[2]
  S_va = state_zero[3]
  S_bu = state_zero[4]
  S_pro = state_zero[5]
  S_ac = state_zero[6]
  S_h2 = state_zero[7]
  S_ch4 = state_zero[8]
  S_IC = state_zero[9]
  S_IN = state_zero[10]
  S_I = state_zero[11]
  X_xc = state_zero[12]
  X_ch = state_zero[13]
  X_pr = state_zero[14]
  X_li = state_zero[15]
  X_su = state_zero[16]
  X_aa = state_zero[17]
  X_fa = state_zero[18]
  X_c4 = state_zero[19]
  X_pro = state_zero[20]
  X_ac = state_zero[21]
  X_h2 = state_zero[22]
  X_I = state_zero[23]
  S_cation =  state_zero[24]
  S_anion = state_zero[25]
  S_H_ion =  state_zero[26]
  S_va_ion = state_zero[27]
  S_bu_ion = state_zero[28]
  S_pro_ion = state_zero[29]
  S_ac_ion = state_zero[30]
  S_hco3_ion =  state_zero[31]
  S_co2 = state_zero[32]
  S_nh3 = state_zero[33]
  S_nh4_ion =  state_zero[34]
  S_gas_h2 = state_zero[35]
  S_gas_ch4 = state_zero[36]
  S_gas_co2 = state_zero[37]


  S_su_in = state_input[0]
  S_aa_in = state_input[1]
  S_fa_in = state_input[2]
  S_va_in = state_input[3]
  S_bu_in =  state_input[4]
  S_pro_in =  state_input[5]
  S_ac_in =  state_input[6]
  S_h2_in =   state_input[7]
  S_ch4_in = state_input[8]
  S_IC_in = state_input[9]
  S_IN_in =  state_input[10]
  S_I_in = state_input[11]
  X_xc_in =  state_input[12]
  X_ch_in = state_input[13]
  X_pr_in = state_input[14]
  X_li_in =  state_input[15]
  X_su_in =  state_input[16]
  X_aa_in =  state_input[17]
  X_fa_in =  state_input[18]
  X_c4_in =  state_input[19]
  X_pro_in =  state_input[20]
  X_ac_in =  state_input[21]
  X_h2_in =  state_input[22]
  X_I_in = state_input[23]
  S_cation_in = state_input[24]
  S_anion_in = state_input[25]

  S_nh4_ion =  (S_IN - S_nh3)

  S_co2 =  (S_IC - S_hco3_ion)

  I_pH_aa =  ((K_pH_aa ** nn_aa) / (S_H_ion ** nn_aa + K_pH_aa ** nn_aa))
  I_pH_ac =  ((K_pH_ac ** n_ac) / (S_H_ion ** n_ac + K_pH_ac ** n_ac))
  I_pH_h2 =  ((K_pH_h2 ** n_h2) / (S_H_ion ** n_h2 + K_pH_h2 ** n_h2))
  I_IN_lim =  (1 / (1 + (K_S_IN / S_IN)))
  I_h2_fa =  (1 / (1 + (S_h2 / K_I_h2_fa)))
  I_h2_c4 =  (1 / (1 + (S_h2 / K_I_h2_c4)))
  I_h2_pro =  (1 / (1 + (S_h2 / K_I_h2_pro)))
  I_nh3 =  (1 / (1 + (S_nh3 / K_I_nh3)))

  I_5 =  (I_pH_aa * I_IN_lim)
  I_6 = I_5
  I_7 =  (I_pH_aa * I_IN_lim * I_h2_fa)
  I_8 =  (I_pH_aa * I_IN_lim * I_h2_c4)
  I_9 = I_8
  I_10 =  (I_pH_aa * I_IN_lim * I_h2_pro)
  I_11 =  (I_pH_ac * I_IN_lim * I_nh3)
  I_12 =  (I_pH_h2 * I_IN_lim)

 

  # biochemical process rates from Rosen et al (2006) BSM2 report
  Rho_1 =  (k_dis * X_xc)   # Disintegration
  Rho_2 =  (k_hyd_ch * X_ch)  # Hydrolysis of carbohydrates
  Rho_3 =  (k_hyd_pr * X_pr)  # Hydrolysis of proteins
  Rho_4 =  (k_hyd_li * X_li)  # Hydrolysis of lipids
  Rho_5 =  k_m_su * S_su / (K_S_su + S_su) * X_su * I_5  # Uptake of sugars
  Rho_6 =  (k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6)  # Uptake of amino-acids
  Rho_7 =  (k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7)  # Uptake of LCFA (long-chain fatty acids)
  Rho_8 =  (k_m_c4 * (S_va / (K_S_c4 + S_va )) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8)  # Uptake of valerate
  Rho_9 =  (k_m_c4 * (S_bu / (K_S_c4 + S_bu )) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9)  # Uptake of butyrate
  Rho_10 =  (k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10)  # Uptake of propionate
  Rho_11 =  (k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11)  # Uptake of acetate
  Rho_12 =  (k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12)  # Uptake of hydrogen
  Rho_13 =  (k_dec_X_su * X_su)  # Decay of X_su
  Rho_14 =  (k_dec_X_aa * X_aa)  # Decay of X_aa
  Rho_15 =  (k_dec_X_fa * X_fa)  # Decay of X_fa
  Rho_16 =  (k_dec_X_c4 * X_c4)  # Decay of X_c4
  Rho_17 =  (k_dec_X_pro * X_pro)  # Decay of X_pro
  Rho_18 =  (k_dec_X_ac * X_ac)  # Decay of X_ac
  Rho_19 =  (k_dec_X_h2 * X_h2)  # Decay of X_h2

  # acid-base rates for the BSM2 ODE implementation from Rosen et al (2006) BSM2 report
  Rho_A_4 =  (k_A_B_va * (S_va_ion * (K_a_va + S_H_ion) - K_a_va * S_va))
  Rho_A_5 =  (k_A_B_bu * (S_bu_ion * (K_a_bu + S_H_ion) - K_a_bu * S_bu))
  Rho_A_6 =  (k_A_B_pro * (S_pro_ion * (K_a_pro + S_H_ion) - K_a_pro * S_pro))
  Rho_A_7 =  (k_A_B_ac * (S_ac_ion * (K_a_ac + S_H_ion) - K_a_ac * S_ac))
  Rho_A_10 =  (k_A_B_co2 * (S_hco3_ion * (K_a_co2 + S_H_ion) - K_a_co2 * S_IC))
  Rho_A_11 =  (k_A_B_IN * (S_nh3 * (K_a_IN + S_H_ion) - K_a_IN * S_IN))

  # gas phase algebraic equations from Rosen et al (2006) BSM2 report
  p_gas_h2 =  (S_gas_h2 * R * T_op / 16)
  p_gas_ch4 =  (S_gas_ch4 * R * T_op / 64)
  p_gas_co2 =  (S_gas_co2 * R * T_op)


  p_gas=  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o)
  q_gas =  (k_p * (p_gas- p_atm))
  if q_gas < 0:    q_gas = 0

  q_ch4 = q_gas * (p_gas_ch4/p_gas) # methane flow

  # gas transfer rates from Rosen et al (2006) BSM2 report
  Rho_T_8 =  (k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2))
  Rho_T_9 =  (k_L_a * (S_ch4 - 64 * K_H_ch4 * p_gas_ch4))
  Rho_T_10 =  (k_L_a * (S_co2 - K_H_co2 * p_gas_co2))

  ##differential equaitons from Rosen et al (2006) BSM2 report
  # differential equations 1 to 12 (soluble matter)
  diff_S_su = q_ad / V_liq * (S_su_in - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5  # eq1

  diff_S_aa = q_ad / V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6  # eq2

  diff_S_fa = q_ad / V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7  # eq3

  diff_S_va = q_ad / V_liq * (S_va_in - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8  # eq4

  diff_S_bu = q_ad / V_liq * (S_bu_in - S_bu) + (1 - Y_su) * f_bu_su * Rho_5 + (1 - Y_aa) * f_bu_aa * Rho_6 - Rho_9  # eq5

  diff_S_pro = q_ad / V_liq * (S_pro_in - S_pro) + (1 - Y_su) * f_pro_su * Rho_5 + (1 - Y_aa) * f_pro_aa * Rho_6 + (1 - Y_c4) * 0.54 * Rho_8 - Rho_10  # eq6

  diff_S_ac = q_ad / V_liq * (S_ac_in - S_ac) + (1 - Y_su) * f_ac_su * Rho_5 + (1 - Y_aa) * f_ac_aa * Rho_6 + (1 - Y_fa) * 0.7 * Rho_7 + (1 - Y_c4) * 0.31 * Rho_8 + (1 - Y_c4) * 0.8 * Rho_9 + (1 - Y_pro) * 0.57 * Rho_10 - Rho_11  # eq7

  #diff_S_h2 is defined with DAE paralel equaitons

  diff_S_ch4 = q_ad / V_liq * (S_ch4_in - S_ch4) + (1 - Y_ac) * Rho_11 + (1 - Y_h2) * Rho_12 - Rho_T_9  # eq9


  ## eq10 start##
  s_1 =  (-1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI) 
  s_2 =  (-1 * C_ch + C_su)
  s_3 =  (-1 * C_pr + C_aa)
  s_4 =  (-1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa)
  s_5 =  (-1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac)
  s_6 =  (-1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac)
  s_7 =  (-1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac)
  s_8 =  (-1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac)
  s_9 =  (-1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac)
  s_10 =  (-1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac)
  s_11 =  (-1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac)
  s_12 =  ((1 - Y_h2) * C_ch4 + Y_h2 * C_bac)
  s_13 =  (-1 * C_bac + C_xc) 

  Sigma =  (s_1 * Rho_1 + s_2 * Rho_2 + s_3 * Rho_3 + s_4 * Rho_4 + s_5 * Rho_5 + s_6 * Rho_6 + s_7 * Rho_7 + s_8 * Rho_8 + s_9 * Rho_9 + s_10 * Rho_10 + s_11 * Rho_11 + s_12 * Rho_12 + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19))

  diff_S_IC = q_ad / V_liq * (S_IC_in - S_IC) - Sigma - Rho_T_10
  ## eq10 end##


 
  diff_S_IN = q_ad / V_liq * (S_IN_in - S_IN) + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I-f_pr_xc * N_aa) * Rho_1 - Y_su * N_bac * Rho_5 + (N_aa - Y_aa * N_bac) * Rho_6 - Y_fa * N_bac * Rho_7 - Y_c4 * N_bac * Rho_8 - Y_c4 * N_bac * Rho_9 - Y_pro * N_bac * Rho_10 - Y_ac * N_bac * Rho_11 - Y_h2 * N_bac * Rho_12 + (N_bac - N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19) # eq11 


  diff_S_I = q_ad / V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1  # eq12


  # Differential equations 13 to 24 (particulate matter)
  diff_X_xc = q_ad / V_liq * (X_xc_in - X_xc) - Rho_1 + Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19  # eq13 

  diff_X_ch = q_ad / V_liq * (X_ch_in - X_ch) + f_ch_xc * Rho_1 - Rho_2  # eq14 

  diff_X_pr = q_ad / V_liq * (X_pr_in - X_pr) + f_pr_xc * Rho_1 - Rho_3  # eq15 

  diff_X_li = q_ad / V_liq * (X_li_in - X_li) + f_li_xc * Rho_1 - Rho_4  # eq16 

  diff_X_su = q_ad / V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13  # eq17

  diff_X_aa = q_ad / V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14  # eq18

  diff_X_fa = q_ad / V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15  # eq19

  diff_X_c4 = q_ad / V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16  # eq20

  diff_X_pro = q_ad / V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17  # eq21

  diff_X_ac = q_ad / V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18  # eq22

  diff_X_h2 = q_ad / V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19  # eq23

  diff_X_I = q_ad / V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1  # eq24 

  # Differential equations 25 and 26 (cations and anions)
  diff_S_cation = q_ad / V_liq * (S_cation_in - S_cation)  # eq25

  diff_S_anion = q_ad / V_liq * (S_anion_in - S_anion)  # eq26

  diff_S_h2 = 0

  # Differential equations 27 to 32 (ion states, only for ODE implementation)
  diff_S_va_ion = 0  # eq27

  diff_S_bu_ion = 0  # eq28

  diff_S_pro_ion = 0  # eq29

  diff_S_ac_ion = 0  # eq30

  diff_S_hco3_ion = 0  # eq31

  diff_S_nh3 = 0  # eq32

  # Gas phase equations: Differential equations 33 to 35
  diff_S_gas_h2 = (q_gas / V_gas * -1 * S_gas_h2) + (Rho_T_8 * V_liq / V_gas)  # eq33

  diff_S_gas_ch4 = (q_gas / V_gas * -1 * S_gas_ch4) + (Rho_T_9 * V_liq / V_gas)  # eq34

  diff_S_gas_co2 = (q_gas / V_gas * -1 * S_gas_co2) + (Rho_T_10 * V_liq / V_gas)  # eq35

  diff_S_H_ion = 0
  diff_S_co2 = 0
  diff_S_nh4_ion = 0 #to keep the output same length as input for ADM1_ODE funcion


  return diff_S_su, diff_S_aa, diff_S_fa, diff_S_va, diff_S_bu, diff_S_pro, diff_S_ac, diff_S_h2, diff_S_ch4, diff_S_IC, diff_S_IN, diff_S_I, diff_X_xc, diff_X_ch, diff_X_pr, diff_X_li, diff_X_su, diff_X_aa, diff_X_fa, diff_X_c4, diff_X_pro, diff_X_ac, diff_X_h2, diff_X_I, diff_S_cation, diff_S_anion, diff_S_H_ion, diff_S_va_ion,  diff_S_bu_ion, diff_S_pro_ion, diff_S_ac_ion, diff_S_hco3_ion, diff_S_co2,  diff_S_nh3, diff_S_nh4_ion, diff_S_gas_h2, diff_S_gas_ch4, diff_S_gas_co2

# Function for integration of ADM1 differential equations 
def simulate(t_step, solvermethod):
  r = scipy.integrate.solve_ivp(ADM1_ODE, t_step, state_zero, method= solvermethod)
  return r.y

# Function for DAE equations adopted from the Rosen et al (2006) BSM2 report bmadm1_report
def DAESolve():
  global S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_nh3, S_H_ion, pH, p_gas_h2, S_h2, S_nh4_ion, S_co2, P_gas, q_gas
  
  ##  DAE calculations 
  eps = 0.0000001
  prevS_H_ion = S_H_ion
  
  #initial values for Newton-Raphson solver parameter
  shdelta = 1.0
  shgradeq = 1.0
  S_h2delta = 1.0
  S_h2gradeq = 1.0
  tol = 10 ** (-12) #solver accuracy tolerance
  maxIter = 1000 #maximum number of iterations for solver
  i = 1
  j = 1
  
  ## DAE solver for S_H_ion from Rosen et al. (2006)
  while ((shdelta > tol or shdelta < -tol) and (i <= maxIter)):
    S_va_ion = K_a_va * S_va / (K_a_va + S_H_ion)
    S_bu_ion = K_a_bu * S_bu / (K_a_bu + S_H_ion)
    S_pro_ion = K_a_pro * S_pro / (K_a_pro + S_H_ion)
    S_ac_ion = K_a_ac * S_ac / (K_a_ac + S_H_ion)
    S_hco3_ion = K_a_co2 * S_IC / (K_a_co2 + S_H_ion)
    S_nh3 = K_a_IN * S_IN / (K_a_IN + S_H_ion)
    shdelta = S_cation + (S_IN - S_nh3) + S_H_ion - S_hco3_ion - S_ac_ion / 64.0 - S_pro_ion / 112.0 - S_bu_ion / 160.0 - S_va_ion / 208.0 - K_w / S_H_ion - S_anion
    shgradeq = 1 + K_a_IN * S_IN / ((K_a_IN + S_H_ion) * (K_a_IN + S_H_ion)) + K_a_co2 * S_IC / ((K_a_co2 + S_H_ion) * (K_a_co2 + S_H_ion))               + 1 / 64.0 * K_a_ac * S_ac / ((K_a_ac + S_H_ion) * (K_a_ac + S_H_ion))               + 1 / 112.0 * K_a_pro * S_pro / ((K_a_pro + S_H_ion) * (K_a_pro + S_H_ion))               + 1 / 160.0 * K_a_bu * S_bu / ((K_a_bu + S_H_ion) * (K_a_bu + S_H_ion))               + 1 / 208.0 * K_a_va * S_va / ((K_a_va + S_H_ion) * (K_a_va + S_H_ion))               + K_w / (S_H_ion * S_H_ion)
    S_H_ion = S_H_ion - shdelta / shgradeq
    if S_H_ion <= 0:
        S_H_ion = tol
    i+=1
  
  # pH calculation
  pH = - np.log10(S_H_ion)
  
  #DAE solver for S_h2 from Rosen et al. (2006) 
  while ((S_h2delta > tol or S_h2delta < -tol) and (j <= maxIter)):
    I_pH_aa = (K_pH_aa ** nn_aa) / (prevS_H_ion ** nn_aa + K_pH_aa ** nn_aa)
  
    I_pH_h2 = (K_pH_h2 ** n_h2) / (prevS_H_ion ** n_h2 + K_pH_h2 ** n_h2)
    I_IN_lim = 1 / (1 + (K_S_IN / S_IN))
    I_h2_fa = 1 / (1 + (S_h2 / K_I_h2_fa))
    I_h2_c4 = 1 / (1 + (S_h2 / K_I_h2_c4))
    I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))
  
    I_5 = I_pH_aa * I_IN_lim
    I_6 = I_5
    I_7 = I_pH_aa * I_IN_lim * I_h2_fa
    I_8 = I_pH_aa * I_IN_lim * I_h2_c4
    I_9 = I_8
    I_10 = I_pH_aa * I_IN_lim * I_h2_pro
  
    I_12 = I_pH_h2 * I_IN_lim
    Rho_5 = k_m_su * (S_su / (K_S_su + S_su)) * X_su * I_5  # Uptake of sugars
    Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  # Uptake of amino-acids
    Rho_7 = k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7  # Uptake of LCFA (long-chain fatty acids)
    Rho_8 = k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va+ 1e-6)) * I_8  # Uptake of valerate
    Rho_9 = k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va+ 1e-6)) * I_9  # Uptake of butyrate
    Rho_10 = k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10  # Uptake of propionate
    Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12  # Uptake of hydrogen
    p_gas_h2 = S_gas_h2 * R * T_ad / 16
    Rho_T_8 = k_L_a * (S_h2 - 16 * K_H_h2 * p_gas_h2)
    S_h2delta = q_ad / V_liq * (S_h2_in - S_h2) + (1 - Y_su) * f_h2_su * Rho_5 + (1 - Y_aa) * f_h2_aa * Rho_6 + (1 - Y_fa) * 0.3 * Rho_7 + (1 - Y_c4) * 0.15 * Rho_8 + (1 - Y_c4) * 0.2 * Rho_9 + (1 - Y_pro) * 0.43 * Rho_10 - Rho_12 - Rho_T_8
    S_h2gradeq = - 1.0 / V_liq * q_ad - 3.0 / 10.0 * (1 - Y_fa) * k_m_fa * S_fa / (K_S_fa + S_fa) * X_fa * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_fa) * (1 + S_h2 / K_I_h2_fa)) / K_I_h2_fa - 3.0 / 20.0 * (1 - Y_c4) * k_m_c4 * S_va * S_va / (K_S_c4 + S_va) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 1.0 / 5.0 * (1 - Y_c4) * k_m_c4 * S_bu * S_bu / (K_S_c4 + S_bu) * X_c4 / (S_bu + S_va + eps) * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_c4 ) * (1 + S_h2 / K_I_h2_c4 )) / K_I_h2_c4 - 43.0 / 100.0 * (1 - Y_pro) * k_m_pro * S_pro / (K_S_pro + S_pro) * X_pro * I_pH_aa / (1 + K_S_IN / S_IN) / ((1 + S_h2 / K_I_h2_pro ) * (1 + S_h2 / K_I_h2_pro )) / K_I_h2_pro - k_m_h2 / (K_S_h2 + S_h2) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) + k_m_h2 * S_h2 / ((K_S_h2 + S_h2) * (K_S_h2 + S_h2)) * X_h2 * I_pH_h2 / (1 + K_S_IN / S_IN) - k_L_a
    S_h2 = S_h2 - S_h2delta / S_h2gradeq
    if S_h2 <= 0:
        S_h2 = tol
    j+=1

## time array definition
t = influent_state['time']


# Initiate the cache data frame for storing simulation results
simulate_results = pd.DataFrame([state_zero])
columns = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "S_cation", "S_anion", "pH", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion", "S_hco3_ion", "S_co2", "S_nh3", "S_nh4_ion", "S_gas_h2", "S_gas_ch4", "S_gas_co2"]
simulate_results.columns = columns

# Setting the solver method for the simulate function
solvermethod = 'DOP853'
t0=0
n=0

# Initiate cache data frame for storing gasflow values
initflow = {'q_gas': [0], 'q_ch4': [0]}
gasflow = pd.DataFrame(initflow)
total_ch4 = 0

# Initiate cache data frame for storing feedflow values
initq = {'q_ad' : [170]}
feedflow = pd.DataFrame(initq)


## Dynamic simulation
# Loop for simlating at each time step and feeding the results to the next time step
for u in t[1:]:
  n+=1
  setInfluent(n)
  
  state_input = [S_su_in,S_aa_in,S_fa_in,S_va_in,S_bu_in,S_pro_in,S_ac_in,S_h2_in,S_ch4_in,S_IC_in,S_IN_in,S_I_in,X_xc_in,X_ch_in,X_pr_in,X_li_in,X_su_in,X_aa_in,X_fa_in,X_c4_in,X_pro_in,X_ac_in,X_h2_in,X_I_in,S_cation_in,S_anion_in]
  
  # Span for next time step
  tstep = [t0,u]

  # Solve and store ODE results for next step 
  sim_S_su, sim_S_aa, sim_S_fa, sim_S_va, sim_S_bu, sim_S_pro, sim_S_ac, sim_S_h2, sim_S_ch4, sim_S_IC, sim_S_IN, sim_S_I, sim_X_xc, sim_X_ch, sim_X_pr, sim_X_li, sim_X_su, sim_X_aa, sim_X_fa, sim_X_c4, sim_X_pro, sim_X_ac, sim_X_h2, sim_X_I, sim_S_cation, sim_S_anion, sim_S_H_ion, sim_S_va_ion, sim_S_bu_ion, sim_S_pro_ion, sim_S_ac_ion, sim_S_hco3_ion, sim_S_co2, sim_S_nh3, sim_S_nh4_ion, sim_S_gas_h2, sim_S_gas_ch4, sim_S_gas_co2 = simulate(tstep, solvermethod)

  # Store ODE simulation result states
  S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2 =   sim_S_su[-1], sim_S_aa[-1], sim_S_fa[-1], sim_S_va[-1], sim_S_bu[-1], sim_S_pro[-1], sim_S_ac[-1], sim_S_h2[-1], sim_S_ch4[-1], sim_S_IC[-1], sim_S_IN[-1], sim_S_I[-1], sim_X_xc[-1], sim_X_ch[-1], sim_X_pr[-1], sim_X_li[-1], sim_X_su[-1], sim_X_aa[-1], sim_X_fa[-1], sim_X_c4[-1], sim_X_pro[-1], sim_X_ac[-1], sim_X_h2[-1], sim_X_I[-1], sim_S_cation[-1], sim_S_anion[-1], sim_S_H_ion[-1], sim_S_va_ion[-1], sim_S_bu_ion[-1], sim_S_pro_ion[-1], sim_S_ac_ion[-1], sim_S_hco3_ion[-1], sim_S_co2[-1], sim_S_nh3[-1], sim_S_nh4_ion[-1], sim_S_gas_h2[-1], sim_S_gas_ch4[-1], sim_S_gas_co2[-1]
  
  # Solve DAE states
  DAESolve()

  # Algebraic equations 
  p_gas_h2 =  (S_gas_h2 * R * T_op / 16)
  p_gas_ch4 =  (S_gas_ch4 * R * T_op / 64)
  p_gas_co2 =  (S_gas_co2 * R * T_op)
  p_gas=  (p_gas_h2 + p_gas_ch4 + p_gas_co2 + p_gas_h2o)
  q_gas =  (k_p * (p_gas- p_atm))
  if q_gas < 0:    
    q_gas = 0
  
  q_ch4 = q_gas * (p_gas_ch4/p_gas) # methane flow
  if q_ch4 < 0:
    q_ch4 = 0

  flowtemp = {'q_gas' : q_gas, 'q_ch4' : q_ch4}
  gasflow = gasflow.append(flowtemp, ignore_index=True)

  S_nh4_ion =  (S_IN - S_nh3)
  S_co2 =  (S_IC - S_hco3_ion)
  total_ch4 = total_ch4 + q_ch4 


  #state transfer
  state_zero = [S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN, S_I, X_xc, X_ch, X_pr, X_li, X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, S_cation, S_anion, S_H_ion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3_ion, S_co2, S_nh3, S_nh4_ion, S_gas_h2, S_gas_ch4, S_gas_co2]
  
  dfstate_zero = pd.DataFrame([state_zero], columns = columns)
  simulate_results = simulate_results.append(dfstate_zero)
  t0 = u
      

# Write the dynamic simulation resutls to csv
phlogarray = -1 * np.log10(simulate_results['pH'])
simulate_results['pH'] = phlogarray
simulate_results.to_csv("dynamic_out.csv", index = False)

## ring test begin
# to compare the resutls with the dynamic simulation data from the BSM2 Matlab implementation
# pyOut = pd.read_csv("dynamic_out.csv")
# pyIn = pd.read_csv("digester_influent.csv")
# MatlabOut = pd.read_csv("Matlabout_dyn.csv")

# pyOut.time = pyIn.time
# pyOut.Q = pyIn.Q
# MatlabOut.Q = pyOut.Q
# mvalue = pvalue = 0
# ringtest = pd.DataFrame(columns=["state", "Matlab", "Python", "error"])

# n = 0
# for i in pyOut.columns:
#   Matlabinteg = integrate.trapz(MatlabOut[i] , MatlabOut.time)
#   pyinteg = integrate.trapz(pyOut[i] , pyOut.time)
#   results =pd.DataFrame([[MatlabOut[i].name, Matlabinteg/280, pyinteg/280, abs(pyinteg-Matlabinteg)/280]], columns=["state", "Matlab", "Python", "error"])
#   ringtest = ringtest.append(results)
#   print("Matlab " + MatlabOut[i].name + " average = " + str(Matlabinteg/280) + " Python " +  pyOut[i].name + " average = " + str(pyinteg/280) + " Error =" + str(abs(pyinteg-Matlabinteg)/280))
  
# ringtest.to_csv("ringtest.csv", index = False)


# for i in pyOut.columns:
#   plt.figure(figsize=(32, 8))
#   plt.plot(MatlabOut.time, MatlabOut[i], label = MatlabOut[i].name, linestyle="-", color = "red")
#   plt.plot(pyOut.time, pyOut[i], label = pyOut[i].name, linestyle="--", color = "blue")
#   plt.legend()
#   plt.show()

## ring test end

