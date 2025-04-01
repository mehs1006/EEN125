"""
Torque-speed mapping from mapped parameters
---------------------------------------------------
This example shows how you can use mapped parameters and Matlab fmincon to 
calculate optimal points for given e-machine.

Torbjörn Thiringer
Meng-Ju Hsieh
Created on Wed Feb 19 12:10:08 2025

"""
# %% Import required modules
from collections import defaultdict
import pandas as pd
import numpy as np
from numpy import pi as pi
from numpy import sin as sin
from numpy import cos as cos
from numpy import sqrt as sq
import matplotlib.pyplot as plt
from statistics import mean as mean
from scipy.interpolate import interp1d, interp2d

import os
import ansys.aedt.core
import pdb
# %% Parameters for Maxwell 2D
# ~~~~~~~~~~~~~~~~~~~~~
# Providing the version, path to the project, and the design name and type for
# Maxwell 2D.

##########################################################
# Set AEDT version
# ~~~~~~~~~~~~~~~~
# Set AEDT version.

aedt_version = "2025.1";              # Version of ANSYS

##########################################################
# Assign current folder
# ~~~~~~~~~~~~~~~~
# Assign current folder of python script.

PWD = (os.getcwd().replace("\\","/"))+"/";

##################################################################
# Assign folder to AEDT file, project, and the design name and type
# ~~~~~~~~~~~~~~~~~~~~~
# Assign folder to AEDT file, project, and the design name and type.
Path_AEDT = "C:/Users/mengju/Documents/Ansoft/";

setup_name = "Setup1";
solver = "TransientXY";

DF_Params = pd.read_csv(PWD+"Design_parameters.csv");
project_name = DF_Params["Value"][DF_Params["Parameter"] == "Project"].iloc[0];
design_name = DF_Params["Value"][DF_Params["Parameter"] == "Design"].iloc[0];

# %% Parameters
R_s = float(DF_Params["Value"][DF_Params["Parameter"] == "R_s[Ohm]"].iloc[0]);
N_pp = float(DF_Params["Value"][DF_Params["Parameter"] == "N_pp"].iloc[0]);
I_max_rms = 260;   # Max phase rms current value
I_max = np.ceil(I_max_rms*sq(2)); # Amplitude of max current [A]
I_min = 5; # [A] Minimum current
I_step = 5; # [A] Current step for current sweep
List_I_s = np.arange(I_min, I_max+I_step*0.5, I_step).tolist();
V_dc = 430;    # Max DC voltage
U_max = np.ceil(V_dc/sq(3)*.95);
Trq_step = 10;
Trq_start = 0;
n_start = 0;
n_step = 1000;
n_final = 12000;
# %% Read mapped data
Dict_Vars = {
    "I_d": "[A]", "I_q": "[A]", "Psi_d": "[Wb]", "Psi_q": "[Wb]", 
    "U_d": "[V]", "U_q":"[V]", "U_s":"[V]", "Trq": "[Nm]", 
    "P_fe": "[W]", "P_PM": "[W]",
    "Psi_m": "[Wb]", "L_d": "[uH]", "L_q": "[uH]"
    };
Dict_Map = defaultdict();
for key, value in Dict_Vars.items():
    Name_file = key + value.replace("[","_in_").replace("]","")+"_map.csv";
    Dict_Map[key+value] = pd.read_csv(PWD+Name_file);
DF_Psi_m = pd.read_csv(PWD+"Psi_m(I_q).csv");
# %% Plot contours of L_d-L_q
# Define the arc parameters
theta = np.linspace(np.pi/2, np.pi, 100);  # 90-degree arc (0 to π/2 radians)
radius = I_max;  # Radius of the arc
# Compute the x and y coordinates
x = radius * np.cos(theta);
y = radius * np.sin(theta);
for Var in Dict_Map.keys():
    if Var != "I_d[A]" and Var != "I_q[A]":
        I_d = Dict_Map[Var].iloc[0, 1:];
        I_q = Dict_Map[Var]["I_q[A]"].iloc[1:];
        Map = Dict_Map[Var].iloc[1:, 1:];
       
        # Create contour plot with different colored lines (no fill)
        plt.figure(figsize=(8, 6), dpi=500);
        # Contour lines with color mapping
        contour_lines = plt.contour(
            I_d, I_q, Map, levels=10, cmap='rainbow'
            );
        # Add contour labels
        plt.clabel(
            contour_lines, inline=True, fontsize=10, fmt="%.2f", colors='black'
            );
        if Var == "Trq[Nm]" or Var == "U_s[V]":
            plt.plot(
                x, y, color='black', linewidth=2, linestyle='--'
                );
        # Customizations
        plt.xlim(-400,0);
        plt.ylim(0,400);
        plt.title("$"+Var+"$");
        plt.xlabel("$I_d$[A]", fontsize=14);
        plt.ylabel("$I_q$[A]", fontsize=14);
        plt.grid(True, linestyle='--', alpha=0.5);
        # Add colorbar to indicate values
        plt.colorbar(contour_lines);
# %% Send variables to Matlab
import matlab.engine
eng = matlab.engine.start_matlab();
MWS = eng.workspace;
eng.addpath(PWD, nargout=0); # Add path to Matlab
def fun_send_DataSet_2_Mtlb(Name_mat, Data):
    # Convert the DataFrame to a MATLAB array (matrix)
    if isinstance(Data, list):
        List = Data;
    elif isinstance(Data, np.ndarray):
        List = Data.tolist();
    elif isinstance(Data, pd.DataFrame):
        Arr = Data.to_numpy();
        List = Arr.tolist();
    elif isinstance(Data, pd.Series):
        Arr = Data.to_numpy();
        List = Arr.tolist();
    else:
        raise TypeError(
            f"Expected a list, np.array, or pd.dataframe, but got {type(Data).__name__}"
            );
    MWS[Name_mat] = matlab.double(List);
    eng.eval(Name_mat, nargout=0);
for Var in Dict_Map.keys():
    Name_mat = Var.split("[")[0];
    if "[uH]" in Var:
        fun_send_DataSet_2_Mtlb(Name_mat+"_mat", Dict_Map[Var].iloc[1:, 1:]*1e-6);
    else:
        fun_send_DataSet_2_Mtlb(Name_mat+"_mat", Dict_Map[Var].iloc[1:, 1:]);
fun_send_DataSet_2_Mtlb("Psi_m_vect", DF_Psi_m["Psi_m[Wb]"]);
fun_send_DataSet_2_Mtlb("I_q_vect", DF_Psi_m["I_q[A]"]);
MWS["R_s"] = R_s;
MWS["N_pp"] = float(N_pp);
MWS["I_max"] = I_max;
MWS["U_max"] = U_max;
eng.eval(
    'fun_L_d = @(I_d, I_q)interp2(I_d_mat, I_q_mat, L_d_mat, I_d, I_q, "spline");'+
    'fun_L_q = @(I_d, I_q)interp2(I_d_mat, I_q_mat, L_q_mat, I_d, I_q, "spline");'+
    'fun_Psi_m = @(I_q)interp1(I_q_vect, Psi_m_vect, I_q, "spline");'+
    'fun_Trq = @(I_d, I_q)interp2(I_d_mat, I_q_mat, Trq_mat, I_d, I_q, "spline");'
    , nargout=0
    );
# %% Find max torque with current constraint
options = eng.optimoptions(
    'fmincon', 'Algorithm','sqp','Display','off',
    'MaxFunEvals',4000,'MaxIter',4000
    );
eng.eval(
    'A = [];'+
    'b = [];'+
    'Aeq = [];'+
    'beq = [];'+
    'lb=[-1.2*I_max, 0*I_max];'+
    'ub=[0*I_max, 1.2*I_max];'+
    'x0=[-10; 10];'+
    'fun = @(x)fun_max_Trq(x, fun_Psi_m, fun_L_d, fun_L_q, N_pp);'+
    'nonlcon = @(x)fun_constr_max_current(x, I_max);'
    , nargout=0
    );
I_dq = eng.fmincon(
    MWS['fun'], MWS['x0'], MWS['A'], MWS['b'], MWS['Aeq'], MWS['beq'], 
    MWS['lb'], MWS['ub'], MWS['nonlcon'], options
    );
I_d_Tmax = I_dq[0][0]; # Convert to python float
I_q_Tmax = I_dq[1][0]; # Convert to python float
Trq_max = eng.eval(
    'fun_Trq('+str(I_d_Tmax)+', '+str(I_q_Tmax)+');'
    );
# %% Find boundary of MTPA (Max Torque Per Ampere) by Fmincom
Dict_OP = {"I_s": "[A]", "I_d": "[A]", "I_q": "[A]",
           "Psi_m": "[Wb]", "L_d": "[uH]", "L_q": "[uH]",
           "Trq": "[Nm]", "U_s": "[V]", "n_r": "[rpm]"
           };

List_Trq = np.arange(Trq_step, Trq_max, Trq_step).tolist() + [Trq_max];
options = eng.optimoptions(
    'fmincon', 'Algorithm','sqp','Display','off',
    'MaxFunEvals',3000,'MaxIter',1000
    );
options_f = eng.optimoptions('fsolve','Display','off'); # fsolve setting
eng.eval(
    'lb=[-1.2*I_max, 0.001*I_max];'+
    'ub=[-0.001*I_max, 1.2*I_max];'+
    'x0=[-10; 10];'+
    'fun_to_min_current = @(x)(x(1)^2+x(2)^2);'+
    'fun = fun_to_min_current;'
    , nargout=0
    );
Dict_Bndry = defaultdict(list);
Dict_Bndry["MTPA:Trq_ref[Nm]"] = List_Trq;
for T_ref in List_Trq:
    MWS["T_ref"] = T_ref;
    eng.eval(
        'nonlcon = @(x)fun_constr_Te_equals_Tref(x, fun_Trq, T_ref);'
        , nargout=0
        );
    I_dq = eng.fmincon(
        MWS['fun'], MWS['x0'], MWS['A'], MWS['b'], MWS['Aeq'], MWS['beq'], 
        MWS['lb'], MWS['ub'], MWS['nonlcon'], options
        );
    I_d = I_dq[0][0];
    I_q = I_dq[1][0];
    I_s = sq(I_d**2 + I_q**2);
    Psi_m = eng.eval(
        'fun_Psi_m('+str(I_q)+');'
        );
    L_d = eng.eval(
        'fun_L_d('+str(I_d)+', '+str(I_q)+');'
        );
    L_q = eng.eval(
        'fun_L_q('+str(I_d)+', '+str(I_q)+');'
        );
    Trq = eng.eval(
        'fun_Trq('+str(I_d)+', '+str(I_q)+');'
        );
    MWS["I_d"] = I_d;
    MWS["I_q"] = I_q;
    MWS["Psi_m"] = Psi_m;
    MWS["L_d"] = L_d;
    MWS["L_q"] = L_q;
    eng.eval(
        'fun_omega0=@(x)(R_s*I_d-x*I_q*L_q)^2+(R_s*I_q+x*I_d*L_d+x*Psi_m)^2-U_max^2;'+
        'x0_f=N_pp*4000/30*pi;'
        , nargout=0
        );
    omega0 = eng.fsolve(MWS['fun_omega0'], MWS['x0_f'], options_f);
    n_r = omega0/N_pp/(2*pi)*60;
    for key, value in Dict_OP.items():
        if key != "U_s" and value != "[uH]":
            Dict_Bndry["MTPA:"+key+value].append(globals()[key]);
        elif value == "[uH]":
            Dict_Bndry["MTPA:"+key+value].append(globals()[key]*1e6);
    eng.eval(
        'x0=['+str(I_d)+'; '+str(I_q)+'];'
        , nargout=0
        );
n_base = int(Dict_Bndry["MTPA:n_r[rpm]"][-1]/n_step*10)*n_step/10;
plt.figure(figsize=(6, 6), dpi=500);
plt.plot(
    Dict_Bndry["MTPA:I_d[A]"], Dict_Bndry["MTPA:I_q[A]"]
    );
plt.xlabel("$I_d$[A]", fontsize=14);
plt.ylabel("$I_q$[A]", fontsize=14);
plt.grid();

plt.figure(figsize=(8, 6), dpi=500);
plt.plot(
    Dict_Bndry["MTPA:I_s[A]"], Dict_Bndry["MTPA:Trq[Nm]"]
    );
plt.xlabel("$I_s$[A]", fontsize=14);
plt.ylabel("Torque[Nm]", fontsize=14);
plt.grid();

plt.figure(figsize=(8, 6), dpi=500);
plt.plot(
    Dict_Bndry["MTPA:I_s[A]"], Dict_Bndry["MTPA:Psi_m[Wb]"]
    );
plt.xlabel("$I_s$[A]", fontsize=14);
plt.ylabel("$Psi_m$[Wb]", fontsize=14);
plt.grid();

plt.figure(figsize=(8, 6), dpi=500);
plt.plot(
    Dict_Bndry["MTPA:I_s[A]"], Dict_Bndry["MTPA:L_d[uH]"]
    );
plt.xlabel("$I_s$[A]", fontsize=14);
plt.ylabel("$L_d$[uH]", fontsize=14);
plt.grid();

plt.figure(figsize=(8, 6), dpi=500);
plt.plot(
    Dict_Bndry["MTPA:I_s[A]"], Dict_Bndry["MTPA:L_q[uH]"]
    );
plt.xlabel("$I_s$[A]", fontsize=14);
plt.ylabel("$L_q$[uH]", fontsize=14);
plt.grid();
# %% Find boundary of FW (Field Weakening) & MTPV (Max Torque Per Voltage) by Fmincom
List_n_r_1 = np.arange(0, n_base, n_step).tolist();
List_n_r_1[0] = 1;
if List_n_r_1[-1] != n_base:
    List_n_r_1 = List_n_r_1 + [n_base];
List_n_r_2 = np.arange(n_base+n_step/10, List_n_r_1[-2]+1.05*n_step, n_step/10).tolist();
List_n_r_3 = np.arange(List_n_r_1[-2]+n_step*2, n_final+n_step*0.5, n_step).tolist();
List_n_r = List_n_r_1 + List_n_r_2 + List_n_r_3;
options = eng.optimoptions(
    'fmincon', 'Algorithm','sqp','Display','off',
    'MaxFunEvals',4000,'MaxIter',4000
    );
eng.eval(
    'lb=[-1.2*I_max, 0.001*I_max];'+
    'ub=[-0.001*I_max, 1.2*I_max];'+
    'x0=['+str(I_d)+'; '+str(I_q)+'];'+
    'fun = @(x)fun_max_Trq(x, fun_Psi_m, fun_L_d, fun_L_q, N_pp);'
    , nargout=0
    );
Dict_Bndry["n_r[rpm]"] = List_n_r;
for n_r in List_n_r:
    if n_r <= n_base:
        Dict_Bndry["Trq_max[Nm]"].append(Trq_max);
    else:
        omega0 = n_r/30*pi*N_pp;
        MWS["omega0"] = omega0;
        eng.eval(
            'nonlcon = @(x)fun_constr_Umax_and_Imax('+
            'x, omega0, fun_Psi_m, fun_L_d, fun_L_q, R_s, U_max, I_max'+
            ');'
            , nargout=0
            );
        I_dq = eng.fmincon(
            MWS['fun'], MWS['x0'], MWS['A'], MWS['b'], MWS['Aeq'], MWS['beq'], 
            MWS['lb'], MWS['ub'], MWS['nonlcon'], options
            );
        I_d = I_dq[0][0];
        I_q = I_dq[1][0];
        I_s = sq(I_d**2+I_q**2);
        Psi_m = eng.eval(
            'fun_Psi_m('+str(I_q)+');'
            );
        L_d = eng.eval(
            'fun_L_d('+str(I_d)+', '+str(I_q)+');'
            );
        L_q = eng.eval(
            'fun_L_q('+str(I_d)+', '+str(I_q)+');'
            );
        Trq = eng.eval(
            'fun_Trq('+str(I_d)+', '+str(I_q)+');'
            );
        if Trq > Trq_max:
            for key, value in Dict_OP.items():
                if key != "U_s":
                    globals()[key] = Dict_Bndry["MTPA:"+key+value][-1];
        Dict_Bndry["Trq_max[Nm]"].append(Trq);
        U_d = R_s*I_d-omega0*I_q*L_q;
        U_q = R_s*I_q+omega0*I_d*L_d+omega0*Psi_m;
        U_s = sq(U_d**2+U_q**2);
        for key, value in Dict_OP.items():
            Dict_Bndry["FW&MTPV:"+key+value].append(globals()[key]);
        eng.eval(
            'x0=['+str(I_d)+'; '+str(I_q)+'];'
            , nargout=0
            );
DF_Tmax = pd.DataFrame(Dict_Bndry, columns=["n_r[rpm]", "Trq_max[Nm]"]);
DF_Tmax.to_csv(
    PWD+"MaxTorque.csv", index=False
    );
plt.figure(figsize=(8, 6), dpi=500);
plt.plot(
    Dict_Bndry["n_r[rpm]"], Dict_Bndry["Trq_max[Nm]"]
    );
plt.xlim(0, n_final);
plt.ylim(0, np.ceil(Trq_max/100)*100+50);
plt.xlabel("$n_r$[rpm]", fontsize=14);
plt.ylabel("Torque[Nm]", fontsize=14);
plt.grid();
# %% Calculate the optimal points according boundaries
def fun_Is_min_4_Te_MTPA(Trq):
    Crv_I_d = interp1d(
        [0]+Dict_Bndry["MTPA:Trq[Nm]"], [0]+Dict_Bndry["MTPA:I_d[A]"]
        );
    I_d = Crv_I_d(Trq).tolist();
    Crv_I_q = interp1d(
        [0]+Dict_Bndry["MTPA:Trq[Nm]"], [0]+Dict_Bndry["MTPA:I_q[A]"]
        );
    I_q = Crv_I_q(Trq).tolist();
    n_r_0 = max(Dict_Bndry["MTPA:n_r[rpm]"]);
    Crv_n_r_max = interp1d(
        [0]+Dict_Bndry["MTPA:Trq[Nm]"], [n_r_0]+Dict_Bndry["MTPA:n_r[rpm]"]
        );
    n_r_max = Crv_n_r_max(Trq).tolist();
    return [ I_d, I_q, n_r_max ]
def fun_Tmax_4_nr_FW_MTPV(n_r):
    List_n_r = Dict_Bndry["FW&MTPV:n_r[rpm]"];
    List_Tmax_4_FW_MTPV = Dict_Bndry["FW&MTPV:Trq[Nm]"];
    f_Trq_max = interp1d(
        List_n_r, List_Tmax_4_FW_MTPV
        );
    Trq_max_4_nr = f_Trq_max(n_r).tolist();
    return Trq_max_4_nr
options = eng.optimoptions(
    'fmincon', 'Algorithm','active-set','Display','off',
    'MaxFunEvals',3000,'MaxIter',1000
    );
    # fmincon solver setting
eng.eval(
    'fun = fun_to_min_current;'
    'lb=[-1.2*I_max, 0.001*I_max];'+
    'ub=[-0.001*I_max, 1.2*I_max];'+
    'x0=[-10; 10];'
    , nargout=0
    );
List_Trq[-1] = float(int(Trq_max));
Dict_temp = defaultdict(list);
List_Trq_4_nr = [];
for idx_nr, n_r_ref in enumerate(List_n_r):
    omega0 = n_r_ref/30*pi*N_pp;
    MWS["omega0"] = omega0;
    for idx_Trq, Trq_ref in enumerate(List_Trq):
        [ I_d_MTPA, I_q_MTPA, n_r_max_MTPA ]  = fun_Is_min_4_Te_MTPA(Trq_ref);        
        if n_r_ref <= n_r_max_MTPA:
            Dict_temp["n_r"].append(str(int(n_r_ref)));
            Dict_temp["I_d"].append(str(round(I_d_MTPA, 8))+"A");
            Dict_temp["I_q"].append(str(round(I_q_MTPA, 8))+"A");
            List_Trq_4_nr.append(Trq_ref);
        elif Trq_ref <= fun_Tmax_4_nr_FW_MTPV(n_r_ref)+Trq_step:
            Trq_max_4_nr = round(fun_Tmax_4_nr_FW_MTPV(n_r_ref),2);
            Dict_temp["n_r"].append(str(int(n_r_ref)));
            if List_Trq[idx_Trq] > Trq_max_4_nr:
                Trq_ref = Trq_max_4_nr;
            MWS["T_ref"] = Trq_ref;
            eng.eval(
                'nonlcon = @(x)fun_constr_Umax_equal_Tref('+
                'x, fun_Psi_m, fun_L_d, fun_L_q, N_pp,'+
                'omega0, R_s, U_max, I_max, T_ref'+
                ');'
                , nargout=0
                );
            I_dq = eng.fmincon(
               MWS['fun'], MWS['x0'], MWS['A'], MWS['b'], MWS['Aeq'], MWS['beq'], 
               MWS['lb'], MWS['ub'], MWS['nonlcon'], options
               );
            I_d = I_dq[0][0];
            I_q = I_dq[1][0];
            Dict_temp["I_d"].append(str(round(I_d, 8))+"A");
            Dict_temp["I_q"].append(str(round(I_q, 8))+"A");
            List_Trq_4_nr.append(Trq_ref);
            eng.eval(
                'x0=['+str(I_d)+'; '+str(I_q)+'];'
                , nargout=0
                );
DF_OP_Pnts = pd.DataFrame.from_dict(Dict_temp);
DF_OP_Pnts["*"] = np.arange(1, len(Dict_temp["I_d"])+0.5, 1);
DF_OP_Pnts["*"] = DF_OP_Pnts["*"].astype(int);
DF_OP_Pnts = DF_OP_Pnts.iloc[ :, [ 3, 0, 1, 2 ] ];
DF_OP_Pnts.to_csv(
    PWD+"OperatingPoints.csv", index=False
    );
DF_OP_Pnts["Trq"] = List_Trq_4_nr;
Series_Id = DF_OP_Pnts["I_d"].str.rstrip('A').astype(float);
Series_Iq = DF_OP_Pnts["I_q"].str.rstrip('A').astype(float);
DF_OP_Pnts["I_s"] = sq(Series_Id**2 + Series_Id**2);
# %% Initialize Maxwell 2D
Dict_Mxwl_Vars = {
    "i_A": ["InputCurrent(Phase_A)", "[A]"],
    "i_B": ["InputCurrent(Phase_B)", "[A]"],
    "i_C": ["InputCurrent(Phase_C)", "[A]"],
    "psi_A": ["FluxLinkage(Phase_A)", "[Wb]"],
    "psi_B": ["FluxLinkage(Phase_B)", "[Wb]"],
    "psi_C": ["FluxLinkage(Phase_C)", "[Wb]"],
    "gamma": ["(Moving1.Position-MechAngle_d_oriented)*N_p/2*PI/180", "[rad]"],
    "u_i_A": ["InducedVoltage(Phase_A)", "[V]"],
    "u_i_B": ["InducedVoltage(Phase_B)", "[V]"],
    "u_i_C": ["InducedVoltage(Phase_C)", "[V]"],
    "Trq": ["Moving1.Torque", "[Nm]"],
    "P_fe": ["CoreLoss", "[W]"],
    "P_cu": ["StrandedLoss", "[W]"],
    "P_PM": ["SolidLoss", "[W]"]
};      
import os

file_path = Path_AEDT+project_name+".aedt.lock";
if os.path.exists(file_path):
    os.remove(file_path);
    print(f"{file_path} has been deleted.");

M2D = ansys.aedt.core.Maxwell2d(
    projectname=Path_AEDT+project_name+".aedt",
    designname=design_name
    );
desktop = ansys.aedt.core.Desktop();
List_designs = desktop.design_list(project=project_name);

if design_name+"_4_loss_map" in List_designs:
    design_name = design_name+"_4_loss_map";
    M2D = ansys.aedt.core.Maxwell2d(
        projectname=Path_AEDT+project_name+".aedt",
        designname=design_name
        );
else:
    desktop.copy_design(
        project_name = project_name, 
        design_name = design_name
        );
    TempName = design_name+"1";
    M2D.set_active_design(TempName);
    M2D.design_name = design_name+"_4_loss_map";

  
    for key, value in Dict_Mxwl_Vars.items():
        M2D.create_output_variable(key, value[0]);
    
    M2D.parametrics.add_from_file(
        PWD+"OperatingPoints.csv", 
        name="OperatingPoints"
        );
    M2D.analyze_setup(
        "OperatingPoints", cores=12, tasks=12, use_auto_settings=False
        );
    M2D.save_project();
# %% Extract raw data and make maps
List_Mxwl_Vars = list(Dict_Mxwl_Vars.keys());
Dict_OP_Vars = {
    "I_d": "[A]", "I_q": "[A]", "I_s": "[A]", 
    "Psi_d": "[Wb]", "Psi_q": "[Wb]", 
    "U_d": "[V]", "U_q": "[V]", "U_s": "[V]", 
    "Trq": "[Nm]", "P_fe": "[W]", "P_cu": "[W]", "P_PM": "[W]",
    "P_out": "[kW]", "P_loss": "[kW]", "Eff": "[%]"
    };
# Get available variations which has been solved in Ansys/Maxwell
Dict_vari = M2D.available_variations.nominal_w_values_dict;

# Dict_vari = defaultdict();
# Create a ditionaries to contain all output maps
Dict_OP_Map = defaultdict();
for Var, Unit in Dict_OP_Vars.items(): 
    Dict_OP_Map[Var+Unit] = defaultdict(list);
    Dict_OP_Map[Var+Unit]["Trq_ref[Nm]"] = [0] + List_Trq;

# Extract raw data from Ansys/Maxwell and map the values
for idx_nr, n_r_ref in enumerate(List_n_r):
    # Make key for operating points which sent to Ansys/Maxwell to run
    key_OP_Pnts = str(int(n_r_ref))+"rpm";
    DF_OP_Pnts_4_nr = DF_OP_Pnts[
        DF_OP_Pnts['n_r'] == str(int(n_r_ref)) 
        ].reset_index();
    # Make keys for given speed for the map of differenct output variables
    for Var, Unit in Dict_OP_Vars.items():
        Dict_OP_Map[Var+Unit]["n_r_"+str(idx_nr)].append(n_r_ref);
    # iterate reference torque
    for idx_Trq, Trq_ref in enumerate(List_Trq):        
        # Check if the points is within the boundaries
        if idx_Trq <= (len(DF_OP_Pnts_4_nr)-1):
            Dict_vari["n_r"] = DF_OP_Pnts_4_nr["n_r"][idx_Trq];
            I_d_in = DF_OP_Pnts_4_nr["I_d"][idx_Trq];
            I_q_in = DF_OP_Pnts_4_nr["I_q"][idx_Trq];
            Dict_vari["I_d"] = I_d_in;
            Dict_vari["I_q"] = I_q_in;
            
            # Create a dictionary to contain raw data from Ansys/Maxwell
            Dict_temp = defaultdict();
            # Extract raw data
            Mxwl_rawdata = M2D.post.get_solution_data(
                expressions=List_Mxwl_Vars, 
                variations=Dict_vari, 
                primary_sweep_variable="Time",
                domain="Sweep"
                );
            for Var, Unit in Dict_Mxwl_Vars.items():
                Var_rawdata = Mxwl_rawdata.data_real(Var);
                Unit_data = Mxwl_rawdata.units_data[Var];
                # pdb.set_trace();
                try:
                    if Unit_data[0] == "m":
                        Var_rawdata = [ Var_rawdata[x]*1e-3 for x in range(len(Var_rawdata)) ];
                    elif Unit_data[0] == "u":
                        Var_rawdata = [ Var_rawdata[x]*1e-6 for x in range(len(Var_rawdata)) ];
                    elif Unit_data[0] == "k":
                        Var_rawdata = [ Var_rawdata[x]*1e3 for x in range(len(Var_rawdata)) ];
                except IndexError:
                    pass;
                Dict_temp[Var+Unit[1]] = Var_rawdata;
                DF_temp = pd.DataFrame.from_dict(Dict_temp);

            I_d_in = float(I_d_in.replace("A", ""));
            I_q_in = float(I_q_in.replace("A", ""));
            i_alpha = DF_temp["i_A[A]"];
            i_beta = (DF_temp["i_B[A]"]-DF_temp["i_C[A]"])/sq(3);
            gamma = DF_temp["gamma[rad]"];
            I_d = round(
                mean(
                    i_alpha*cos(gamma)+i_beta*sin(gamma)
                    ), 3
                );
            I_q = round(
                mean(
                    -i_alpha*sin(gamma)+i_beta*cos(gamma)
                    ), 3
                );
            # check that simulation was done according to desired Id and Iq
            I_diff = sq((I_d_in - I_d)**2 + (I_q_in - I_q)**2);
            if I_diff > 1:
                print("***  Warning  ******  desired Id/Iq not obatined");
                break;
            I_s = sq(I_d**2 + I_q**2);
            
            psi_alpha = DF_temp["psi_A[Wb]"];
            psi_beta = (DF_temp["psi_B[Wb]"]-DF_temp["psi_C[Wb]"])/sq(3);
            Psi_d = mean(
                psi_alpha*cos(gamma)+psi_beta*sin(gamma)
                );
            Psi_q = mean(
                -psi_alpha*sin(gamma)+psi_beta*cos(gamma)
                );    
            
            u_A = DF_temp["u_i_A[V]"]+R_s*DF_temp["i_A[A]"];
            u_B = DF_temp["u_i_B[V]"]+R_s*DF_temp["i_B[A]"];
            u_C = DF_temp["u_i_C[V]"]+R_s*DF_temp["i_C[A]"];
            u_alpha = u_A;
            u_beta = (u_B-u_C)/sq(3);
            U_d = mean(
                u_alpha*cos(gamma)+u_beta*sin(gamma)
                );
            U_q = mean(
                -u_alpha*sin(gamma)+u_beta*cos(gamma)
                );
            U_s = sq(U_d**2 + U_q**2);
            
            Trq = mean(DF_temp["Trq[Nm]"]);
            
            P_out = Trq*n_r_ref/30*pi/1000; # Output power in [W]
            # Check if the desired torque is achieved.
            Trq_diff = (Trq_ref - Trq)/Trq_ref;
            if Trq_diff > 0.05:
                print("***  Warning  ******  desired Torque is not obatined");
            else:
                print("Desired Torque is obatined.");
            P_fe = mean(DF_temp["P_fe[W]"]);
            P_cu = 3*R_s*I_s**2/2;
            P_PM = mean(DF_temp["P_PM[W]"]);
            P_loss = (P_fe+P_cu+P_PM)/1000;
            Eff = P_out/(P_out+P_loss)*100;

            print(
                "*** Finish extracting raw data with n_r = "+key_OP_Pnts+
                " and Trq = "+str(Trq_ref)+"Nm ***"
                );
        else:
            for Var in Dict_OP_Vars:
                globals()[Var] = "NaN";
        for Var, Unit in Dict_OP_Vars.items():
            Dict_OP_Map[Var+Unit][
                "n_r_"+str(idx_nr)
                ].append(globals()[Var]);
        # pdb.set_trace();
for key in Dict_OP_Map:
    Dict_temp = Dict_OP_Map[key];
    DF_temp = pd.DataFrame.from_dict(Dict_temp);
    Dict_OP_Map[key] = DF_temp;
    Name_file = "OP_"+key.replace("[", "_in_").replace("]", "")+"_map.csv"
    Dict_OP_Map[key].to_csv(
        PWD+Name_file, index=False
        );
# %% Close AEDT
# ~~~~~~~~~~
# Close AEDT.

M2D.release_desktop();
# %% Plot
Dict_OP_Vars = {
    "I_d": "[A]", "I_q": "[A]", "I_s": "[A]", 
    "Psi_d": "[Wb]", "Psi_q": "[Wb]", 
    "U_d": "[V]", "U_q": "[V]", "U_s": "[V]", 
    "Trq": "[Nm]", "P_fe": "[W]", "P_cu": "[W]",
    "P_out": "[kW]", "P_loss": "[kW]", "Eff": "[%]"
    };
Dict_OP_titles = {
    "I_d": "d-axix current, $I_{sd}$", "I_q": "q-axix current, $I_{sq}$", 
    "I_s": "Stator current magnitude, $I_{s}$", 
    "Psi_d": "d-axix flux, ${\Psi}_{sd}$", "Psi_q": "q-axix flux, ${\Psi}_{sq}$", 
    "U_d": "d-axix voltage, $U_{sd}$", "U_q": "q-axix voltage, $U_{sq}$", 
    "U_s": "Stator voltage magnitude, $U_{s}$", 
    "Trq": "Output torque", "P_fe": "Coreloss, $P_{fe}$", 
    "P_cu": "Stator resistive loss, $P_{cu}$", 
    "P_out": "Output power", "P_loss": "Total loss", 
    "Eff": "Electrical efficiency"
    };
for Var, Unit in Dict_OP_Vars.items():
    key = Var+Unit;
    Name_file = "OP_"+key.replace("[", "_in_").replace("]", "")+"_map.csv"
    Dict_OP_Map[key] = pd.read_csv(
        PWD+Name_file
        );
DF_Tmax = pd.read_csv(
    PWD+"MaxTorque.csv"
    );
for Var, Unit in Dict_OP_Vars.items():
    if Var != "Trq":
        # x-axis
        n_r = Dict_OP_Map[Var+Unit].iloc[0, 1:];
        # y-axis
        Trq = Dict_OP_Map[Var+Unit]["Trq_ref[Nm]"].iloc[1:];
        # values
        Map = Dict_OP_Map[Var+Unit].iloc[1:, 1:].apply(
            pd.to_numeric, errors='coerce'
            );
               
        # Create contour plot with different colored lines (no fill)
        plt.figure(figsize=(8, 6), dpi=500);
        
        # Max torque line
        plt.plot(
            DF_Tmax["n_r[rpm]"], DF_Tmax["Trq_max[Nm]"],
            color='black', linewidth=2, linestyle='--'
            );
        
        # Contour lines with color mapping
        if Var == "Eff":
            Level_Eff = [80, 90, 92, 94, 95, 96, 97, 98];
            contour_lines = plt.contour(
                n_r, Trq, Map, levels=Level_Eff, 
                cmap='rainbow'
                );
        else:
            contour_lines = plt.contour(
                n_r, Trq, Map, levels=10, cmap='rainbow'
                );
        # Add contour labels
        if Var == "Eff":
            plt.clabel(
                contour_lines, inline=True, fontsize=10, fmt="%.2f", colors='black',
                levels=Level_Eff
                );
        else:
            plt.clabel(
                contour_lines, inline=True, fontsize=10, fmt="%.2f", colors='black'
                );            
        # Customizations
        plt.title(Dict_OP_titles[Var], fontsize=16);
        plt.xlabel("Speed [rpm]", fontsize=14);
        plt.ylabel("Torque [Nm]", fontsize=14);
        plt.grid(True, linestyle='--', alpha=0.5);
        # Add colorbar to indicate values
        cbar = plt.colorbar(contour_lines);
        cbar.set_label(Unit, fontsize=14);


