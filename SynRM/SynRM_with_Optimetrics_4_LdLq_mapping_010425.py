"""
Maxwell 2D: Syn-RM
---------------------------------------------------
This example shows how you can use PyAEDT to create a Maxwell 2D transient 
analysis for a Syn-RM using PyAEDT.

Meng-Ju Hsieh

Created on Mon Feb 17 16:38:29 2025

"""
# %% Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Perform required imports.

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

import csv
import os
import ansys.aedt.core
import pdb

# %% AEDT parameters
# Set AEDT version
# ~~~~~~~~~~~~~~~~
# Set AEDT version.

aedt_version = "2025.1";              # Version of ANSYS

##########################################################
# Assign current folder
# ~~~~~~~~~~~~~~~~
# Assign current folder.

PWD = (os.getcwd().replace("\\","/"))+"/";

##################################################################
# Initialize Maxwell 2D
# ~~~~~~~~~~~~~~~~~~~~~
# Initialize Maxwell 2D, providing the version, path to the project, and the design
# name and type.

setup_name = "Setup1";
solver = "TransientXY";

project_name = "LCA_SynRM";
design_name = "SynRM";

##################################################################
# %% E-machine parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
l_a = 240;                # Active length, [mm]
N_p = 4;	              # Number of poles
N_pp = N_p/2;             # Number of pole pairs
MechAngle_d_oriented = 0;
n_r = 3000;               # Rotor speed, [rpm]
f_0 = n_r/60*N_pp;        # Supplied frequency
Is_rms = 260;             # RMS current
Is_max = Is_rms*sq(2);    # Maximum current

Mat_core = "M235-35A";    # Material of iron core
Mat_coil = "Copper";      # Material of coils

# Stator
d_so = 200;               # Stator outer diameter, [mm]
d_si = 133.3;             # Stator inner diameter, [mm]
Q_s = 48;                 # Number of stator slots
q_s = Q_s/N_p/3;          # Number of stator slots per pole per phase
SlotType = 3;             # SlotType: 1 to 6
Hs0 = 0.3;                # Stator slot opening height, [mm]
Hs01 = 0;                 # Stator slot closed bridge height, [mm]
Hs1 = 0.3;                # Stator slot wedge height, [mm]
Hs2 = 15.2;                 # Stator slot body height, [mm]
Bs0 = 2.4;                # Stator slot opening width, [mm]
Bs1 = 4.3;                # Stator slot wedge maximum width, [mm]
Bs2 = 6.3;                # Stator slot body bottom width, [mm]
Rs = 0.2;                 # Stator slot body bottom fillet

# Windings
N_coil_layer = 1;         # Number of layers of stator coil
f_slot_fill = 0.45;       # Slot fill factor
EndExt = 5;               # Extended straigt part at the end windings
SpanExt = 18;             # Axial length of end span; 0 for no span
N_coil_pitch = 8;         # Coil pitch measured in slots
N_s = 11;                 # Number of turns per coil
N_pb = 4;                 # Number of paralell branches
l_end_ext_in_mm = 25;     # End-extended part of windings. [mm]
Temp_Coil = 120;          # Temperature of coils

# Rotor
d_ro = 132.3;             # Rotor outter diameter, [mm]
d_ri = 50;                # Rotor inner diameter, i.e. shaft diameter, [mm]
l_g = (d_si - d_ro)/2;    # Length of air gap
PoleType = 2;             # Pole type: 1 to 6
Barriers = 3;             # Barriers per Pole, for PoleType 2, 3 & 4 only
H = 1.15;                 # Bridge thickness, for PoleType 2, 3 & 4 only, [mm]
W = 1;                    # Rib width, for PoleType 2, 3 & 4 only, [mm]
R = 1;                  # Barrier tip fillet radius, for PoleType 2, 3 & 4 only
R0 = 55;                  # Barrier bottom fillet radius, for PoleType 2 & 4 only; R0<0 for PM aided, [mm]
Rb = 28;                  # Barrier bottom location radius, [mm]
Y0 = 6;                   # Yoke bottom thickness, [mm]
B0 = 6;                   # Barrier bottom thickness, for PoleType of 2, 3 & 4 only, [mm]

##################################################################
# Initialize definitions for stator, rotor, and shaft 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize geometry parameter definitions for the stator, rotor, and shaft.
# The naming refers to RMxprt primitives.

geom_params = {
    "l_a": str(l_a)+"mm",
    "d_so": str(d_so)+"mm",
    "d_si": str(d_si)+"mm",
    "d_ro": str(d_ro)+"mm",
    "d_ri": str(d_ri)+"mm",
    "l_g": "(d_si-d_ro)/2 mm",
    "Q_s": str(int(Q_s)),
    "PoleType": str(PoleType),
}

##################################################################
# Initialize definitions for stator windings
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize geometry parameter definitions for the stator windings. The naming
# refers to RMxprt primitives.

wind_params = {
    "N_s": str(int(N_s)),
    "N_pb": str(N_pb)
}

##################################################################
# Initialize definitions for model setup
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize geometry parameter definitions for the model setup.

mod_params = {
    "N_p": str(int(N_p)),
    "mapping_angle": "0.125*4deg",
};

##################################################################
# Initialize definitions for operational machine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize geometry parameter definitions for the operational machine. This
# identifies the operating point for the transient setup.

oper_params = {
    "MechAngle_d_oriented": str(MechAngle_d_oriented)+"deg",
    "I_d":"0A",
    "I_q":"0A",
    "Is_max": "sqrt(I_d^2+I_q^2) A",
    "n_r": str(n_r), # Rotor speed
    "RotorSpeed": "n_r rpm", # Rotor speed, [rpm]
    "f_0": "n_r/60*N_p/2 Hz", # Electrical frequency
    "TimeStep": "1/f_0/360*10",
    "StopTime": "2/f_0",
    "Theta_i": "atan2(I_q, I_d) rad",
};

##################################################################
# Set non-graphical mode
# ~~~~~~~~~~~~~~~~~~~~~~
# Set non-graphical mode. ``"PYAEDT_NON_GRAPHICAL"`` is needed to
# generate documentation only.
# You can set ``non_graphical`` either to ``True`` or ``False``.

non_graphical = False;

##################################################################
# Launch Maxwell 2D
# ~~~~~~~~~~~~~~~~~
# Launch Maxwell 2D and save the project.

M2D = ansys.aedt.core.Maxwell2d(
    project=project_name,
    version=aedt_version,
    design=design_name,
    solution_type=solver,
    new_desktop=True,
    non_graphical=non_graphical
    );

##########################################################
# Create object to access 2D modeler
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create the object ``mod2D`` to access the 2D modeler easily.

mod2D = M2D.modeler;
mod2D.delete();
mod2D.model_units = "mm";

##########################################################
# Define variables from dictionaries
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define design variables from the created dictionaries.

for k, v in geom_params.items():
    M2D[k] = v
for k, v in wind_params.items():
    M2D[k] = v
for k, v in mod_params.items():
    M2D[k] = v
for k, v in oper_params.items():
    M2D[k] = v

##########################################################
# %% Define material properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define material properties.

##########################################################
# Create materials
# ~~~~~~~~~~~~~~~~~~~~~
# Create function to read material properties
import json
def read_material_properties(Name_Material):
    file = open(PWD+Name_Material+".json");
    Dict_temp = json.load(file);
    Dict_OP = Dict_temp["materials"][Name_Material];
    file.close();
    return Dict_OP

# Create the material for iron core.

M2D.materials.import_materials_from_file(
    PWD+Mat_core+".json"
    );

# Create the material for windings.

M2D.materials.import_materials_from_file(
    PWD+Mat_coil+".json"
    );

# Calculate the Resistivity of coil for given material and temperature
Dict_Cndctr_S = read_material_properties(Mat_coil);
Conductivity_20C = float(Dict_Cndctr_S["conductivity"]);
Resistivity_20C = 1/Conductivity_20C;
Coff_Termal = float(Dict_Cndctr_S["resistivity_thermal_coefficient"]);
Rstvty_Coil = Resistivity_20C*(1+Coff_Termal*(Temp_Coil-20));
Cnductvty_Coil = 1/Rstvty_Coil;

Mat_coil = Mat_coil+"_"+str(int(Temp_Coil))+"C";
M2D_Mat_coil = M2D.materials.add_material(Mat_coil);
M2D_Mat_coil.update();
M2D_Mat_coil.conductivity = str(Cnductvty_Coil);
M2D_Mat_coil.permeability = "1"
##########################################################
# %% Calculate stator resistance
# ================================================================
# Create "calculate_slot_size" function
# ~~~~~~~~~~~~
# Create a function to calculate the size of slot.

def calculate_slot_size(SlotType, SlotDimensions, Casted=False):
    """
    Calculate the area of cross-section and the depth of slot.
    Parameters
    ----------
    SlotType : int
        Slot type defined in AEDT Maxwell.
    SlotDimensions : list
        A list of 7 dimensions of slot.
        SlotDimensions = [ Bs0, Bs1, Bs2, Hs0, Hs1, Hs2, Rs ]
    Casted : bool
        If True, the slot head area will be included. 
    Returns
    -------
    d_slot_in_mm : float
        Depth of slot.
    a_slot_in_mm : TYPE
        Area of cross-section of slot.
    """
    [ Bs0, Bs1, Bs2, Hs0, Hs1, Hs2, Rs ] = SlotDimensions;
    match SlotType:
        case 1:
            d_slot_in_mm = (Hs0+Bs1/2+Hs2+Bs2/2);
            a_slot_in_mm = (pi*(Bs1/2)**2)/2+(Bs1+Bs2)*Hs2/2+(
                pi*(Bs2/2)**2)/2;
        case 2:
            d_slot_in_mm = Hs0+Hs1+Hs2+Bs2/2;
            a_slot_in_mm = (Bs0+Bs0)*Hs1/2+(
                Bs1+Bs2)*Hs2/2+(pi*(Bs2/2)**2)/2;
        case 3:
            d_slot_in_mm = (
                Hs0+Hs1+Hs2+Rs);
            a_slot_in_mm = (Bs0+Bs0)*Hs1/2+(Bs1+Bs2)*Hs2/2+(
                pi*Rs**2)/2+(Bs2-Rs*2)*Rs;
        case 4:
            d_slot_in_mm = Hs0+Hs1+Hs2+Rs;
            a_slot_in_mm = (pi*Hs1**2)/2+(Bs1-2*Hs1)*Hs1+(
                Bs1+Bs2)*Hs2/2+(pi*Rs**2)/2+(Bs2-Rs*2)*Rs;
        case 5:
            d_slot_in_mm = Hs0+Hs1+Hs2;
            a_slot_in_mm = (Bs2*Hs1)-(Bs2-Bs0)*Hs1/2+(
                Bs1-Bs2)*Hs1/2+(Hs2*Bs2);
        case 6:
            d_slot_in_mm = Hs0+Hs1+Hs2;
            a_slot_in_mm = (Bs1+Bs2)*Hs1/2+Hs2*Bs2;
    if Casted:
        a_slot_in_mm = a_slot_in_mm+Hs0*Bs0;
    return d_slot_in_mm, a_slot_in_mm

# ================================================================
# Calculate the size of slot
# ~~~~~~~~~~~~
# Calculate the size of slot.

SlotDimensions = [ Bs0, Bs1, Bs2, Hs0, Hs1, Hs2, Rs ]; # Create a list of parameters of stator slot
d_slot_in_mm, a_slot_in_mm = calculate_slot_size(
    SlotType=SlotType, SlotDimensions=SlotDimensions
    );

a_coil = a_slot_in_mm*1e-6*f_slot_fill/N_coil_layer;

r_slotcenter = d_si/2+(d_slot_in_mm+Hs0)/2; # in [mm]
l_wp = 2*pi*r_slotcenter*N_coil_pitch/Q_s;
l_end_coil = 2*(1.2*l_wp+2*l_end_ext_in_mm)*1e-3; # in [m], length of end-coil
l_ew = l_end_coil*N_p*q_s*N_s/(2*N_pb**2); # in [m], total end-winding length
l_aw = 2*l_a*1e-3*N_p*q_s*N_s/(2*N_pb**2); # in [m], total active winding length
l_tw = l_ew+l_aw; # Total winding length in [m]
l_coil = l_a*l_tw/l_aw; # Single coil length in [m]
a_turn = a_coil/N_s; # Cross-sectional area of one turn in [m^2]

R_s_ew = Rstvty_Coil*l_ew/a_turn;
R_s_aw = Rstvty_Coil*l_aw/a_turn;
R_s = Rstvty_Coil*l_tw/a_turn; # Equivelent stator resistance
Dict_DesignParms = {
    "Parameter": [
        "Project",
        "Design",
        "N_s",
        "N_pp",
        "Active length[mm]",
        "R_s[Ohm]", 
        "R_s_eW[Ohm]",
        "Temp_coil[C]"
        ],
    "Value": [
        project_name,
        design_name,
        N_s,
        N_pp,
        l_a,
        R_s,
        R_s_ew,
        Temp_Coil
        ]
    };
DF_DesignParms = pd.DataFrame.from_dict(Dict_DesignParms);
DF_DesignParms.to_csv(
    PWD+"Design_parameters.csv"
    );

v_ew = a_coil*l_end_coil*Q_s*N_coil_layer; # Total end-winding volume in [m^3]
v_aw = a_coil*l_coil*Q_s*N_coil_layer; # Total active winding volume in [m^3]
# %% Create geometry and assign properties
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
##################################################################
# Create geometry and assign properties for stator core
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Create the geometry for the stator core. It is created via
#    the RMxprt user-defined primitive. A list of lists is
#    created with the proper UDP parameters.
# 2. Assign properties to the stator core. The following code 
#    assigns the material, name, color, transparency, and 
#    ``solve_inside`` properties.

udp_par_list_stator = [
    ["DiaGap", "d_si"], 
    ["DiaYoke", "d_so"], 
    ["Length", "0mm"],
    ["Skew", "0deg"], 
    ["Slots", "Q_s"], 
    ["SlotType", str(int(SlotType))],
    ["Hs0", str(Hs0)+"mm"], 
    ["Hs01", str(Hs01)+"mm"], 
    ["Hs1", str(Hs1)+"mm"],
    ["Hs2", str(Hs2)+"mm"],
    ["Bs0", str(Bs0)+"mm"], 
    ["Bs1", str(Bs1)+"mm"], 
    ["Bs2", str(Bs2)+"mm"], 
    ["Rs", str(Rs)+"mm"],
    ["FilletType", "0"], 
    ["HalfSlot", "0"], 
    ["VentHoles", "0"], 
    ["HoleDiaIn", "0mm"],
    ["HoleDiaOut", "0mm"],
    ["HoleLocIn", "0mm"],
    ["HoleLocOut", "0mm"],
    ["VentDucts", "0"],
    ["DuctWidth", "0mm"],
    ["DuctPitch", "0mm"],
    ["SegAngle", "0deg"],
    ["LenRegion", "0mm"],
    ["InfoCore", "0"]
    ];

stator_id = mod2D.create_udp(
    dll="RMxprt/VentSlotCore.dll", 
    parameters=udp_par_list_stator, 
    library='syslib',
    name='Stator'
    );  # name not taken

M2D.assign_material(
    assignment=stator_id, material=Mat_core
    );
# stator_id.name = "Stator";
stator_id.color = (192, 192, 192)  # rgb
stator_id.transparency = 0;
stator_id.solve_inside = True  # to be reassigned: M2D.assign material puts False if not dielectric

##################################################################
# Create geometry and assign properties for rotor core
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Create the geometry for the rotor core. It is created via
#    the RMxprt user-defined primitive. A list of lists is
#    created with the proper UDP parameters.
# 2. Assign properties to the rotor core. The following code 
#    assigns the material, name, color, transparency, and 
#    ``solve_inside`` properties.

udp_par_list_rotor = [
    ["DiaGap", "d_ro"], 
    ["DiaYoke", "d_ri"], 
    ["Length", "0mm"],
    ["Poles", "N_p"], 
    ["PoleType", "PoleType"], # Only works with Ansys/Maxwell parameter
    ["Barriers", str(Barriers)+"mm"],
    ["H", str(H)+"mm"], 
    ["W", str(W)+"mm"], 
    ["R", str(R)+"mm"],
    ["R0", str(R0)+"mm"],
    ["Rb", str(Rb)+"mm"], 
    ["Y0", str(Y0)+"mm"], 
    ["B0", str(B0)+"mm"], 
    ["LenRegion", "0mm"],
    ["InfoCore", "0"]
    ];

rotor_id = mod2D.create_udp(
    dll="RMxprt/SynRMCore.dll", 
    parameters=udp_par_list_rotor, 
    library='syslib',
    name="Rotor"
    );
    
M2D.assign_material(
    assignment=rotor_id, material=Mat_core
    );
# rotor_id.name = "Rotor";
rotor_id.color = (192, 192, 192);  # rgb
rotor_id.transparency = 0;
rotor_id.solve_inside = True;  # to be reassigned: M2D.assign material puts False if not dielectric

# ================================================================
# Create Rotor1, Rotor2, Rotor3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create 3 more layers between magnets and air gap to refine mesh.
rotor_id.clone();
rotor1_id = mod2D.get_objects_w_string(
    "Rotor1", case_sensitive=True
    );
mod2D.create_circle(
    origin=[0, 0, 0], radius=str(d_ro/2-H), name="Rotor1_tool"
    );
mod2D.subtract("Rotor1", "Rotor1_tool", keep_originals=True);
mod2D.intersect(["Rotor", "Rotor1_tool"]);

mod2D.clone("Rotor1");
mod2D.create_circle(
    origin=[0, 0, 0], radius=str((d_ro - 2*H/3)/2), 
    name="Rotor2_tool"
    );
mod2D.subtract("Rotor1", "Rotor2_tool", keep_originals=True);
mod2D.intersect(["Rotor2", "Rotor2_tool"]);

mod2D.clone("Rotor2");
mod2D.create_circle(
    origin=[0, 0, 0], radius=str((d_ro - 4*H/3)/2), 
    name="Rotor3_tool"
    );
mod2D.subtract("Rotor2", "Rotor3_tool", keep_originals=True);
mod2D.intersect(["Rotor3", "Rotor3_tool"]);

rotor_objs_list = mod2D.get_objects_w_string(
    "Rotor", case_sensitive=True
    );
rotor_id_list = [];
for name in rotor_objs_list:
    rotor_id_list.append(
        mod2D.get_object_from_name(name)
        );

##################################################################
# Create coils
# ~~~~~~~~~~~~
# Create the coils.
# ================================================================
# Create the geometry of coils
# ~~~~~~~~~~~~
# Create the geometry of coils.
d_coil = round(d_slot_in_mm*0.8/N_coil_layer, 2); # Depth of single coil
a_coil = a_slot_in_mm*f_slot_fill/N_coil_layer; # Area of single coil
w_coil = round(a_coil/d_coil, 2);
l_g = (d_si-d_ro)/2; # Length of air gap

r_coil = d_ro/2+l_g+(Hs0+Hs1)*2; # Radius of coil head

coil_id = mod2D.create_rectangle(
    origin=[str(r_coil), str(-w_coil/2), 0],
    sizes=[str(d_coil), str(w_coil), 0],
    name='Coil1', material=Mat_coil
    );
coil_id.color = (255, 128, 0);

if N_coil_layer == 2:
    coil_id.clone();
    coil2_id = mod2D.get_object_from_name("Coil2");
    mod2D.move(
        assignment="Coil2",
        vector=[d_coil*1.1, 0, 0]
        );
Tht_slot = 360/Q_s;
mod2D.rotate(
    assignment=coil_id, axis="Z", angle=str(Tht_slot/2)+"deg"
    );
coil_id.duplicate_around_axis(
    axis="Z", angle=str(Tht_slot), clones="Q_s",
    create_new_objects=True
    );

if N_coil_layer == 2:
    mod2D.rotate(
        assignment=coil2_id, axis="Z", 
        angle=str(Tht_slot*(N_coil_pitch+1.5))+"deg"
        );
    coil2_id.duplicate_around_axis(
        axis="Z", angle=str(Tht_slot), clones="Q_s",
        create_new_objects=True
        );    

coils_id_list = [];
for x in range(int(Q_s)):
    if x == 0:
        str_idx = "";
    else:
        str_idx = "_"+str(int(x));
    coil_id = mod2D.get_object_from_name(assignment="Coil1"+str_idx);
    coils_id_list.append(coil_id);
    idx_p = int(x/(Q_s/N_p));
    match int(idx_p%2):
        case 0:
            str_polar_1 = "_pos";
            str_polar_2 = "_neg";
        case 1:
            str_polar_1 = "_neg";
            str_polar_2 = "_pos";
    match int(x/q_s%3):
        case 0:
            coil_id.name = "B"+str(int(x%q_s+idx_p*q_s+1))+str_polar_2;
            coil_id.color = (0, 0, 255);
            coil_id.transparency = 0;
        case 1:
            coil_id.name = "A"+str(int(x%q_s+idx_p*q_s+1))+str_polar_1;
            coil_id.color = (255, 0, 0);
            coil_id.transparency = 0;
        case 2:
            coil_id.name = "C"+str(int(x%q_s+idx_p*q_s+1))+str_polar_2;            
            coil_id.color = (0, 128, 0);
            coil_id.transparency = 0;
    if N_coil_layer == 2:
        N_x = Q_s/N_p - N_coil_pitch;
        coil_id = mod2D.get_object_from_name(assignment="Coil2"+str_idx);
        coils_id_list.append(coil_id);
        match int(x/q_s%3):
            case 0:
                coil_id.name = "B"+str(int(x%q_s+idx_p*q_s+1))+str_polar_1;
                coil_id.color = (0, 0, 255);
                coil_id.transparency = 0;
            case 1:
                coil_id.name = "A"+str(int(x%q_s+idx_p*q_s+1))+str_polar_2;
                coil_id.color = (255, 0, 0);
                coil_id.transparency = 0;
            case 2:
                coil_id.name = "C"+str(int(x%q_s+idx_p*q_s+1))+str_polar_1;            
                coil_id.color = (0, 128, 0);
                coil_id.transparency = 0;
    # pdb.set_trace();

##################################################################
# Create shaft and region
# ~~~~~~~~~~~~~~~~~~~~~~~
# Create the shaft and region.

region_id = mod2D.create_circle(
    origin=[0, 0, 0], radius="d_so/2",
    num_sides="0.25", is_covered=True, name="Region"
    );
shaft_id = mod2D.create_circle(
    origin=[0, 0, 0], radius="d_ri/2",
    num_sides="0.25", is_covered=True, name="Shaft"
    );
##################################################################
# Create bands
# ~~~~~~~~~~~~
# Create the inner band, band, and outer band.

band_id = mod2D.create_circle(
    origin=[0, 0, 0], radius=str((d_si - l_g)/2),
    num_sides="mapping_angle", is_covered=True, name="Band"
    );
bandOUT_id = mod2D.create_circle(
    origin=[0, 0, 0], radius=str(d_si/2),
    num_sides="mapping_angle", is_covered=True, name="Outer_Band"
    );

##################################################################
# Create list of vacuum objects
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a list of vacuum objects and assign color.

vacuum_obj_id_list = [shaft_id, region_id, band_id, bandOUT_id];  # put shaft first
for item in vacuum_obj_id_list:
    item.color = (128, 255, 255);

# %% Create section of machine
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a section of the machine. This allows you to take
# advantage of symmetries.

object_list = [
    stator_id
    ] + rotor_id_list+ coils_id_list + vacuum_obj_id_list;

mod2D.create_coordinate_system(
    origin=[0, 0, 0], reference_cs="Global", name="Section",
    mode="axis",
    x_pointing=["cos(360deg/N_p)", "sin(360deg/N_p)", 0],
    y_pointing=["-sin(360deg/N_p)", "cos(360deg/N_p)", 0]
    );

mod2D.set_working_coordinate_system("Section");
mod2D.split(object_list, "ZX", sides="NegativeOnly");
coils_id_list = mod2D.get_objects_by_material(Mat_coil);
object_list = [
    stator_id
    ] + rotor_id_list+ coils_id_list + vacuum_obj_id_list;
mod2D.set_working_coordinate_system("Global");
mod2D.split(object_list, "ZX", sides="PositiveOnly");

# %% Assign motion setup to object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Assign a motion setup to a ``Band`` object.

M2D.assign_rotate_motion(
    assignment="Band", 
    coordinate_system="Global", 
    axis="Z", 
    positive_movement=True, 
    start_position="MechAngle_d_oriented", 
    angular_velocity="RotorSpeed"
    );

# %% Create boundary conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create independent and dependent boundary conditions.
# Edges for assignment are picked by position.
# The points for edge picking are in the airgap.

pos_1 = (d_si - l_g)/4;
id_bc_1 = mod2D.get_edgeid_from_position(
    position=[pos_1, 0, 0], assignment="Region"
    );
id_bc_2 = mod2D.get_edgeid_from_position(
    position=[
        pos_1*cos(2*pi/N_p), 
        pos_1*sin(2*pi/N_p), 0], 
    assignment="Region"
    );
M2D.assign_master_slave(
    independent=id_bc_1, 
    dependent=id_bc_2, 
    reverse_master=True, 
    reverse_slave=False,
    same_as_master=False, 
    boundary="Matching"
    );

#==============================================================================
# Assign vector potential
# ~~~~~~~~~~~~~~~~~~~~~~~
# Assign a vector potential of ``0`` to the second position.

pos_2 = (d_so/2);
id_bc_az = mod2D.get_edgeid_from_position(
    position=[
        pos_2*cos(2*pi/N_p/2), 
        pos_2*sin(2*pi/N_p/2), 
        0
        ],
    assignment="Region");
M2D.assign_vector_potential(
    id_bc_az, vector_value=0, boundary="VectorPotentialZero"
    );

# %% Create excitations
# ~~~~~~~~~~~~~~~~~~
# Create excitations.

#==============================================================================
# Define phase currents
# ~~~~~~~~~~~~~~~~~~
# Define phase currents for the windings

PhA_current = "Is_max*cos(2*pi*f_0*time+Theta_i)";
PhB_current = "Is_max*cos(2*pi*f_0*time-2*pi/3+Theta_i)";
PhC_current = "Is_max*cos(2*pi*f_0*time+2*pi/3+Theta_i)";

#==============================================================================
# Define windings in phase A
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define windings in phase A.
coils_id_list = mod2D.get_objects_by_material(Mat_coil);
Dict_phase = defaultdict(list);
Dict_phase["Phase list"] = [ "A", "B", "C" ];
Dict_phase["Current list"] = [ PhA_current, PhB_current, PhC_current ];
for coil_id in coils_id_list:
    if "_pos" in coil_id.name:       
        M2D.assign_coil(
            assignment=[coil_id.name], conductors_number=N_s, 
            polarity="Positive", name=coil_id.name
            );
    elif "_neg" in coil_id.name:       
        M2D.assign_coil(
            assignment=[coil_id.name], conductors_number=N_s, 
            polarity="Negative", name=coil_id.name
            );
    for ph_x in Dict_phase["Phase list"]:
        if ph_x in coil_id.name:
            Dict_phase["Phase_"+ph_x+" coil list"].append(coil_id.name);
for idx_ph, ph_x in enumerate(Dict_phase["Phase list"]):
    M2D.assign_winding(
        winding_type="Current", 
        resistance = R_s_ew,
        is_solid=False, 
        current=Dict_phase["Current list"][idx_ph],
        parallel_branches=N_pb,
        name="Phase_"+ph_x
        );
    M2D.add_winding_coils(
        assignment="Phase_"+ph_x, 
        coils=Dict_phase["Phase_"+ph_x+" coil list"]
        );

# %% Create mesh operations
# ~~~~~~~~~~~~~~~~~~~~~~
# Create the mesh operations.

M2D.mesh.assign_length_mesh(
    coils_id_list, 
    inside_selection=True, 
    maximum_length=3, 
    maximum_elements=None, 
    name="coils"
    );
M2D.mesh.assign_length_mesh(
    stator_id, 
    inside_selection=True, 
    maximum_length=3, 
    maximum_elements=None, 
    name="stator"
    );
M2D.mesh.assign_length_mesh(
    rotor_id, 
    inside_selection=True, 
    maximum_length=3, 
    maximum_elements=None, 
    name="rotor"
    );

# %% Other essential setups
# ~~~~~~~~~~~~~~~~~~~~
# Set "eddy effects", "core loss", "transient inductance", "model depth", 
# "symmetry factor".

#==============================================================================
# Turn on eddy effects
# ~~~~~~~~~~~~~~~~~~~~
# Turn on eddy effects.

# M2D.eddy_effects_on(
#     eddy_effects_list,
#     activate_eddy_effects=True, 
#     activate_displacement_current=False
#     );

#==============================================================================
# Turn on core loss
# ~~~~~~~~~~~~~~~~~
# Turn on core loss.

core_loss_ids_list = mod2D.get_objects_by_material(Mat_core);
core_loss_names_list = [];
for item_id in core_loss_ids_list:
    core_loss_names_list.append(
        item_id.name
        );
M2D.set_core_losses(core_loss_names_list);

#==============================================================================
# Compute transient inductance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Compute the transient inductance.

M2D.change_inductance_computation(
    compute_transient_inductance=True, 
    incremental_matrix=False
    );

#==============================================================================
# Set model depth
# ~~~~~~~~~~~~~~~
# Set the model depth.

M2D.model_depth = "l_a";

#==============================================================================
# Set symmetry factor
# ~~~~~~~~~~~~~~~~~~~
# Set the symmetry factor.

M2D.change_symmetry_multiplier("N_p");

# %% Create setup and validate
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# Create the setup and validate it.

setup = M2D.create_setup(name=setup_name);
setup.props["StopTime"] = "StopTime";
setup.props["TimeStep"] = "TimeStep";
setup.props["SaveFieldsType"] = "Every N Steps";
setup.props["N Steps"] = "10";
setup.props["Steps From"] = "StopTime/2";
setup.props["Steps To"] = "StopTime";
setup.props["NonlinearSolverResidual"] = "1e-5";
setup.props["OutputPerObjectCoreLoss"] = True;
setup.props["OutputPerObjectSolidLoss"] = True;
setup.props["OutputError"] = True;
setup.update();
M2D.validate_simple();

# %% setting up Optimetrics

# Define the range
Id_start=0;
Id_stop=np.ceil(Is_max/100)*100;
Id_step=Id_stop/5;

Iq_start=0;
Iq_stop=Id_stop;
Iq_step=Id_step;

Arr_Id = np.arange(Id_start, Id_stop+Id_step*.5, Id_step);
Arr_Id[0] = 5;
List_Id = list(Arr_Id);
Arr_Iq = np.arange(Iq_start, Iq_stop+Iq_step*.5, Iq_step);
Arr_Iq[0] = 5;
List_Iq = list(Arr_Iq);

# Generate a grid using np.meshgrid
Arr_Id, Arr_Iq = np.meshgrid(
    Arr_Id, Arr_Iq
    );

# Flatten the arrays to get all combinations
Series_Id = Arr_Id.flatten();
Series_Iq = Arr_Iq.flatten();

DF_I_dq = pd.DataFrame();
DF_I_dq["*"] = np.arange(1, len(Series_Id)+1, 1);
DF_I_dq["I_d"] = Series_Id;
DF_I_dq["I_q"] = Series_Iq;
DF_I_dq["I_d"] = DF_I_dq["I_d"].astype(str)+"A";
DF_I_dq["I_q"] = DF_I_dq["I_q"].astype(str)+"A";

DF_I_dq.to_csv(
    PWD+"I_dq.csv", index=False
    );

M2D.parametrics.add_from_file(
    PWD+"I_dq.csv", 
    name="I_dq_sweep"
    );

##########################################################
# Analyze and save project
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Analyze and save the project.

M2D.save_project();
M2D.analyze_setup(setup_name);

M2D.analyze_setup(
    "I_dq_sweep", cores=12, tasks=12, 
    use_auto_settings=False
    );


# %% Initialize definitions for output variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize the definitions for the output variables.
# These will be used later to generate reports.

Dict_OP_vars = {
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
};
List_OP_Vars = list(Dict_OP_vars.keys());
##########################################################
# Create output variables for postprocessing
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create output variables for postprocessing.

for key, value in Dict_OP_vars.items():
    M2D.create_output_variable(key, value[0]);

##################################################################
# Initialize definition for postprocessing multiplots
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize the definition for postprocessing multiplots.

post_params_multiplot = {  # reports
    ("u_A", "u_B", "u_C", "u_i_A", "u_i_B", "u_i_C"): "PhaseVoltages",
    ("CoreLoss", "SolidLoss"): "Losses",
    ("i_A", "i_B", "i_C"): "PhaseCurrents",
    ("psi_A", "psi_B", "psi_C"): "PhaseFluxes",
    ("CoreLoss", "CoreLoss(Stator)", "CoreLoss(Rotor)"): "CoreLosses",
    ("EddyCurrentLoss", "EddyCurrentLoss(Stator)", "EddyCurrentLoss(Rotor)"): "EddyCurrentLosses (Core)",
    ("ExcessLoss", "ExcessLoss(Stator)", "ExcessLoss(Rotor)"): "ExcessLosses (Core)",
    ("HysteresisLoss", "HysteresisLoss(Stator)", "HysteresisLoss(Rotor)"): "HysteresisLosses (Core)",
};
# %% Export rawdata from Ansys
Dict_vari = M2D.available_variations.nominal_w_values_dict;
Dict_Mxwl = defaultdict(list);
for idx in DF_I_dq["*"]:    
    I_d_in = float(DF_I_dq["I_d"][idx-1].replace("A", ""));    
    I_q_in = float(DF_I_dq["I_q"][idx-1].replace("A", ""));
    Dict_vari["I_d"] = DF_I_dq["I_d"][idx-1];
    Dict_vari["I_q"] = DF_I_dq["I_q"][idx-1];
    # Create a dictionary to contain raw data from Ansys/Maxwell
    Dict_temp = defaultdict();
    # Extract raw data
    Mxwl_rawdata = M2D.post.get_solution_data(
        expressions=List_OP_Vars, 
        variations=Dict_vari,
        primary_sweep_variable="Time"
        );
    for Var, Unit in Dict_OP_vars.items():
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
    Dict_Mxwl["I_d[A]"].append(I_d);
    Dict_Mxwl["I_q[A]"].append(I_q);    
    
    psi_alpha = DF_temp["psi_A[Wb]"];
    psi_beta = (DF_temp["psi_B[Wb]"]-DF_temp["psi_C[Wb]"])/sq(3);
    Dict_Mxwl["Psi_d[Wb]"].append(
        mean(
            psi_alpha*cos(gamma)+psi_beta*sin(gamma)
            )
        );
    Dict_Mxwl["Psi_q[Wb]"].append(
        mean(
            -psi_alpha*sin(gamma)+psi_beta*cos(gamma)
            )
        );    
    u_A = DF_temp["u_i_A[V]"]+R_s_ew*DF_temp["i_A[A]"];
    u_B = DF_temp["u_i_B[V]"]+R_s_ew*DF_temp["i_B[A]"];
    u_C = DF_temp["u_i_C[V]"]+R_s_ew*DF_temp["i_C[A]"];
    u_alpha = u_A;
    u_beta = (u_B-u_C)/sq(3);
    Dict_Mxwl["U_d[V]"].append(
        mean(
            u_alpha*cos(gamma)+u_beta*sin(gamma)
            )
        );
    Dict_Mxwl["U_q[V]"].append(
        mean(
            -u_alpha*sin(gamma)+u_beta*cos(gamma)
            )
        );
    
    Dict_Mxwl["Trq[Nm]"].append(mean(DF_temp["Trq[Nm]"]));
    Dict_Mxwl["P_fe[W]"].append(mean(DF_temp["P_fe[W]"]));
    print(
        "*** Finish extracting raw data with I_d = "+DF_I_dq[
            "I_d"][idx-1]+" and I_q = "+DF_I_dq["I_q"][idx-1]+" ***"
        );
    # pdb.set_trace();
DF_Mxwl = pd.DataFrame.from_dict(Dict_Mxwl);

# pdb.set_trace();

# %% Map L_d & L_q
Dict_Map = defaultdict();

DF_Mxwl["L_d[mH]"] = DF_Mxwl["Psi_d[Wb]"]/DF_Mxwl["I_d[A]"]*1e3;
DF_Mxwl["L_q[mH]"] = DF_Mxwl["Psi_q[Wb]"]/DF_Mxwl["I_q[A]"]*1e3;
for Var in DF_Mxwl.keys():
    # if Var != "I_d[A]" and Var != "I_q[A]":
    Dict_temp = defaultdict(list);
    Dict_temp["I_q[A]"] = [0] + List_Iq;
    for idx_Id, I_d in enumerate(List_Id):
        key = "Id_"+str(idx_Id+1);
        idcs = DF_Mxwl[DF_Mxwl["I_d[A]"] == I_d].index;
        Dict_temp[key] = [I_d] + list(DF_Mxwl[Var][idcs]);
    Dict_Map[Var] = pd.DataFrame.from_dict(Dict_temp);
    file_name = Var.replace("[","_in_").replace("]","")+"_map.csv";
    Dict_Map[Var].to_csv(PWD+file_name, index=False);
    # `index=False` avoids saving row indices
# %% Close AEDT
# ~~~~~~~~~~
# Close AEDT.
M2D.release_desktop();
# %% Plot contours of L_d-L_q
# Define the arc parameters
theta = np.linspace(0, np.pi/2, 100);  # 90-degree arc (0 to π/2 radians)
radius = Is_max;  # Radius of the arc
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
        plt.xlim(0,400);
        plt.ylim(0,400);
        plt.title("$"+Var+"$");
        plt.xlabel("$I_d$[A]", fontsize=14);
        plt.ylabel("$I_q$[A]", fontsize=14);
        plt.grid(True, linestyle='--', alpha=0.5);
        # Add colorbar to indicate values
        plt.colorbar(contour_lines);