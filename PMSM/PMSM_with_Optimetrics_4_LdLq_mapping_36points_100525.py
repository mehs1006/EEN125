"""
Maxwell 2D: Prius2D 2-layer coil & 2-layer PM
---------------------------------------------------
This example shows how you can use PyAEDT to create a Maxwell 2D transient 
analysis for PMSM using PyAEDT.

Torbjörn Thiringer
Meng-Ju Hsieh

Created on Thu Sep 12 16:38:43 2024

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

# %% ##########################################################
# Set AEDT version
# ~~~~~~~~~~~~~~~~
# Set AEDT version.

aedt_version = "2025.1";              # Version of ANSYS

##########################################################
# Assign current folder
# ~~~~~~~~~~~~~~~~♣
# Assign current folder.

PWD = (os.getcwd().replace("\\","/"))+"/";

##################################################################
# Initialize Maxwell 2D
# ~~~~~~~~~~~~~~~~~~~~~
# Initialize Maxwell 2D, providing the version, path to the project, and the design
# name and type.

setup_name = "Setup1";
solver = "TransientXY";

project_name = "LCA_REE_PMSM";
design_name = "Ref_PMSM";

##################################################################
# %% Project parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
l_a = 127;                # Active length, [mm]
N_p = 8;	              # Number of poles
N_pp = N_p/2;             # Number of pole pairs

# Stator
d_so = 200;             # Stator outer diameter, [mm]
d_si = 135;             # Stator inner diameter, [mm]
Q_s = 48;                 # Number of stator slots
q_s = Q_s/N_p/3;          # Number of stator slots per pole per phase
SlotType = 2;             # SlotType: 1 to 6
Hs0 = 1;               # Stator slot opening height, [mm]
Hs01 = 0;                 # Stator slot closed bridge height, [mm]
Hs1 = 0;                  # Stator slot wedge height, [mm]
Hs2 = 17;               # Stator slot body height, [mm]
Bs0 = 2;               # Stator slot opening width, [mm]
Bs1 = 4;                  # Stator slot wedge maximum width, [mm]
Bs2 = 6;                  # Stator slot body bottom width, [mm]
Rs = 3;                   # Stator slot body bottom fillet

# Windings
N_coil_layer = 2;         # Number of layers of stator coil
f_slot_fill = 0.45;       # Slot fill factor
EndExt = 5;               # Extended straigt part at the end windings
SpanExt = 18;             # Axial length of end span; 0 for no span
N_coil_pitch = 5;         # Coil pitch measured in slots
N_s = 7;                  # Number of turns per coil
N_pb = 4;                 # Number of paralell branches
l_end_ext_in_mm = 25;     # End-extended part of windings. [mm]
Temp_Coil = 120;          # Temperature of coils

# Rotor
d_ro = 133.5;             # Rotor outter diameter, [mm]
d_ri = 80;                # Rotor inner diameter, i.e. shaft diameter, [mm]
PoleType = 3;             # Pole type: 1 to 6
l_g = (d_si - d_ro)/2;    # Length of air gap

# 1st layer PM
D1 = 130.5;              # Limited diameter of PM ducts
O1 = 2;                   # Bottom width for separate or flat-bottom duct, or duct opening width for type 1 (not for type 2)
O2 = 14;                # Distance from duct bottom to shaft surface, or Gmax-Gmin for type 1
B1 = 4;                 # Duct thickness
Rib = 6.69;                 # Rib width
HRib = 3;                 # Rib height (for types 1 & 3~5)
DminMag = 3;            # Minimum distance between side magnets (for types 3~5)
ThickMag = 4.55;          # Magnet thickness
WidthMag = 36;            # Total width of all magnet per pole

# Number of PM layers
N_pm_layer = 1;

# 2nd layer PM
D1_2 = 135;              # Limited diameter of PM ducts
O1_2 = 3;                # Bottom width for separate or flat-bottom duct, or duct opening width for type 1 (not for type 2)
O2_2 = 10;               # Distance from duct bottom to shaft surface, or Gmax-Gmin for type 1
B1_2 = 4.7;              # Duct thickness
Rib_2 = 14;              # Rib width
HRib_2 = 3;              # Rib height (for types 1 & 3~5)
DminMag_2 = 4.5;         # Minimum distance between side magnets (for types 3~5)
ThickMag_2 = 6.48;       # Magnet thickness
WidthMag_2 = 32;         # Total width of all magnet per pole

# Other parameters
n_r = 4000;               # Rotor speed
MechAngle_d_oriented = 0; # Initial angle for d-oriented
if N_coil_layer == 2 and N_coil_pitch != Q_s/N_p:
    if N_coil_pitch < Q_s/N_p:
        MechAngle_d_oriented = 360 - 360/Q_s/2*(
            Q_s/N_p-N_coil_pitch
            );
    else:
        MechAngle_d_oriented = 360/Q_s/2*(
            Q_s/N_p-N_coil_pitch
            );
f_0 = n_r/60*N_pp;                # Supplied frequency
Is_rms = 260;             # RMS current
Is_max = Is_rms*sq(2);     # Maximum current
Mat_core = "NO30";        # Material of iron core
Mat_coil = "Copper";      # Material of coils, Copper or Aluminum
Mat_magnet = "NMF-37F_70C";   # Material of magnets
N_step_per_period = 36;     # Number of steps per electrical period
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
    "I_d":"-160A",
    "I_q":"320A",
    "Is_max": "sqrt(I_d^2+I_q^2) A",
    "n_r": str(n_r), # Rotor speed
    "RotorSpeed": "n_r rpm", # Rotor speed, [rpm]
    "f_0": "n_r/60*N_p/2 Hz", # Electrical frequency
    "TimeStep": "1/f_0/"+str(N_step_per_period),
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

# Create the material for windings.
M2D.materials.import_materials_from_file(
    PWD+Mat_magnet+".json"
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
l_ew = l_end_coil*N_p*q_s*N_s/(2*N_pb**2)*N_coil_layer; # in [m], total end-winding length
l_aw = 2*l_a*1e-3*N_p*q_s*N_s/(2*N_pb**2)*N_coil_layer; # in [m], total active winding length
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

mod2D.fit_all(); # Fit the window to all models

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
    ["PoleType", int(PoleType)], # doesn't work
    ["D1", str(D1)+"mm"],
    ["O1", str(O1)+"mm"], 
    ["O2", str(O2)+"mm"], 
    ["B1", str(B1)+"mm"],
    ["Rib", str(Rib)+"mm"],
    ["HRib", str(HRib)+"mm"], 
    ["DminMag", str(DminMag)+"mm"], 
    ["ThickMag", str(ThickMag)+"mm"], 
    ["WidthMag", str(WidthMag)+"mm"],
    ["LenRegion", "0mm"],
    ["InfoCore", "0"]
    ];

rotor_id = mod2D.create_udp(
    dll="RMxprt/IPMCore.dll", 
    parameters=udp_par_list_rotor, 
    library='syslib',
    name="Rotor"
    );

if N_pm_layer == 2:
    rotor_temp_id = rotor_id;
    rotor_temp_id.name = "Rotor_temp";
    udp_par_list_rotor2 = [
        ["DiaGap", "d_ro"], 
        ["DiaYoke", "d_ri"], 
        ["Length", "0mm"],
        ["Poles", "N_p"], 
        ["PoleType", int(PoleType)], # doesn't work
        ["D1", str(D1_2)+"mm"],
        ["O1", str(O1_2)+"mm"], 
        ["O2", str(O2_2)+"mm"], 
        ["B1", str(B1_2)+"mm"],
        ["Rib", str(Rib_2)+"mm"],
        ["HRib", str(HRib_2)+"mm"], 
        ["DminMag", str(DminMag_2)+"mm"], 
        ["ThickMag", str(ThickMag_2)+"mm"], 
        ["WidthMag", str(WidthMag_2)+"mm"],
        ["LenRegion", "0mm"],
        ["InfoCore", "0"]
        ];

    rotor_id = mod2D.create_udp(
        dll="RMxprt/IPMCore.dll", 
        parameters=udp_par_list_rotor2, 
        library='syslib',
        name="Rotor"
        );
    
    mod2D.create_circle(
        origin=[0, 0, 0], radius="d_ro/2", name="Rotor_tool"
        );
    
    mod2D.subtract("Rotor_tool", "Rotor_temp", keep_originals=False);
    mod2D.subtract("Rotor", "Rotor_tool", keep_originals=False);
    
M2D.assign_material(
    assignment=rotor_id, material=Mat_core
    );
# rotor_id.name = "Rotor";
rotor_id.color = (192, 192, 192)  # rgb
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
    origin=[0, 0, 0], radius=str(D1/2), name="Rotor1_tool"
    );
mod2D.subtract("Rotor1", "Rotor1_tool", keep_originals=True);
mod2D.intersect(["Rotor", "Rotor1_tool"]);

mod2D.clone("Rotor1");
mod2D.create_circle(
    origin=[0, 0, 0], radius=str((d_ro - (d_ro - D1)/3)/2), 
    name="Rotor2_tool"
    );
mod2D.subtract("Rotor1", "Rotor2_tool", keep_originals=True);
mod2D.intersect(["Rotor2", "Rotor2_tool"]);

mod2D.clone("Rotor2");
mod2D.create_circle(
    origin=[0, 0, 0], radius=str((d_ro - 2*(d_ro - D1)/3)/2), 
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
# Create geometry and assign properties for magnets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Create the geometry for the magnets. It is created via
#    the RMxprt user-defined primitive. A list of lists is
#    created with the proper UDP parameters.
# 2. Assign properties to the magnets. The following code 
#    assigns the material, name, color, transparency, and 
#    ``solve_inside`` properties.

udp_par_list_magnets = udp_par_list_rotor[:-1]+[["InfoCore", "1"]];

magnets_id = mod2D.create_udp(
    dll="RMxprt/IPMCore.dll", 
    parameters=udp_par_list_magnets, 
    library='syslib',
    name='Magnets'
    );

M2D.assign_material(
    assignment=magnets_id, material=Mat_magnet
    );
# magnets_id.name = "Magnets";
magnets_id.color = (255, 128, 128);  # rgb
magnets_id.transparency = 0;
magnets_id.solve_inside = True;  # to be reassigned: M2D.assign material puts False if not dielectric
PMs_id_list = [ magnets_id ];

if N_pm_layer == 2:
    udp_par_list_magnets2 = udp_par_list_rotor2[:-1]+[["InfoCore", "1"]];
    magnets2_id = mod2D.create_udp(
        dll="RMxprt/IPMCore.dll", 
        parameters=udp_par_list_magnets2, 
        library='syslib',
        name='Magnets_2nd_layer'
        );
    M2D.assign_material(
        assignment=magnets2_id, material=Mat_magnet
        );
    # magnets_id.name = "Magnets";
    magnets2_id.color = (255, 128, 128);  # rgb
    magnets2_id.transparency = 0;
    magnets2_id.solve_inside = True;  # to be reassigned: M2D.assign material puts False if not dielectric
    PMs_id_list.append(magnets2_id);

##################################################################
# Create coils
# ~~~~~~~~~~~~
# Create the coils.
##################################################################
# Create the geometry of stator coils
# ~~~~~~~~~~~~
# Create the geometry of stator coils.

d_coil = round(d_slot_in_mm*0.8/N_coil_layer, 2); # Depth of single coil
a_coil = a_slot_in_mm*f_slot_fill/N_coil_layer; # Area of single coil
w_coil = round(a_coil/d_coil, 2);
l_g = (d_si-d_ro)/2; # Length of air gap

r_coil = d_ro/2+l_g+(Hs0+Hs1)*1.5; # Radius of coil head

stator_coil_id = mod2D.create_rectangle(
    origin=[str(r_coil), str(-w_coil/2), 0],
    sizes=[str(d_coil), str(w_coil), 0],
    name='Coil1', material=Mat_coil
    );
stator_coil_id.color = (255, 128, 0);

if N_coil_layer == 2:
    r_coil2 = r_coil+d_coil*1.5;
    stator_coil_id.clone();
    coil2_id = mod2D.get_object_from_name("Coil2");
    mod2D.move(
        assignment="Coil2",
        vector=[d_coil*1.1, 0, 0]
        );
Tht_slot = 360/Q_s;
mod2D.rotate(
    assignment=stator_coil_id, axis="Z", angle=str(Tht_slot*0.5)+"deg"
    );
stator_coil_id.duplicate_around_axis(
    axis="Z", angle=str(Tht_slot), clones="Q_s/N_p",
    create_new_objects=True
    );

if N_coil_layer == 2:
    mod2D.rotate(
        assignment=coil2_id, axis="Z", 
        angle=str(Tht_slot*0.5)+"deg"
        );
    coil2_id.duplicate_around_axis(
        axis="Z", angle=str(Tht_slot), clones="Q_s/N_p",
        create_new_objects=True
        );    
    
stator_coils_id_list = [];

for x in range(int(Q_s/N_p)):
    if x == 0:
        str_idx = "";
    else:
        str_idx = "_"+str(int(x));
    stator_coil_id = mod2D.get_object_from_name(assignment="Coil1"+str_idx);
    stator_coils_id_list.append(stator_coil_id);
    idx_p = int(x/(Q_s/N_p));
    str_coil_idx = str(int(x%q_s+idx_p*q_s+1));
    match int(idx_p%2):
        case 0:
            str_polar_1 = "_pos";
            str_polar_2 = "_neg";
        case 1:
            str_polar_1 = "_neg";
            str_polar_2 = "_pos";
    match int(x/q_s%3):
        case 0:
            stator_coil_id.name = "B"+str_coil_idx+str_polar_2;
            stator_coil_id.color = (0, 0, 255);
            stator_coil_id.transparency = 0;
        case 1:
            stator_coil_id.name = "A"+str_coil_idx+str_polar_1;
            stator_coil_id.color = (255, 0, 0);
            stator_coil_id.transparency = 0;
        case 2:
            stator_coil_id.name = "C"+str_coil_idx+str_polar_2;            
            stator_coil_id.color = (0, 128, 0);
            stator_coil_id.transparency = 0;
    if N_coil_layer == 2:
        N_x = N_coil_pitch - Q_s/N_p; # start number of the second layer
        if N_x+x < 0:
            N_x = Q_s/N_p + N_x;
            IsPolorSwp = True;
        else:
            IsPolorSwp = False;
        if N_x+x >= Q_s/N_p:
            N_x = N_x - Q_s/N_p;
        Tht_coil2 = Tht_slot*(N_x+x+0.5)/180*pi;
        coil_name = mod2D.get_bodynames_from_position(
            [r_coil2*cos(Tht_coil2), r_coil2*sin(Tht_coil2), 0], 
            include_non_model=True
            );
        coil_name = coil_name[0];
        stator_coil_id = mod2D.get_object_from_name(assignment=coil_name);
        stator_coils_id_list.append(stator_coil_id);
        str_coil2_idx = str(int(x%q_s+idx_p*q_s+q_s+1));
        match int(x/q_s%3):
            case 0:
                stator_coil_id.name = "B"+str_coil2_idx+str_polar_2;
                if IsPolorSwp:
                    stator_coil_id.name = "B"+str_coil2_idx+str_polar_1;
                stator_coil_id.color = (0, 0, 255);
                stator_coil_id.transparency = 0;
            case 1:
                stator_coil_id.name = "A"+str_coil2_idx+str_polar_1;
                if IsPolorSwp:
                    stator_coil_id.name = "A"+str_coil2_idx+str_polar_2;
                stator_coil_id.color = (255, 0, 0);
                stator_coil_id.transparency = 0;
            case 2:
                stator_coil_id.name = "C"+str_coil2_idx+str_polar_2;            
                if IsPolorSwp:
                    stator_coil_id.name = "C"+str_coil2_idx+str_polar_1;
                stator_coil_id.color = (0, 128, 0);
                stator_coil_id.transparency = 0;

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
    ] + rotor_id_list+ PMs_id_list + vacuum_obj_id_list;

mod2D.create_coordinate_system(
    origin=[0, 0, 0], reference_cs="Global", name="Section",
    mode="axis",
    x_pointing=["cos(360deg/N_p)", "sin(360deg/N_p)", 0],
    y_pointing=["-sin(360deg/N_p)", "cos(360deg/N_p)", 0]
    );

mod2D.set_working_coordinate_system("Section");
mod2D.split(object_list, "ZX", sides="NegativeOnly");
mod2D.set_working_coordinate_system("Global");
mod2D.split(object_list, "ZX", sides="PositiveOnly");
mod2D.fit_all(); # Fit the window to all models

# %% Create coordinate system for PMs
# ~~~~~~~~~~~~~~~~~~~~~~~
# Create coordinate system for the PMs (permanent magnets). 
# In Maxwell 2D, you assign magnetization via the coordinate 
# system. Because each PM needs to have a coordinate system in 
# the face center, auxiliary functions are created. 
# ================================================================
# Create function: ``find_elements()``
# ~~~~~~~~~~~~~~~~~~~~~~~
# Create the auxiliary function ``find_elements(lst1, lst2)`` to 
# find the elements in list ``lst1`` with indexes in list ``lst2``.

def find_elements(lst1, lst2):
    return [lst1[i] for i in lst2]

# ================================================================
# Create function: ``find_n_largest ()``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Use the auxiliary function ``find_n_largest (input_len_list, n_largest_edges)``
# to find the ``n`` largest elements in the list ``input_len_list``.

def find_n_largest(input_len_list, n_largest_edges):
    tmp = list(input_len_list)
    copied = list(input_len_list)
    copied.sort()  # sort list so that largest elements are on the far right
    index_list = []
    for n in range(1, n_largest_edges + 1):  # get index of the nth largest element
        index_list.append(tmp.index(copied[-n]))
        tmp[tmp.index(copied[-n])] = 0  # index can only get the first occurrence that solves the problem
    return index_list

# ================================================================
# Create function: ``create_cs_magnets``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a function to create coordinate system for the PMs. 
# The inputs are the object name, coordinate system name, and 
# inner or outer magnetization. Find the two longest edges of the 
# magnets and get the midpoint of the outer edge. You must have 
# this point to create the face coordinate systems in case of 
# outer magnetization.

def create_cs_magnets(pm_id, cs_name, point_direction):
    """
    Parameters
    ----------
    pm_id : ANSYS object_3d
        DESCRIPTION.
    cs_name : String
        Name of cs coordination.
    point_direction : String
        Direction of point, "outer" or "inner".
    """
    pm_face_id = mod2D.get_object_faces(pm_id.name)[0];  # works with name only
    pm_edges = mod2D.get_object_edges(pm_id.name);  # gets the edges of the PM object
    edge_len_list = list(
        map(mod2D.get_edge_length, pm_edges)
        );  # apply method get_edge_length to all elements of list pm_edges
    index_2_longest = find_n_largest(edge_len_list, 2)  # find the 2 longest edges of the PM
    longest_edge_list = find_elements(pm_edges, index_2_longest);
    edge_center_list = list(
        map(mod2D.get_edge_midpoint, longest_edge_list)
        );  # apply method get_edge_midpoint to all elements of list longest_edge_list

    rad = lambda x: sq(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    index_largest_r = find_n_largest(
        list(map(rad, edge_center_list)), 2
        );
    longest_edge_list2 = [
        longest_edge_list[i] for i in index_largest_r
        ];  # reorder: outer first element of the list
    if point_direction == "outer":
        my_axis_pos = longest_edge_list2[0];
    elif point_direction == "inner":
        my_axis_pos = longest_edge_list2[1];

    mod2D.create_face_coordinate_system(
        face=pm_face_id, origin=pm_face_id, 
        axis_position=my_axis_pos, axis="X", name=cs_name
        );
    pm_id.part_coordinate_system = cs_name;
    mod2D.set_working_coordinate_system('Global');

# ================================================================
# Create outer and inner PMs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create the outer and inner PMs by applying "seperate_bodies" on 
# "Magnets", retrive their ids, and rename them for later use of 
# building cs coodination systems.

mod2D.separate_bodies(assignment="Magnets");

Dict_OPM1 = defaultdict(list);
Dict_IPM1 = defaultdict(list);
IPM1_id = mod2D.get_object_from_name("Magnets");
Dict_IPM1["id"] = IPM1_id;
IPM1_id.name = "IPM1";
OPM1_id = mod2D.get_object_from_name(
    "Magnets_Separate1"
    );
Dict_OPM1["id"] = OPM1_id;
OPM1_id.name = "OPM1";

Dict_IPM1["vertice ids"] = mod2D.get_object_vertices(
    "IPM1"
    );
Dict_OPM1["vertice ids"] = mod2D.get_object_vertices(
    "OPM1"
    );

for idx, vertice_id in enumerate(Dict_OPM1["vertice ids"]):    
    [x, y, z] = mod2D.get_vertex_position(
        assignment=vertice_id
        );
    Dict_OPM1["vertice x"].append(x);
    Dict_OPM1["vertice y"].append(y);
    Dict_OPM1["vertice z"].append(z);
    Dict_OPM1["vertice position"].append([x, y, z]);
    [x, y, z] = mod2D.get_vertex_position(
        assignment=Dict_IPM1["vertice ids"][idx]
        );
    Dict_IPM1["vertice x"].append(x);
    Dict_IPM1["vertice y"].append(y);
    Dict_IPM1["vertice z"].append(z);
    Dict_IPM1["vertice position"].append([x, y, z]); 
if N_pm_layer == 2:
    mod2D.separate_bodies(assignment="Magnets_2nd_layer");
    IPM2_id = mod2D.get_object_from_name("Magnets_2nd_layer");
    IPM2_id.name = "IPM2";
    OPM2_id = mod2D.get_object_from_name(
        "Magnets_2nd_layer_Separate1"
        );
    OPM2_id.name = "OPM2";

# ================================================================
# Create coordinate system for PMs in face center
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create the coordinate system for PMs in the face center.

create_cs_magnets(IPM1_id, 'CS_' + IPM1_id.name, 'inner');
create_cs_magnets(OPM1_id, 'CS_' + OPM1_id.name, 'outer');
if N_pm_layer == 2:
    create_cs_magnets(IPM2_id, 'CS_' + IPM2_id.name, 'inner');
    create_cs_magnets(OPM2_id, 'CS_' + OPM2_id.name, 'outer');


# %% Assign motion setup to object
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Assign a motion setup to a ``Band`` object.

M2D.assign_rotate_motion(
    assignment="Band", 
    coordinate_system="Global", 
    axis="Z", 
    positive_movement=True, 
    start_position="MechAngle_d_oriented", 
    has_rotation_limits=False,
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
            assignment=[coil_id.name], conductors_number="N_s", 
            polarity="Positive", name=coil_id.name
            );
    elif "_neg" in coil_id.name:       
        M2D.assign_coil(
            assignment=[coil_id.name], conductors_number="N_s", 
            polarity="Negative", name=coil_id.name
            );
    for ph_x in Dict_phase["Phase list"]:
        if ph_x in coil_id.name:
            Dict_phase["Phase_"+ph_x+" coil list"].append(coil_id.name);
for idx_ph, ph_x in enumerate(Dict_phase["Phase list"]):
    M2D.assign_winding(
        winding_type="Current", 
        is_solid=False, 
        current=Dict_phase["Current list"][idx_ph],
        parallel_branches="N_pb",
        name="Phase_"+ph_x
        );
    M2D.add_winding_coils(
        assignment="Phase_"+ph_x, 
        coils=Dict_phase["Phase_"+ph_x+" coil list"]
        );
    
#==============================================================================
# Assign total current on PMs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Assign a total current of ``0`` on the PMs.
PMs_id_list = mod2D.get_objects_by_material(Mat_magnet);
for PM_id in PMs_id_list:
    M2D.assign_current(
        PM_id, 
        amplitude=0, 
        solid=True, 
        name=PM_id.name + "coil"
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
M2D.mesh.assign_length_mesh(
    PM_id, 
    inside_selection=True, 
    maximum_length=3, 
    maximum_elements=None, 
    name="magnet"
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
Id_start=-np.ceil(Is_max/100)*100;
Id_stop=0;
Id_step=-Id_start/5;

Iq_start=0;
Iq_stop=np.ceil(Is_max/100)*100;
Iq_step=Iq_stop/5;

Arr_Id = np.arange(Id_start, Id_stop+Id_step*.5, Id_step);
List_Id = list(Arr_Id);
List_Id[-1] = -5;
Arr_Id = np.insert(Arr_Id, len(Arr_Id)-1, -5);
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
# new_data = pd.DataFrame({
#     "*": [len(DF_I_dq["*"])+1],
#     "I_d": ["0A"],
#     "I_q": ["0A"],
# });
# DF_I_dq = pd.concat([DF_I_dq, new_data], ignore_index=True);

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
    "P_PM": ["SolidLoss", "[W]"]
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
        primary_sweep_variable="Time",
        domain="Sweep"
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
        DF_temp = DF_temp.iloc[-N_step_per_period:];


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
    u_A = DF_temp["u_i_A[V]"]+R_s*DF_temp["i_A[A]"];
    u_B = DF_temp["u_i_B[V]"]+R_s*DF_temp["i_B[A]"];
    u_C = DF_temp["u_i_C[V]"]+R_s*DF_temp["i_C[A]"];
    u_alpha = u_A;
    u_beta = (u_B-u_C)/sq(3);
    U_d = mean(u_alpha*cos(gamma)+u_beta*sin(gamma));
    U_q = mean(-u_alpha*sin(gamma)+u_beta*cos(gamma));
    Dict_Mxwl["U_d[V]"].append(U_d);
    Dict_Mxwl["U_q[V]"].append(U_q);
    Dict_Mxwl["U_s[V]"].append(
        sq(U_d**2+U_q**2)
        );
    
    Dict_Mxwl["Trq[Nm]"].append(mean(DF_temp["Trq[Nm]"]));
    Dict_Mxwl["P_fe[W]"].append(mean(DF_temp["P_fe[W]"]));
    Dict_Mxwl["P_PM[W]"].append(mean(DF_temp["P_PM[W]"]));
    print(
        "*** Finish extracting raw data with I_d = "+DF_I_dq[
            "I_d"][idx-1]+" and I_q = "+DF_I_dq["I_q"][idx-1]+" ***"
        );
    # pdb.set_trace();
DF_Mxwl = pd.DataFrame.from_dict(Dict_Mxwl);
idcs = DF_Mxwl[DF_Mxwl["I_d[A]"] == 0].index;
Arr_Psi_m = np.array(DF_Mxwl["Psi_d[Wb]"][idcs]);
Arr_I_q = np.array(DF_Mxwl["I_q[A]"][idcs]);
DF_Psi_m = pd.DataFrame();
DF_Psi_m["Psi_m[Wb]"] = Arr_Psi_m;
DF_Psi_m["I_q[A]"] = Arr_I_q;
DF_Psi_m.to_csv(PWD+"Psi_m(I_q).csv");

plt.figure(dpi=300);
plt.grid();
plt.scatter(DF_Psi_m["I_q[A]"], DF_Psi_m["Psi_m[Wb]"]);
plt.xlabel("I_q [A]");
plt.ylabel("Psi_m [Wb]");

Arr_I_q, Arr_Psi_m = np.meshgrid(
    Arr_I_q, Arr_Psi_m
    );

# Flatten the arrays to get all combinations
Series_Psi_m = Arr_Psi_m.flatten();

DF_Mxwl = DF_Mxwl.drop(idcs);
DF_Mxwl["Psi_m[Wb]"] = Series_Psi_m;
# pdb.set_trace();
# %% Close AEDT
# ~~~~~~~~~~
# Close AEDT.
M2D.release_desktop();
# %% Map L_d & L_q
Dict_Map = defaultdict();

DF_Mxwl["L_d[uH]"] = (
    DF_Mxwl["Psi_d[Wb]"] - DF_Mxwl["Psi_m[Wb]"]
    )/DF_Mxwl["I_d[A]"]*1e6;
DF_Mxwl["L_q[uH]"] = DF_Mxwl["Psi_q[Wb]"]/DF_Mxwl["I_q[A]"]*1e6;
for Var in DF_Mxwl.keys():
    # if Var != "I_d[A]" and Var != "I_q[A]":
    Dict_temp = defaultdict(list);
    Dict_temp["I_q[A]"] = [0] + DF_Psi_m["I_q[A]"].to_list();
    for idx_Id, I_d in enumerate(List_Id):
        key = "Id_"+str(idx_Id+1);
        idcs = DF_Mxwl[DF_Mxwl["I_d[A]"] == I_d].index;
        Dict_temp[key] = [I_d] + list(DF_Mxwl[Var][idcs]);
    Dict_Map[Var] = pd.DataFrame.from_dict(Dict_temp);
    file_name = Var.replace("[","_in_").replace("]","")+"_map.csv";
    Dict_Map[Var].to_csv(PWD+file_name, index=False);
    # `index=False` avoids saving row indices
# %% Plot contours of L_d-L_q
# Define the arc parameters
theta = np.linspace(np.pi/2, np.pi, 100);  # 90-degree arc (0 to π/2 radians)
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
        plt.xlim(-400,0);
        plt.ylim(0,400);
        plt.title("$"+Var+"$");
        plt.xlabel("$I_d$[A]", fontsize=14);
        plt.ylabel("$I_q$[A]", fontsize=14);
        plt.grid(True, linestyle='--', alpha=0.5);
        # Add colorbar to indicate values
        plt.colorbar(contour_lines);
