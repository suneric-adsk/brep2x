import numpy as np

PRIM_COMMANDS = ['SOL', 'EOS', 'box', 'prism', 'cylinder', 'cone', 'sphere']

FEAT_COMMANDS = ['SOL', 'EOS', 
                'rect_slot', 'tri_slot', 'cir_slot', 
                'rect_psg', 'tri_psg', 'hexa_psg', 
                'hole', 'rect_step', 'tside_step', 
                'slant_step', 'rect_b_step', 'tri_step', 'cir_step', 
                'rect_b_slot', 'cir_b_slot', 'u_b_slot', 
                'rect_pkt', 'key_pkt', 'tri_pkt', 'hexa_pkt', 
                'o_ring', 'b_hole', 'chamfer', 'fillet']

BOX_IDX = PRIM_COMMANDS.index('box')
PRISM_IDX = PRIM_COMMANDS.index('prism')
CYLINDER_IDX = PRIM_COMMANDS.index('cylinder')
CONE_IDX = PRIM_COMMANDS.index('cone')
SPHERE_IDX = PRIM_COMMANDS.index('sphere')

RECT_SLOT_IDX = FEAT_COMMANDS.index('rect_slot')
TRI_SLOT_IDX = FEAT_COMMANDS.index('tri_slot')
CIR_SLOT_IDX = FEAT_COMMANDS.index('cir_slot')
RECT_PSG_IDX = FEAT_COMMANDS.index('rect_psg')
TRI_PSG_IDX = FEAT_COMMANDS.index('tri_psg')
HEXA_PSG_IDX = FEAT_COMMANDS.index('hexa_psg')
HOLE_IDX = FEAT_COMMANDS.index('hole')
RECT_STEP_IDX = FEAT_COMMANDS.index('rect_step')
TSIDE_STEP_IDX = FEAT_COMMANDS.index('tside_step')
SLANT_STEP_IDX = FEAT_COMMANDS.index('slant_step')
RECT_B_STEP_IDX = FEAT_COMMANDS.index('rect_b_step')
TRI_STEP_IDX = FEAT_COMMANDS.index('tri_step')
CIR_STEP_IDX = FEAT_COMMANDS.index('cir_step')
RECT_B_SLOT_IDX = FEAT_COMMANDS.index('rect_b_slot')
CIR_B_SLOT_IDX = FEAT_COMMANDS.index('cir_b_slot')
U_B_SLOT_IDX = FEAT_COMMANDS.index('u_b_slot')
RECT_PKT_IDX = FEAT_COMMANDS.index('rect_pkt')
KEY_PKT_IDX = FEAT_COMMANDS.index('key_pkt')
TRI_PKT_IDX = FEAT_COMMANDS.index('tri_pkt')
HEXA_PKT_IDX = FEAT_COMMANDS.index('hexa_pkt')
O_RING_IDX = FEAT_COMMANDS.index('o_ring')
B_HOLE_IDX = FEAT_COMMANDS.index('b_hole')
CHAMFER_IDX = FEAT_COMMANDS.index('chamfer')
FILLET_IDX = FEAT_COMMANDS.index('fillet')

SOL_IDX = PRIM_COMMANDS.index('SOL')
EOS_IDX = PRIM_COMMANDS.index('EOS')
EXT_IDX = -1
PAD_VAL = -1

N_PRIM_COMMANDS = len(PRIM_COMMANDS)
N_FEAT_COMMANDS = len(FEAT_COMMANDS)

N_PRIM_PARAM = 11  # primitive parameters: L1, L2, L3, E, Tx, Ty, Tz, Q0, Q1, Q2, Q3
N_FEAT_PARAM = 11  # feature parameters:  X1, Y1, Z1, X2, Y2, Z2, W, L, R, W1, D

PRIM_SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_PRIM_PARAM)])
PRIM_EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_PRIM_PARAM)])
FEAT_SOL_VEC = np.array([SOL_IDX, *([PAD_VAL] * N_FEAT_PARAM)])
FEAT_EOS_VEC = np.array([EOS_IDX, *([PAD_VAL] * N_FEAT_PARAM)])

PRIM_PARAM_MASK = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SOL
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # EOS
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],  # box
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],  # prism
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # cylinder
    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],  # cone
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # sphere
]) # L1 L2 L3 E Tx Ty Tz Q0 Q1 Q2 Q3

FEAT_PARAM_MASK = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # SOL
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # EOS
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # rect_slot
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # tri_slot
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # cir_slot
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # rect_psg
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # tri_psg
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # hexa_psg
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # hole
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # rect_step
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1],  # tside_step
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # slant_step
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # rect_b_step
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # tri_step
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1],  # cir_step
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # rect_b_slot
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # cir_b_slot
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # u_b_slot
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # rect_pkt
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],  # key_pkt
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # tri_pkt
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # hexa_pkt
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # o_ring
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # b_hole
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # chamfer
    [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # fillet
]) # X1 Y1 Z1 X2 Y2 Z2 W  L  R  W1 D     

MAX_PRIM = 10 # maximum number of main_feature
MAX_FEAT = 12  # maximum number of sub_feature
PARAM_DIM = 258  #-1-256

N_ALL_SAMPLE = 1000
N_PART_SAMPLE = 210
GRID_BOUND = 500.0
GRID_SIZE = 64


