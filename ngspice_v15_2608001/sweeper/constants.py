"""Shared paths, circuit constants, and editor field definitions."""

from __future__ import annotations

from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]

# Starting folders for the WAV to PWL and Transient tabs.  These are the only
# machine-specific paths in the program; everything else is derived from
# APP_DIR.  Edit these two lines when the dataset lives elsewhere.  Both are
# only prefilled defaults, so the GUI folder pickers still override them.
DEFAULT_WAV_DATASET = Path(
    r"C:\Users\duyou\Desktop\Personal\10_Study\10_SemiContest"
    r"\10_AI_Project\speech_commands_for_sim"
)
DEFAULT_PWL_DATASET = Path(
    r"C:\Users\duyou\Desktop\Personal\10_Study\10_SemiContest"
    r"\10_AI_Project\speech_commands_pwl"
)

# SPICE model libraries shared by every analysis.  The detector op-amp is
# selectable, so its library is included per run instead of being hard-coded in
# the netlist template.
MODEL_LIBRARY_DIR = Path(
    r"C:\Users\duyou\Desktop\Personal\10_Study\10_SemiContest"
    r"\10_AI_Project\lib"
)
# Every entry was checked against its .lib file: all three op-amps declare the
# same terminal order (IN+, IN-, V+, V-, OUT), so only the model name changes.
DETECTOR_OPAMP_CHOICES = (
    ("OPA379", "OPA379.LIB"),
    ("TLV9041D", "TLV9041D.lib"),
    ("TLV9042", "TLV9042.lib"),
)
DETECTOR_OPAMP_NAMES = tuple(name for name, _library in DETECTOR_OPAMP_CHOICES)
DETECTOR_OPAMP_LIBRARIES = dict(DETECTOR_OPAMP_CHOICES)
DEFAULT_DETECTOR_OPAMP = DETECTOR_OPAMP_NAMES[0]
# Reference designator of the active detector op-amp in netlist_template.cir.
DETECTOR_OPAMP_REFERENCE = "XU3"

DEFAULT_TABLE = APP_DIR / "channel_components.csv"
DEFAULT_NETLIST = APP_DIR / "netlist_template.cir"
DEFAULT_RESULTS = APP_DIR / "results"
DEFAULT_SCHEMATIC = APP_DIR / "circuit_template.png"
DEFAULT_COMPONENT_VERSIONS = APP_DIR / "component_versions"
AC_RAW_FILENAME = "ac.raw"
OP_RAW_FILENAME = "op.raw"
TRAN_RAW_FILENAME = "tran.raw"
DETECTOR_RAW_FILENAME = "detector.raw"

# Canonical hierarchical net names verified against netlist(6).txt and
# sch(1).pdf on 2026-07-31.  Generators and parsers import these constants so a
# stale alias cannot silently reappear in one analysis path.
VIN_NODE = "/vin"
V_FILT_NODE = "/v_filt"
V_ENV_NODE = "/v_env"
V_THR_NODE = "/v_thr"
V_COMP_NODE = "/v_comp"

TRANSIENT_OUTPUT_STEP_DEFAULT_S = 10e-6
TRANSIENT_MAXIMUM_STEP_DEFAULT_S = 5e-6
TRANSIENT_INPUT_VPP_DEFAULT_V = 0.010
DETECTOR_FREQUENCY_DEFAULT_HZ = 1000.0
DETECTOR_INPUT_VPP_DEFAULT_V = 0.010
DETECTOR_GATE_ON_DEFAULT_S = 0.050
DETECTOR_GATE_DURATION_DEFAULT_S = 0.200
DETECTOR_TOTAL_TIME_DEFAULT_S = 0.350
DETECTOR_MAXIMUM_STEP_DEFAULT_S = 5e-6
DETECTOR_TIMESTEP_WARNING_POINTS_PER_CYCLE = 20
AUTO_RANGE_DATA_FRACTION = 0.90
DIVIDER_SUPPLY_V = 1.8
DIVIDER_NOMINAL_TOTAL_KOHM = 1000.0
RESISTANCE_DECIMALS = 2
DIVIDER_RESISTANCE_DECIMALS = RESISTANCE_DECIMALS
# DC operating-point voltages are shown in millivolts with this many decimals.
# Results keep full precision in the rawfile and CSV; this is display only.
VOLTAGE_DISPLAY_DECIMALS = 2
OP_DEFAULT_SCHEMATIC_FRACTION = 0.82

# Pixel centres of the "{NAME}" placeholder labels in circuit_template.png
# (1280x590), measured from the image itself.  Each applied value is drawn on
# top of its own placeholder so no reference designator is covered.
SCHEMATIC_COMPONENT_POSITIONS = (
    ("R1", 227, 117),
    ("C1", 318, 101),
    ("RA", 310, 187),
    ("C1", 343, 218),
    ("RA", 454, 275),
    ("R2", 613, 303),
    ("R2", 614, 404),
    ("R4", 646, 277),
    ("R5", 787, 101),
    ("R6", 714, 342),
    ("C3", 956, 334),
    ("R7", 1033, 202),
    ("R8", 1033, 346),
)
# Centre of the template's printed "OPA379" text beside U3.
SCHEMATIC_DETECTOR_OPAMP_POSITION = (807, 228)
# The placeholders are only about 22 px wide, so the overlay uses a small font
# and a one-pixel border to stay inside them.
SCHEMATIC_VALUE_FONT_SIZE = 8

PARAM_NAMES = ("RA", "R1", "R2", "R4", "R5", "R6", "R7", "R8", "C1", "C3")
CHANNEL_CSV_COLUMNS = (
    "ch",
    "f_c_hz",
    "Q",
    "RA_kohm",
    "C_nF",
    "R1_kohm",
    "gain_dB",
    "R2_kohm",
    "R4_kohm",
    "R5_kohm",
    "R6_kohm",
    "C3_nF",
    "R7_kohm",
    "R8_kohm",
    "Vthr_V",
    "detector_opamp",
)
COMPONENT_EDIT_FIELDS = (
    ("RA", "RA1, RA2 [kΩ]", "ra_kohm"),
    ("R1", "R1 [kΩ]", "r1_kohm"),
    ("R2", "R2, R3 [kΩ]", "r2_kohm"),
    ("R4", "R4 [kΩ]", "r4_kohm"),
    ("R5", "R5 [kΩ]", "r5_kohm"),
    ("R6", "R6 [kΩ]", "r6_kohm"),
    ("R7", "R7 [kΩ]", "r7_kohm"),
    ("R8", "R8 [kΩ]", "r8_kohm"),
    ("C1", "C1, C2 [nF]", "c_nf"),
    ("C3", "C3 [nF]", "c3_nf"),
)
RESISTANCE_EDIT_KEYS = frozenset(
    {"RA", "R1", "R2", "R4", "R5", "R6", "R7", "R8"}
)
REQUIRED_COLUMNS = {
    "ch",
    "f_c_hz",
    "Q",
    "RA_kohm",
    "C_nF",
    "R1_kohm",
    "R7_kohm",
    "R8_kohm",
}
DEFAULT_COMMON = {
    "R2_kohm": 100.0,
    "R4_kohm": 10.0,
    "R5_kohm": 47.0,
    "R6_kohm": 8.25,
    "C3_nF": 100.0,
}

C3_ALLOWED_NF = (
    1.0,
    1.2,
    1.5,
    1.8,
    2.2,
    2.7,
    3.3,
    3.9,
    4.7,
    5.6,
    6.8,
    8.2,
    10.0,
    12.0,
    15.0,
    18.0,
    22.0,
    27.0,
    33.0,
    39.0,
    47.0,
    56.0,
    68.0,
    82.0,
    100.0,
)
C3_ALLOWED_NF_LABELS = (
    "1.0",
    "1.2",
    "1.5",
    "1.8",
    "2.2",
    "2.7",
    "3.3",
    "3.9",
    "4.7",
    "5.6",
    "6.8",
    "8.2",
    "10",
    "12",
    "15",
    "18",
    "22",
    "27",
    "33",
    "39",
    "47",
    "56",
    "68",
    "82",
    "100",
)
