"""
config.py -- All EdgeTrust parameters in one place, matching the manuscript's
GLOBAL STATE & PARAMETERS block (Section IV) exactly. Change values here,
not inside trust_engine.py, so every script stays consistent.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import os

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "CICIoT2023")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

for d in (OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------
# EdgeTrust equation parameters (Eqs. 1-6), matching the manuscript's
# CURRENT parameter block. delta=0.08 is what the current manuscript
# states; change to 0.05 here if you have applied the earlier
# consistency fix (unified deadband + 3-epoch promotion confirmation).
# ----------------------------------------------------------------------
@dataclass
class EdgeTrustParams:
    m: int = 14                 # number of behavioral criteria (Section III.3)
    rho: float = 0.3            # historical-trust weight, Eq. (3)
    alpha: float = 0.5          # classifier-evidence weight, Eq. (3)
    alpha_H: float = 0.4        # Holt level smoothing, Eq. (4)
    beta_H: float = 0.2         # Holt trend smoothing, Eq. (4)
    h: int = 2                  # forecast horizon (epochs), Eq. (4)
    tau_base: float = 0.35      # sigmoid lower bound, Eq. (5)
    tau_max: float = 0.65       # sigmoid upper bound, Eq. (5)
    kappa: float = 10.0         # sigmoid steepness, Eq. (5)
    mu: float = 0.5             # sigmoid midpoint, Eq. (5)
    delta: float = 0.08         # hysteresis deadband half-width, Eq. (6)
    promote_confirm: int = 3    # consecutive epochs to confirm a promotion (Section III.2)
    hard_block_threshold: float = 0.90  # Layer-1 immediate-block threshold on q_t

PARAMS = EdgeTrustParams()

# ----------------------------------------------------------------------
# Attack-category taxonomy (7 top-level categories, 33 sub-attacks),
# per Neto et al. 2023 (ref [34]) and the CIC IoT Lab dataset page.
# Label strings below reflect the commonly distributed CICIoT2023 CSV
# naming convention. If your CSVs use different exact strings, the
# loader in data_pipeline.py prints every UNMATCHED label it encounters
# so you can extend this dict rather than fail silently.
# ----------------------------------------------------------------------
LABEL_TO_CATEGORY: Dict[str, str] = {
    # DDoS
    "DDoS-ICMP_Flood": "DDoS", "DDoS-UDP_Flood": "DDoS", "DDoS-TCP_Flood": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS", "DDoS-SYN_Flood": "DDoS", "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS", "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS", "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-SlowLoris": "DDoS", "DDoS-HTTP_Flood": "DDoS",
    # DoS
    "DoS-UDP_Flood": "DoS", "DoS-SYN_Flood": "DoS", "DoS-TCP_Flood": "DoS",
    "DoS-HTTP_Flood": "DoS",
    # Mirai
    "Mirai-greeth_flood": "Mirai", "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Recon
    "Recon-PingSweep": "Recon", "Recon-OSScan": "Recon", "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon", "Recon-HostDiscovery": "Recon",
    # Spoofing
    "DNS_Spoofing": "Spoofing", "MITM-ArpSpoofing": "Spoofing",
    # Web-based
    "BrowserHijacking": "Web-based", "Backdoor_Malware": "Web-based", "XSS": "Web-based",
    "Uploading_Attack": "Web-based", "SqlInjection": "Web-based", "CommandInjection": "Web-based",
    # Brute Force
    "DictionaryBruteForce": "BruteForce",
    # Benign
    "BenignTraffic": "Benign", "Benign": "Benign",
}

CATEGORIES: List[str] = ["Benign", "DDoS", "DoS", "Recon", "Web-based", "BruteForce", "Spoofing", "Mirai"]

# ----------------------------------------------------------------------
# All 47 raw CICIoT2023 features, EXACT column names as published on the
# CIC IoT Lab dataset page. LABEL_COL is whatever your CSV calls the
# ground-truth column, "label" is the standard release name (lowercase);
# change here once if your copy differs.
# ----------------------------------------------------------------------
LABEL_COL = "label"

RAW_FEATURES: List[str] = [
    "flow_duration", "Header_Length", "Protocol Type", "Duration", "Rate", "Srate", "Drate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "urg_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
    "TCP", "UDP", "DHCP", "ARP", "ICMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]

N_TOP_FEATURES = 14

RANDOM_SEED = 20260913
