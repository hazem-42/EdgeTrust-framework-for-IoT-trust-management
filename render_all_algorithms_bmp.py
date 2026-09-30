"""
render_all_algorithms_bmp.py

Renders publication-grade 300 DPI BMP (and PNG) images for:
1. Algorithm 2: Full Layer-2 Multi-Criteria Trust Engine & ABAC Workflow
2. Algorithm 3: Operational Scenario 1 (First Join / Onboarding)
3. Algorithm 4: Operational Scenario 2 (Rejoin After Normal Disconnection)
4. Algorithm 5: Operational Scenario 3 (Rejoin After Blocking / DENY)

Exact styling matches Algorithm 1 (IEEE / Q1 Computer Modern serif typography,
thick top/bottom rules, thin mid-rule, line numbering, indentations, and right-aligned ▷ comments).
"""

import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

OUT_DIR = r"D:\PHD\Implementation\edgetrust_pipeline\outputs\figures"
ARTIFACT_DIR = r"C:\Users\Hazem\.gemini\antigravity\brain\538e81b7-7d30-48dd-a733-22e007868d04"
os.makedirs(OUT_DIR, exist_ok=True)


def draw_algorithm_canvas(fig_w, fig_h, title_str, req_str, lines, out_base_name):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    x_left = 0.015
    x_lineno = 0.018
    x_code = 0.048
    x_right = 0.985

    n_lines = len(lines)
    y_top = 0.970
    y_title = 0.935
    y_mid = 0.900
    y_req = 0.865

    avail_h = y_req - 0.045
    line_dy = min(0.038, avail_h / (n_lines + 0.5))
    y_code_start = y_req - 0.042

    # Top rule (thick)
    ax.plot([x_left, x_right], [y_top, y_top], color="black", lw=1.6)

    # Title
    ax.text(x_left, y_title, title_str, fontsize=11.0, color="black", va="center")

    # Mid rule (thin)
    ax.plot([x_left, x_right], [y_mid, y_mid], color="black", lw=0.75)

    # Require / Ensure line
    ax.text(x_left, y_req, req_str, fontsize=9.4, color="black", va="center")

    # Code lines
    for i, (lineno, code_text, comment_text, indent, is_header) in enumerate(lines):
        y = y_code_start - i * line_dy
        
        if is_header:
            ax.text(x_code, y, code_text, fontsize=8.8, color="#0b5394", fontweight="bold", va="center")
        else:
            ax.text(x_lineno, y, f"{lineno}:", fontsize=8.8, color="black", va="center")
            indent_x = x_code + indent * 0.028
            ax.text(indent_x, y, code_text, fontsize=8.9, color="black", va="center")
            if comment_text:
                ax.text(x_right, y, comment_text, fontsize=8.4, color="#333333", va="center", ha="right")

    # Bottom rule (thick)
    y_bottom = y_code_start - n_lines * line_dy + 0.010
    ax.plot([x_left, x_right], [y_bottom, y_bottom], color="black", lw=1.6)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)

    png_path = os.path.join(OUT_DIR, f"{out_base_name}.png")
    bmp_path = os.path.join(OUT_DIR, f"{out_base_name}.bmp")

    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

    with Image.open(png_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(bmp_path, format="BMP")

    # Copy to artifact directory
    art_png = os.path.join(ARTIFACT_DIR, f"{out_base_name}.png")
    art_bmp = os.path.join(ARTIFACT_DIR, f"{out_base_name}.bmp")
    shutil.copy2(png_path, art_png)
    shutil.copy2(bmp_path, art_bmp)

    print(f"Rendered {out_base_name}:")
    print(f"  BMP: {bmp_path} ({os.path.getsize(bmp_path):,} bytes, {rgb_img.size})")
    print(f"  PNG: {png_path} ({os.path.getsize(png_path):,} bytes)")


# ======================================================================
# 1. Algorithm 2: Full Layer-2 Multi-Criteria Trust Engine & ABAC
# ======================================================================
def render_algorithm_2():
    title = r"$\mathbf{Algorithm\ 2}$ Layer 2 — Multi-Criteria Dynamic Trust Engine & Access Control (runs per epoch, per device)"
    req = r"$\mathbf{Require:}$ device $d$, Layer-1 threat score $q_t$, telemetry $x_t$, state cache $\{T_{t-1}, l_{t-1}, b_{t-1}, c, s\}$"
    lines = [
        (0, "// Stage 1: Dynamic Evidence Normalization & Objective Criteria Weighting", "", 0, True),
        (1, r"$r_t \leftarrow$ normalize telemetry metrics in $x_t$ into unit compliance $[0, 1]$ using cost-benefit bounds", r"$\triangleright$ direction-aware normalization", 0, False),
        (2, r"$w \leftarrow$ calculate Shannon entropy across active window and derive objective criteria weights", r"$\triangleright$ dynamic importance weighting", 0, False),
        (3, r"$D_t \leftarrow$ compute instantaneous direct observation trust as weighted sum: $\sum w_j \cdot r_{t,j}$", r"$\triangleright$ direct evidence score", 0, False),
        (0, "// Stage 2: Composite Trust Aggregation", "", 0, True),
        (4, r"$T_t \leftarrow$ aggregate direct trust $D_t$, historical memory $T_{t-1}$, and Layer-1 threat penalty $(1 - q_t)$", r"$\triangleright$ Eq. (3) bounded in $[0, 1]$", 0, False),
        (5, r"$T_{\mathrm{hist}} \leftarrow T_t$", r"$\triangleright$ persist trust memory", 0, False),
        (0, "// Stage 3: Holt Linear Trend Forecasting (Anticipatory Tracking)", "", 0, True),
        (6, r"$l_t \leftarrow \alpha_H \cdot T_t + (1 - \alpha_H) \cdot (l_{t-1} + b_{t-1})$", r"$\triangleright$ update smoothed level", 0, False),
        (7, r"$b_t \leftarrow \beta_H \cdot (l_t - l_{t-1}) + (1 - \beta_H) \cdot b_{t-1}$", r"$\triangleright$ update drift velocity", 0, False),
        (8, r"$T_{\mathrm{pred}} \leftarrow \mathrm{clip}(l_t + b_t,\ 0,\ 1)$", r"$\triangleright$ one-step-ahead forecast", 0, False),
        (0, "// Stage 4: Dynamic Decision Threshold Modulation via Sigmoid", "", 0, True),
        (9, r"$\tau_t \leftarrow \tau_{\mathrm{base}} + (\tau_{\max} - \tau_{\mathrm{base}}) \cdot \mathrm{sigmoid}(\kappa \cdot (q_t - \mu))$", r"$\triangleright$ adaptive scrutiny via Sigmoid", 0, False),
        (0, "// Stage 5: Dual-Threshold Asymmetric Hysteresis Controller", "", 0, True),
        (10, r"$\mathbf{if}\ T_{\mathrm{pred}} < (\tau_t - 0.08)\ \mathbf{then}$", r"$\triangleright$ demotion condition", 0, False),
        (11, r"$s \leftarrow \min(s + 1,\ \mathrm{QUARANTINE});\ c \leftarrow 0$", r"$\triangleright$ instantaneous downgrade", 1, False),
        (12, r"$\mathbf{else\ if}\ T_{\mathrm{pred}} > (\tau_t + 0.08)\ \mathbf{then}$", r"$\triangleright$ promotion candidate", 0, False),
        (13, r"$c \leftarrow c + 1$", r"$\triangleright$ increment clean epochs", 1, False),
        (14, r"$\mathbf{if}\ c \geq 3\ \mathbf{and}\ s > \mathrm{TRUSTED}\ \mathbf{then}$", r"$\triangleright$ 3-epoch confirmation", 1, False),
        (15, r"$s \leftarrow \max(s - 1,\ \mathrm{TRUSTED});\ c \leftarrow 0$", r"$\triangleright$ promote state", 2, False),
        (16, r"$\mathbf{end\ if}$", "", 1, False),
        (17, r"$\mathbf{else}$", r"$\triangleright$ inside deadband $[-0.08, +0.08]$", 0, False),
        (18, r"$c \leftarrow 0$", r"$\triangleright$ maintain state; suppress chattering", 1, False),
        (19, r"$\mathbf{end\ if}$", "", 0, False),
        (0, "// Stage 6: Attribute-Based Access Control (ABAC) Enforcement", "", 0, True),
        (20, r"update persistent device cache with $\{T_{\mathrm{hist}}, l_t, b_t, c, s\}$", r"$\triangleright$ commit for next epoch", 0, False),
        (21, r"$\mathbf{if}\ s = \mathrm{TRUSTED}\ \mathbf{then}\ \mathbf{return}\ \mathrm{PERMIT}$", r"$\triangleright$ full operational access", 0, False),
        (22, r"$\mathbf{else\ if}\ s = \mathrm{WARNING}\ \mathbf{then}\ \mathbf{return}\ \mathrm{RESTRICT}$", r"$\triangleright$ read-only telemetry; actuation denied", 0, False),
        (23, r"$\mathbf{else}\ \mathbf{return}\ \mathrm{DENY}$", r"$\triangleright$ isolate in quarantine sandbox", 0, False),
    ]
    draw_algorithm_canvas(12.0, 8.0, title, req, lines, "algorithm2_layer2_full")


# ======================================================================
# 2. Algorithm 3: Scenario 1 (First Join / Onboarding)
# ======================================================================
def render_algorithm_3():
    title = r"$\mathbf{Algorithm\ 3}$ Scenario 1: First Join of IoT Device (New Device Onboarding Workflow)"
    req = r"$\mathbf{Require:}$ newly discovered device $d$, initial telemetry window $W_0$, baseline parameters"
    lines = [
        (0, "// Stage 1: Identity Query & Prior Trust Initialization", "", 0, True),
        (1, r"$\mathbf{if}\ \mathrm{device}\ d\ \mathrm{has\ no\ prior\ record\ in\ edge\ registry\ (cold\text{-}start)\ then}$", r"$\triangleright$ unobserved entity", 0, False),
        (2, r"initialize neutral uninformed prior trust: $T_{\mathrm{hist}} \leftarrow 0.50$", r"$\triangleright$ neutral baseline", 1, False),
        (3, r"initialize Holt tracker: level $l \leftarrow 0.50$, trend velocity $b \leftarrow 0.0$", r"$\triangleright$ zero initial momentum", 1, False),
        (4, r"reset clean epoch confirmation counter: $c \leftarrow 0$", "", 1, False),
        (5, r"assign provisional onboarding security state: $s_0 \leftarrow \mathrm{TRUSTED}$", r"$\triangleright$ allow onboarding handshake", 1, False),
        (6, r"establish dynamic decision threshold baseline: $\tau_0 \leftarrow 0.50$", r"$\triangleright$ active scrutiny threshold", 1, False),
        (7, r"$\mathbf{end\ if}$", "", 0, False),
        (0, "// Stage 2: Initial Gatekeeper Screening", "", 0, True),
        (8, r"$q_0 \leftarrow$ execute Layer-1 gatekeeper inference on initial telemetry $W_0$", r"$\triangleright$ inspect initial traffic", 0, False),
        (9, r"$\mathbf{if}\ q_0 \geq 0.90\ \mathbf{then}$", r"$\triangleright$ volumetric exploit detected", 0, False),
        (10, r"$s_0 \leftarrow \mathrm{QUARANTINE};\ \mathbf{return}\ \mathrm{DENY}$", r"$\triangleright$ abort onboarding immediately", 1, False),
        (11, r"$\mathbf{end\ if}$", "", 0, False),
        (0, "// Stage 3: Initial Provisioning & Continuous Layer-2 Tracking", "", 0, True),
        (12, r"$\mathbf{grant\ initial\ access\ entitlement:}\ \mathrm{PERMIT}$", r"$\triangleright$ enable legitimate configuration", 0, False),
        (13, r"commit initial state cache $\{T_{\mathrm{hist}}, l, b, c, s_0\}$ and begin Layer-2 monitoring", r"$\triangleright$ engage Algorithm 2", 0, False),
        (14, r"$\mathbf{if}\ \mathrm{subsequent\ epochs\ are\ confirmed\ benign}\ \mathbf{then}$", r"$\triangleright$ legitimate device behavior", 0, False),
        (15, r"$T_t\ \mathrm{rapidly\ consolidates\ to\ steady\text{-}state}\ (>0.85);\ \mathrm{maintain}\ \mathrm{PERMIT}$", r"$\triangleright$ trust consolidation within 3--5 epochs", 1, False),
        (16, r"$\mathbf{else}$", r"$\triangleright$ stealth attack or reconnaissance", 0, False),
        (17, r"$T_{\mathrm{pred}}\ \mathrm{drops\ below}\ (\tau_0 - 0.08);\ \mathrm{demote}\ s \leftarrow \mathrm{WARNING}\ (\mathrm{RESTRICT})$", r"$\triangleright$ rapid containment within 1--2 epochs", 1, False),
        (18, r"$\mathbf{end\ if}$", "", 0, False),
    ]
    draw_algorithm_canvas(12.0, 6.4, title, req, lines, "algorithm3_scenario1_first_join")


# ======================================================================
# 3. Algorithm 4: Scenario 2 (Rejoin After Normal Disconnection)
# ======================================================================
def render_algorithm_4():
    title = r"$\mathbf{Algorithm\ 4}$ Scenario 2: Rejoin After Normal Disconnection (Decayed Trust Restoration)"
    req = r"$\mathbf{Require:}$ reconnecting device $d$, cached trust profile $\{T_{\mathrm{cached}}, l_{\mathrm{cached}}\}$, timestamps $t_{\mathrm{disc}}, t_{\mathrm{reconn}}$"
    lines = [
        (0, "// Stage 1: Temporal Inactivity Measurement & Trust Decay", "", 0, True),
        (1, r"$\Delta t \leftarrow (t_{\mathrm{reconn}} - t_{\mathrm{disc}})\ \mathrm{in\ hours}$", r"$\triangleright$ elapsed offline duration", 0, False),
        (2, r"$T_{\mathrm{rejoin}} \leftarrow 0.50 + (T_{\mathrm{cached}} - 0.50) \cdot e^{-\lambda \cdot \Delta t}$", r"$\triangleright$ decay toward neutral prior ($\lambda = 0.05/\mathrm{h}$)", 0, False),
        (3, r"set Holt tracker: level $l \leftarrow T_{\mathrm{rejoin}}$, velocity $b \leftarrow 0.0$", r"$\triangleright$ reset momentum; retain level", 0, False),
        (4, r"reset clean epoch confirmation counter: $c \leftarrow 0$", "", 0, False),
        (0, "// Stage 2: Fast-Path Reconnection Evaluation", "", 0, True),
        (5, r"$\mathbf{if}\ \Delta t \leq 8.0\ \mathrm{hours\ and}\ T_{\mathrm{rejoin}} \geq 0.50\ \mathbf{then}$", r"$\triangleright$ brief sleep or routine reboot", 0, False),
        (6, r"$s_{\mathrm{rejoin}} \leftarrow \mathrm{TRUSTED};\ \mathbf{grant\ entitlement}\ \mathrm{PERMIT}$", r"$\triangleright$ zero-warmup latency resumption", 1, False),
        (7, r"$\mathbf{else}$", r"$\triangleright$ prolonged absence / potential tampering", 0, False),
        (8, r"$s_{\mathrm{rejoin}} \leftarrow \mathrm{WARNING};\ \mathbf{grant\ entitlement}\ \mathrm{RESTRICT}$", r"$\triangleright$ cautious re-audit probation", 1, False),
        (9, r"$\mathbf{end\ if}$", "", 0, False),
        (0, "// Stage 3: Ingestion & Trust Re-synchronization", "", 0, True),
        (10, r"ingest initial arrival telemetry and execute Layer-1 & Layer-2 evaluation", r"$\triangleright$ engage Algorithm 2", 0, False),
        (11, r"$\mathbf{if}\ \mathrm{arrival\ traffic\ is\ clean\ and\ compliant\ then}$", "", 0, False),
        (12, r"$T_t\ \mathrm{rapidly\ restores\ to\ pre\text{-}disconnection\ baseline}\ (>0.88)\ \mathrm{within\ 1\text{--}2\ clean\ epochs}$", r"$\triangleright$ fast re-synchronization", 1, False),
        (13, r"$\mathbf{else}$", r"$\triangleright$ compromised during offline period", 0, False),
        (14, r"$\mathrm{demote}\ s_{\mathrm{rejoin}}\ \mathrm{to\ RESTRICT\ or\ QUARANTINE\ on\ violation}$", r"$\triangleright$ immediate breach containment", 1, False),
        (15, r"$\mathbf{end\ if}$", "", 0, False),
        (16, r"$\mathbf{return}\ \mathrm{updated\ access\ decision\ from\ Layer\ 2}$", "", 0, False),
    ]
    draw_algorithm_canvas(12.0, 6.4, title, req, lines, "algorithm4_scenario2_rejoin_normal")


# ======================================================================
# 4. Algorithm 5: Scenario 3 (Rejoin After Blocking / DENY)
# ======================================================================
def render_algorithm_5():
    title = r"$\mathbf{Algorithm\ 5}$ Scenario 3: Rejoin After Blocking / DENY (Post-Quarantine Probation & Relapse Interception)"
    req = r"$\mathbf{Require:}$ rehabilitated device $d$ previously in QUARANTINE ($s=2$), incoming telemetry window $W_t$"
    lines = [
        (0, "// Stage 1: Administrative Clearance & Depressed Floor Initialization", "", 0, True),
        (1, r"verify administrative clearance flag and remove device $d$ from static blacklist", r"$\triangleright$ administrator re-admission", 0, False),
        (2, r"initialize depressed punitive trust baseline: $T_{\mathrm{hist}} \leftarrow 0.20$", r"$\triangleright$ penalty floor reflects prior breach", 0, False),
        (3, r"initialize Holt tracker: level $l \leftarrow 0.20$, trend velocity $b \leftarrow 0.0$", r"$\triangleright$ zero historical momentum", 0, False),
        (4, r"reset clean epoch confirmation counter: $c \leftarrow 0$", "", 0, False),
        (5, r"assign probationary security state: $s_t \leftarrow \mathrm{WARNING}$", r"$\triangleright$ strictly quarantined probation", 0, False),
        (6, r"$\mathbf{grant\ restricted\ entitlement:}\ \mathrm{RESTRICT}$", r"$\triangleright$ telemetry read-only; actuation blocked", 0, False),
        (0, "// Stage 2: Post-Reconnection Relapse Interception & Probation Lifecycle", "", 0, True),
        (7, r"$\mathbf{for}\ \mathrm{each\ subsequent\ monitoring\ epoch}\ t\ \mathbf{do}$", "", 0, False),
        (8, r"$q_t \leftarrow$ execute Layer-1 gatekeeper inference on $W_t$", r"$\triangleright$ continuous screening", 1, False),
        (9, r"$\mathbf{if}\ q_t \geq 0.90\ \mathbf{then}$", r"$\triangleright$ immediate relapse detected", 1, False),
        (10, r"$s_t \leftarrow \mathrm{QUARANTINE};\ c \leftarrow 0;\ \mathbf{return}\ \mathrm{DENY}$", r"$\triangleright$ instant hardware re-isolation", 2, False),
        (11, r"$\mathbf{end\ if}$", "", 1, False),
        (12, r"execute Layer-2 evaluation to compute predicted trust $T_{\mathrm{pred}}$ and threshold $\tau_t$", r"$\triangleright$ engage Algorithm 2", 1, False),
        (13, r"$\mathbf{if}\ T_{\mathrm{pred}} < (\tau_t - 0.08)\ \mathbf{then}$", r"$\triangleright$ behavioral violation / stealth relapse", 1, False),
        (14, r"$s_t \leftarrow \mathrm{QUARANTINE};\ c \leftarrow 0;\ \mathbf{return}\ \mathrm{DENY}$", r"$\triangleright$ zero-tolerance re-isolation", 2, False),
        (15, r"$\mathbf{else\ if}\ T_{\mathrm{pred}} > (\tau_t + 0.08)\ \mathbf{then}$", r"$\triangleright$ compliant probationary epoch", 1, False),
        (16, r"$c \leftarrow c + 1$", r"$\triangleright$ increment clean streak", 2, False),
        (17, r"$\mathbf{if}\ c \geq 3\ \mathbf{then}$", r"$\triangleright$ 3 clean epochs confirmed", 2, False),
        (18, r"$s_t \leftarrow \mathrm{TRUSTED};\ c \leftarrow 0;\ \mathbf{return}\ \mathrm{PERMIT}$", r"$\triangleright$ graduate to full access", 3, False),
        (19, r"$\mathbf{else}$", "", 2, False),
        (20, r"$\mathrm{maintain}\ s_t \leftarrow \mathrm{WARNING};\ \mathbf{return}\ \mathrm{RESTRICT}$", r"$\triangleright$ remain in probation", 3, False),
        (21, r"$\mathbf{end\ if}$", "", 2, False),
        (22, r"$\mathbf{else}$", r"$\triangleright$ inside deadband buffer", 1, False),
        (23, r"$\mathrm{maintain}\ s_t \leftarrow \mathrm{WARNING};\ c \leftarrow 0;\ \mathbf{return}\ \mathrm{RESTRICT}$", r"$\triangleright$ probation continues; reset streak", 2, False),
        (24, r"$\mathbf{end\ if}$", "", 1, False),
        (25, r"$\mathbf{end\ for}$", "", 0, False),
    ]
    draw_algorithm_canvas(12.0, 7.8, title, req, lines, "algorithm5_scenario3_rejoin_blocked")


if __name__ == "__main__":
    print("[Rendering All Algorithms as BMP & PNG]...")
    render_algorithm_2()
    render_algorithm_3()
    render_algorithm_4()
    render_algorithm_5()
    print("[Complete] All algorithms successfully rendered!")
