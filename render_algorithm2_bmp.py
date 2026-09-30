import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# Configure Computer Modern LaTeX serif font matching standard IEEE/Q1 papers
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

def render_algorithm2():
    fig_w, fig_h = 11.5, 6.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Coordinates
    x_left = 0.015
    x_lineno = 0.018
    x_code = 0.050
    x_right = 0.985
    
    y_top = 0.970
    y_title = 0.932
    y_mid = 0.895
    y_req = 0.860
    
    line_dy = 0.038
    y_code_start = 0.815
    
    # Top rule (thick)
    ax.plot([x_left, x_right], [y_top, y_top], color="black", lw=1.6)
    
    # Title
    ax.text(
        x_left, y_title,
        r"$\mathbf{Algorithm\ 2}$ Layer 2 — behavioral trust engine and ABAC policy enforcement (runs every epoch, per device)",
        fontsize=11.2, color="black", va="center"
    )
    
    # Mid rule (thin)
    ax.plot([x_left, x_right], [y_mid, y_mid], color="black", lw=0.75)
    
    # Require line
    req_text = r"$\mathbf{Require:}$ device $i$, threat score $q_t$, telemetry $x_t$, state cache $(s, c, T_{\mathrm{hist}}, l, b)$"
    ax.text(x_left, y_req, req_text, fontsize=9.8, color="black", va="center")
    
    # Lines of code (number, text, comment, indent_level)
    lines = [
        (1, r"$r_t \leftarrow \mathrm{normalize}\ x_t\ \mathrm{using\ cost\text{-}benefit\ bounds}$", r"$\triangleright$ direction-aware evidence", 0),
        (2, r"$w \leftarrow \mathrm{entropy\_weights}(r_t)$", r"$\triangleright$ Shannon entropy weighting", 0),
        (3, r"$T_t \leftarrow \mathrm{composite\_trust}(w, r_t, T_{\mathrm{hist}}, q_t)$", r"$\triangleright$ composite score (Eq. 3)", 0),
        (4, r"$l_t, b_t, T_{\mathrm{pred}} \leftarrow \mathrm{holt\_forecast}(T_t, l, b)$", r"$\triangleright$ Holt trend forecast (Eq. 4)", 0),
        (5, r"$\tau_t \leftarrow \mathrm{sigmoid\_threshold}(T_{\mathrm{pred}})$", r"$\triangleright$ adaptive threshold (Eq. 5)", 0),
        (6, r"$\mathbf{if}\ s < 2\ \mathbf{and}\ T_{\mathrm{pred}} < \tau_t - 0.08\ \mathbf{then}$", r"$\triangleright$ demote on confirmed violation", 0),
        (7, r"$s \leftarrow s + 1;\ c \leftarrow 0$", "", 1),
        (8, r"$\mathbf{else\ if}\ s > 0\ \mathbf{and}\ T_{\mathrm{pred}} > \tau_t + 0.08\ \mathbf{then}$", r"$\triangleright$ candidate promotion", 0),
        (9, r"$c \leftarrow c + 1$", "", 1),
        (10, r"$\mathbf{if}\ c \geq 3\ \mathbf{then}$", r"$\triangleright$ 3-epoch confirmation", 1),
        (11, r"$s \leftarrow s - 1;\ c \leftarrow 0$", "", 2),
        (12, r"$\mathbf{end\ if}$", "", 1),
        (13, r"$\mathbf{else}$", r"$\triangleright$ deadband: retain state", 0),
        (14, r"$c \leftarrow 0$", "", 1),
        (15, r"$\mathbf{end\ if}$", "", 0),
        (16, r"$\mathrm{persist\_cache}(i, s, c, T_t, l_t, b_t)$", r"$\triangleright$ commit state for next epoch", 0),
        (17, r"$\mathbf{if}\ s = 0\ \mathbf{then}\ \mathbf{return}\ \mathrm{permit}$", r"$\triangleright$ TRUSTED: full access", 0),
        (18, r"$\mathbf{if}\ s = 1\ \mathbf{then}\ \mathbf{return}\ \mathrm{restrict}$", r"$\triangleright$ WARNING: telemetry only", 0),
        (19, r"$\mathbf{return}\ \mathrm{deny}$", r"$\triangleright$ QUARANTINE: full block", 0),
    ]
    
    for i, (lineno, code_text, comment_text, indent) in enumerate(lines):
        y = y_code_start - i * line_dy
        # Line number
        ax.text(x_lineno, y, f"{lineno}:", fontsize=9.2, color="black", va="center")
        
        # Indented code
        indent_x = x_code + indent * 0.035
        ax.text(indent_x, y, code_text, fontsize=9.3, color="black", va="center")
        
        # Comment on the right
        if comment_text:
            ax.text(x_right, y, comment_text, fontsize=9.0, color="#212121", va="center", ha="right")
            
    # Bottom rule (thick)
    y_bottom = y_code_start - len(lines) * line_dy + 0.012
    ax.plot([x_left, x_right], [y_bottom, y_bottom], color="black", lw=1.6)
    
    plt.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02)
    
    out_dir = r"D:\PHD\Implementation\edgetrust_pipeline\outputs\figures"
    os.makedirs(out_dir, exist_ok=True)
    
    png_path = os.path.join(out_dir, "algorithm2_layer2.png")
    bmp_path = os.path.join(out_dir, "algorithm2_layer2.bmp")
    
    # Save high-res PNG first
    plt.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    
    # Convert and save as 24-bit RGB BMP (uncompressed bitmap)
    with Image.open(png_path) as img:
        rgb_img = img.convert("RGB")
        rgb_img.save(bmp_path, format="BMP")
        
    print(f"Successfully generated:")
    print(f"  BMP: {bmp_path} (Size: {os.path.getsize(bmp_path):,} bytes, Dimensions: {rgb_img.size})")
    print(f"  PNG: {png_path} (Size: {os.path.getsize(png_path):,} bytes)")

if __name__ == "__main__":
    render_algorithm2()
