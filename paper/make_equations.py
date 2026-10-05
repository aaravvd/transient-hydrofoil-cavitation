from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"


OUTPUT = Path(__file__).resolve().parent / "equations"
EQUATIONS = [
    r"\sigma=\frac{p_{\mathrm{out}}-p_v}{0.5\rho_l U_\infty^2},",
    r"\nabla\!\cdot\!\mathbf{U}=0,",
    r"\frac{\partial(\rho_m\mathbf{U})}{\partial t}+\nabla\!\cdot\!(\rho_m\mathbf{UU})=-\nabla p+\nabla\!\cdot\!\mathbf{\tau}_{\mathrm{eff}},",
    r"\frac{\partial\alpha_v}{\partial t}+\nabla\!\cdot\!(\alpha_v\mathbf{U})=\frac{\dot m_v}{\rho_v},",
    r"h_{l+1}=\mathrm{GELU}\!\left(W_lh_l+\mathcal{F}^{-1}\!\left(R_l\mathcal{F}(h_l)\right)\right),",
    r"\widehat y_q(\mathbf{x})=\frac{\mathbf{b}_q^{\mathrm{T}}\mathbf{t}_q(\mathbf{x})}{\sqrt{48}}+c_q,",
    r"\mathcal{L}=\mathcal{L}_{\mathrm{sup}}+10^{-3}\!\left[\mathrm{MSE}(r_c)+\mathrm{MSE}(r_x)+\mathrm{MSE}(r_y)+\mathrm{MSE}(r_\alpha)\right],",
    r"r_c=\frac{\partial U_x}{\partial x}+\frac{\partial U_y}{\partial y},",
    r"r_i=\frac{\partial(\rho_m U_i)}{\partial t}+\frac{\partial(\rho_m U_iU_j)}{\partial x_j}+\frac{\partial p}{\partial x_i}-\mu_{\mathrm{eff}}\nabla^2U_i,\quad i\in\{x,y\},",
    r"r_\alpha=\frac{\partial\alpha_v}{\partial t}+\nabla\!\cdot\!(\alpha_v\mathbf{U})-S_{\mathrm{SS}}(p,\alpha_v),",
]

OUTPUT.mkdir(exist_ok=True)
for number, latex in enumerate(EQUATIONS, 1):
    path = OUTPUT / f"equation_{number}.png"
    figure = plt.figure(figsize=(5.8, 0.48), dpi=300)
    figure.patch.set_alpha(0)
    font_size = 13 if len(latex) > 100 else 15 if len(latex) > 70 else 17
    figure.text(0.47, 0.5, f"${latex}$", ha="center", va="center", fontsize=font_size)
    figure.text(0.98, 0.5, f"({number})", ha="right", va="center", fontsize=14)
    figure.savefig(path, dpi=300, transparent=True)
    plt.close(figure)
