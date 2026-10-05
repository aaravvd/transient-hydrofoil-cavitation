from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "paper/work/Initial-submission-US-version-Template.docx"
OUTPUT = ROOT / "paper/cavitation_surrogate_southeastcon.docx"
FIGURES = ROOT / "paper/figures"
EQUATIONS = ROOT / "paper/equations"


def clear_template(document):
    for paragraph in list(document.paragraphs[13:]):
        paragraph._element.getparent().remove(paragraph._element)
    for table in list(document.tables):
        table._element.getparent().remove(table._element)


def shade(cell, fill="D9E2F3"):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def heading(document, text, level=1):
    paragraph = document.add_paragraph(text, style=f"Heading {level}")
    paragraph.paragraph_format.keep_with_next = True
    paragraph.paragraph_format.keep_together = True
    return paragraph


def body(document, text):
    return document.add_paragraph(text, style="Body Text")


def equation(document, latex, number):
    path = EQUATIONS / f"equation_{number}.png"
    if not path.exists():
        raise FileNotFoundError(f"Run paper/make_equations.py first: {path}")
    paragraph = document.add_paragraph(style="equation")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(str(path), width=Inches(3.25))


def figure(document, filename, caption, width=3.42):
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(FIGURES / filename), width=Inches(width))
    caption_paragraph = document.add_paragraph(caption, style="figure caption")
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def full_width_figure(document, filename, caption, width=7.0):
    wide_section = document.add_section(WD_SECTION.CONTINUOUS)
    set_columns(wide_section, 1)
    figure(document, filename, caption, width=width)
    column_section = document.add_section(WD_SECTION.CONTINUOUS)
    set_columns(column_section, 2)


def table(document, title, headers, rows, widths, page_break_before=False, column_break_before=False):
    if column_break_before:
        document.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)
    caption = document.add_paragraph(title, style="table head")
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.keep_with_next = True
    caption.paragraph_format.page_break_before = page_break_before
    result = document.add_table(rows=1, cols=len(headers))
    result.alignment = WD_TABLE_ALIGNMENT.CENTER
    result.autofit = False
    for index, header in enumerate(headers):
        cell = result.rows[0].cells[index]
        cell.text = header
        shade(cell)
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for values in rows:
        cells = result.add_row().cells
        for index, value in enumerate(values):
            cells[index].text = str(value)
    for row_index, row in enumerate(result.rows):
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.keep_with_next = row_index < len(result.rows) - 1
                for run in paragraph.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(7)
    for column, width in zip(result.columns, widths):
        column.width = Inches(width)
    grid = result._tbl.tblGrid
    for grid_column, width in zip(grid.gridCol_lst, widths):
        grid_column.set(qn("w:w"), str(int(width * 1440)))


def set_columns(section, count):
    columns = section._sectPr.find(qn("w:cols"))
    if columns is None:
        columns = OxmlElement("w:cols")
        section._sectPr.append(columns)
    columns.set(qn("w:num"), str(count))
    columns.set(qn("w:space"), "360")


doc = Document(TEMPLATE)
clear_template(doc)
set_columns(doc.sections[-1], 2)
doc.paragraphs[0].text = "Benchmarking Neural Surrogates for Transient Hydrofoil Cavitation Forecasting"
doc.paragraphs[0].style.font.name = "Times New Roman"
doc.paragraphs[0].style.font.size = Pt(24)
for run in doc.paragraphs[0].runs:
    run.font.name = "Times New Roman"
    run.font.size = Pt(24)
doc.paragraphs[1].text = ""
for index in range(3, 10):
    doc.paragraphs[index].text = ""
abstract_text = (
    "Transient cavitation is governed by moving vapor structures rather than a pressure threshold, "
    "yet neural flow surrogates are often assessed only by one-step field error. This paper presents a controlled "
    "benchmark of CNN-U-Net, Fourier neural operator, DeepONet, and a transient multiphase physics-informed neural network (PINN) "
    "for autoregressive hydrofoil-cavitation prediction. Thirty new OpenFOAM trajectories span five NACA families, "
    "two angles of attack, and three cavitation numbers; geometry families are separated across training, validation, "
    "and final testing. Each architecture is trained with three seeds and compared with persistence using pressure, "
    "velocity, vapor-fraction, cavity-overlap, rollout, and runtime metrics. On the unseen NACA 2415 family, the "
    "PINN gives the lowest learned one-step pressure error (1.08 kPa) and highest mean learned "
    "20-step cavity intersection over union (0.519). Persistence remains strongest for several slowly varying field "
    "metrics, exposing a limitation that one-step rankings conceal. Learned surrogates complete 20-step forecasts "
    "275–885 times faster than the computational-fluid-dynamics workflow, although spatial checks show that the "
    "reference cavity and force statistics remain mesh-sensitive."
)
doc.paragraphs[11].clear()
abstract_label = doc.paragraphs[11].add_run("Abstract")
abstract_label.italic = True
doc.paragraphs[11].add_run("—" + abstract_text)
doc.paragraphs[12].text = (
    "Keywords—cavitation, hydrofoil, neural operator, transient flow, multiphase computational fluid dynamics"
)

heading(doc, "Introduction")
body(doc, "Cavitation changes hydrofoil loading, efficiency, vibration, and erosion through the growth and collapse of vapor structures. Resolving those structures requires transient multiphase computational fluid dynamics (CFD), which becomes expensive when many geometries or operating points must be screened. Neural surrogates can reduce that repeated cost, but useful forecasts must preserve cavity topology and remain stable when their own predictions are fed back over time; low average field error alone is not enough.")
body(doc, "Convolutional networks have been used to reconstruct cavitating fields [1], while Fourier neural operators (FNOs) have recently been combined with U-Net structure for steady and unsteady hydrofoil cavitation [2]. Reduced-order and super-resolution approaches provide related alternatives [3], [4]. Physics-informed neural networks (PINNs) offer a complementary route by penalizing governing-equation residuals during training [10]. These studies establish the promise of data-driven cavitation modeling, but they do not provide a common autoregressive comparison of convolutional, spectral, branch–trunk, and multiphase physics-informed models against a no-change baseline on an untouched hydrofoil family. That comparison matters because a model can look accurate over one small time step and still drift during a practically useful rollout.")
body(doc, "This work makes three contributions. First, it introduces a new transient CFD dataset in which complete geometry families are assigned to training, validation, and final-test sets. Second, it benchmarks four neural model families under identical targets, seeds, validation-based stopping rules, and 20-step autoregressive tests, with persistence reported beside every learned model. Third, it connects predictive accuracy to cavity overlap and computational cost while documenting mesh and time-step sensitivity in the CFD reference. The task is forecasting actual vapor volume fraction, not inferring cavitation risk from pressure.")

heading(doc, "Related Work")
heading(doc, "Field reconstruction and neural operators", 2)
body(doc, "Hydrofoil surrogates first established that convolutional encoder-decoder models could recover pressure and velocity fields from geometry and operating conditions at far lower cost than Reynolds-averaged Navier–Stokes simulation [12]. Cavitation-specific studies then extended image-to-field prediction to vapor fraction. Noda and Okabayashi built a data-driven cavitation framework from CFD snapshots [1], and Liu and Park trained a DeepCFD U-Net across non-cavitating, sheet-cavitating, and detached-cavity cases [13]. Zhang et al. later combined U-Net features with a Fourier neural operator for steady and unsteady hydrofoil cavitation [2]. Together, these studies show that local multiscale features and global spectral coupling can both recover cavitating-flow structure.")
body(doc, "Other formulations solve related but distinct problems. Reduced-order modeling compresses cavity dynamics into a lower-dimensional state [3], while super-resolution methods reconstruct pressure fields from sparse or coarse observations [4]. DeepONet instead separates operating-condition features from spatial coordinates through branch and trunk networks [9], and Fourier neural operators learn resolution-aware mappings in spectral space [8]. Such choices impose different spatial and parametric biases. Reported performance cannot be compared directly, however, because prior studies use different geometries, train-test splits, forecast intervals, and definitions of field accuracy. Most emphasize reconstructed snapshots rather than recursive prediction in which model error becomes part of the next input.")
heading(doc, "Physics-informed and temporal prediction", 2)
body(doc, "Physics-informed learning provides a second route: governing-equation residuals constrain predictions between or alongside labeled states [10]. Ouyang et al. developed a chain-style PINN for three-dimensional cavitation-field reconstruction [14]. Gao et al. used a rotation-equivariant hypergraph network to forecast cavitation dynamics around a NACA 0012 hydrofoil and highlighted the difficulty of long-term prediction across cavity events [15]. More recently, Chen et al. embedded Navier–Stokes and Schnerr–Sauer constraints in a correction network that improved a baseline RANS solution against experimental observations [16]. These results motivate physical regularization and temporal evaluation, but reconstruction, correction of an existing CFD field, and free autoregressive rollout are not interchangeable tasks.")
body(doc, "Cavitation also makes metric selection consequential. Pressure and velocity errors measure global field agreement, whereas vapor-fraction error and cavity-mask overlap measure the moving phase boundary that drives the application. A one-step predictor can score well simply because adjacent CFD snapshots differ little; persistence must therefore be reported to establish whether learning adds information. Once predictions are recursively fed back, small local errors can alter cavity position, shape, and subsequent evolution. Seed variability, error growth with horizon, and performance by cavitation regime are consequently part of model assessment rather than optional diagnostics.")
heading(doc, "Gap and study position", 2)
body(doc, "The unresolved question is evaluative rather than architectural: the literature does not show whether convolutional, spectral, branch–trunk, and multiphase physics-informed surrogates retain their one-step ranking when recursively forecasting an unseen hydrofoil family. This study holds the CFD data, geometry split, targets, optimizer protocol, seeds, and rollout procedure fixed; includes a no-change persistence baseline; and reports field errors, cavity topology, residual components, stability across forecast horizon, condition-wise behavior, and workflow runtime. The contribution is a controlled account of where each modeling assumption helps or fails, not another single-architecture accuracy claim.")

heading(doc, "Cavitating-Flow Dataset")
heading(doc, "Governing model and numerical setup", 2)
body(doc, "The reference trajectories were generated with OpenFOAM v2312 interPhaseChangeFoam. The homogeneous-mixture solver advances incompressible two-phase mass and momentum balances with a volume-of-fluid interface treatment, k–omega shear-stress-transport turbulence closure [5], and the Schnerr–Sauer cavitation mass-transfer model [6]. Vaporization and condensation are therefore represented through transported vapor fraction rather than a post-processed pressure criterion. The cavitation number is")
equation(doc, r"\sigma=\frac{p_{\mathrm{out}}-p_v}{\tfrac{1}{2}\rho_l U_\infty^2},", 1)
body(doc, "where p_out is outlet pressure, p_v = 2300 Pa is vapor pressure, rho_l = 997 kg/m³ is liquid density, and U_inf = 10 m/s. A 0.1 m chord and liquid kinematic viscosity of 10⁻⁶ m²/s give Reynolds number 10⁶. The two-dimensional domain extends from -2c to 5c streamwise and -2c to 2c vertically. Each trajectory covers 0.03 s, or three convective times, with an adaptive time step limited to Courant number 0.5; fields are written every 0.001 s.")
body(doc, "For mixture density rho_m = alpha_l rho_l + alpha_v rho_v and velocity U, the solved conservation equations may be summarized as")
equation(doc, r"\nabla\!\cdot\!\mathbf{U}=0,", 2)
equation(doc, r"\frac{\partial(\rho_m\mathbf{U})}{\partial t}+\nabla\!\cdot\!(\rho_m\mathbf{UU})=-\nabla p+\nabla\!\cdot\!\boldsymbol{\tau}_{\mathrm{eff}},", 3)
equation(doc, r"\frac{\partial\alpha_v}{\partial t}+\nabla\!\cdot\!(\alpha_v\mathbf{U})=\frac{\dot m_v}{\rho_v},", 4)
body(doc, "where alpha_l and alpha_v are liquid and vapor volume fractions, rho_v is vapor density, tau_eff contains laminar and modeled turbulent stresses, and m_dot_v is the net Schnerr–Sauer vapor mass source. The source changes sign between evaporation and condensation according to local pressure relative to p_v. These equations define the CFD data generator; the neural models learn its discrete snapshot evolution rather than directly solving (2)–(4).")

heading(doc, "Cases, representation, and splits", 2)
body(doc, "Five four-digit NACA families are evaluated at angles of attack 6 and 9 degrees and cavitation numbers 0.8, 1.2, and 1.8, producing 30 trajectories. NACA 0012, 2412, and 4412 form the training set; NACA 0015 is used only for validation; and NACA 2415 remains untouched until final evaluation. Entire trajectories stay within one split. After the first convective time is discarded as warm-up, consecutive snapshots provide 360 training, 120 validation, and 120 test transitions. Cell-centered solutions are interpolated onto a fixed 128 x 64 grid.")
table(doc, "DATASET AND SPLIT DESIGN", ["Split", "Families", "Cases", "Transitions"], [["Train", "0012, 2412, 4412", "18", "360"], ["Validation", "0015", "6", "120"], ["Final test", "2415", "6", "120"]], [0.75, 1.1, 0.55, 0.75])
body(doc, "At time t, the network input contains U_x, U_y, pressure, vapor fraction, normalized coordinates, a fluid mask, angle of attack, cavitation number, and normalized time. The target contains U_x, U_y, pressure, and vapor fraction at t + 0.001 s. Outputs are standardized with training-set statistics; the vapor channel receives five times the supervised loss weight.")

heading(doc, "Reference-solution checks", 2)
body(doc, "All 30 production runs completed, and vapor fraction remained within 10⁻⁴ of its physical bounds. A NACA 2412 case at 9 degrees and σ = 1.2 was repeated on 1,910, 3,555, 7,077, and 13,825-cell meshes. Halving the medium-grid Courant limit changed mean lift, drag, and cavity area by 0.7%, 0.9%, and 1.4%, respectively. The fine-to-extra-fine changes were 6.0%, 4.8%, and 4.8%, so the 7,077-cell grid is a cost-conscious reference, not a grid-independent truth. Figure 1 reports the full trend.")
figure(doc, "convergence.png", "Grid sensitivity of mean lift, drag, and normalized cavity area for NACA 2412 at 9 degrees and σ = 1.2.")

heading(doc, "Neural Forecasting Benchmark")
heading(doc, "Architectures", 2)
body(doc, "CNN-U-Net uses three encoder scales, bilinear decoding, and skip connections [7]. The FNO lifts the input to 32 channels and applies four spectral layers retaining 12 vertical and 20 streamwise modes [8]. DeepONet uses a convolutional branch encoder for the current field and a coordinate trunk network, whose rank-48 inner products produce the four outputs [9]. The fourth model is a transient multiphase PINN with a four-layer pointwise backbone. All models receive the same ten inference channels and predict the same four next-state fields.")
equation(doc, r"h_{l+1}=\operatorname{GELU}\!\left(W_lh_l+\mathcal{F}^{-1}\!\left(R_l\mathcal{F}(h_l)\right)\right),", 5)
body(doc, "expresses one FNO layer, where F and F⁻¹ are the forward and inverse Fourier transforms, R_l contains learned spectral weights, and W_l is a learned local transformation. For DeepONet, branch features b_q from the current field are paired with coordinate-dependent trunk features t_q to predict output channel q:")
equation(doc, r"\widehat y_q(\mathbf{x})=\frac{\mathbf{b}_q^{\mathsf T}\mathbf{t}_q(\mathbf{x})}{\sqrt{48}}+c_q,", 6)
body(doc, "Here c_q is a learned channel bias. The PINN minimizes")
equation(doc, r"\mathcal{L}=\mathcal{L}_{\mathrm{sup}}+10^{-3}\!\left[\mathrm{MSE}(r_c)+\mathrm{MSE}(r_x)+\mathrm{MSE}(r_y)+\mathrm{MSE}(r_\alpha)\right],", 7)
equation(doc, r"r_c=\frac{\partial U_x}{\partial x}+\frac{\partial U_y}{\partial y},", 8)
equation(doc, r"r_i=\frac{\partial(\rho_m U_i)}{\partial t}+\frac{\partial(\rho_m U_iU_j)}{\partial x_j}+\frac{\partial p}{\partial x_i}-\mu_{\mathrm{eff}}\nabla^2U_i,\quad i\in\{x,y\},", 9)
equation(doc, r"r_\alpha=\frac{\partial\alpha_v}{\partial t}+\nabla\!\cdot\!(\alpha_v\mathbf{U})-S_{\mathrm{SS}}(p,\alpha_v),", 10)
body(doc, "where L_sup is the masked, channel-weighted next-state mean-square error; r_c, r_x, and r_y are continuity and conservative mixture-momentum residuals; and r_alpha is the vapor-transport residual. S_SS is the Schnerr–Sauer source evaluated from predicted pressure and vapor fraction using bubble density 1.6 x 10^13 m^-3, nuclei diameter 2 micrometers, and unit evaporation and condensation coefficients, matching the CFD cases. Predictions are denormalized before residual evaluation, temporal derivatives use the current and predicted states separated by 0.001 s, and spatial derivatives use dimensional grid spacing. The effective viscosity combines molecular mixture viscosity with saved OpenFOAM RANS eddy viscosity. This is training-only CFD-derived information: it is used only to evaluate the PINN momentum residual, is unavailable to the other models, and is not a PINN inference input. Residuals are nondimensionalized by characteristic continuity, momentum, and phase scales and evaluated only where the finite-difference stencil remains in the fluid. Parameter counts are 267,124 for CNN-U-Net, 989,988 for FNO, 61,116 for DeepONet, and 34,948 for the PINN.")
table(doc, "MODEL CAPACITY AND SPATIAL INDUCTIVE BIAS", ["Model", "Parameters", "Field coupling", "Latent size"], [["CNN-U-Net", "267,124", "Local + multiscale", "24 base"], ["FNO", "989,988", "Global spectral", "32 x 4"], ["DeepONet", "61,116", "Branch–trunk", "rank 48"], ["PINN", "34,948", "Pointwise + PDE", "128"]], [0.8, 0.75, 1.15, 0.65])

heading(doc, "Training and evaluation", 2)
body(doc, "Each architecture is trained from seeds 7, 17, and 29 with AdamW, learning rate 0.002, weight decay 10⁻⁵, batch size 8, and a maximum of 20 epochs. Training stops after five epochs without validation improvement, after a six-epoch minimum; PINN gradients are clipped to unit norm. Best checkpoints occur at epochs 18–20 for CNN-U-Net, 9–20 for FNO, 18–20 for DeepONet, and 13–18 for the PINN. Mean training times are 5.79, 4.00, 1.89, and 2.72 min, respectively. The lowest-validation-loss checkpoint is evaluated once on the final-test family. This protocol removes the visibly undertrained six-epoch comparison, although it does not equalize parameter count or exhaustively tune each architecture.")
figure(doc, "training_curves.png", "Training and validation loss for all architectures and seeds. Checkpoints are selected by validation loss; the final-test family is not used for selection.")
body(doc, "Metrics include root-mean-square error (RMSE) for velocity, pressure, and vapor fraction; vapor-mask intersection over union (IoU) at a vapor-fraction threshold of 0.5; absolute cavity-volume error on the common grid; and a 20-step autoregressive rollout. Persistence, which copies the current field forward, tests whether a learned model improves on the strong short-horizon assumption that the flow barely changes. Reported uncertainty is the sample standard deviation across three training seeds.")

heading(doc, "Results")
heading(doc, "One-step prediction", 2)
body(doc, "Table III shows that no architecture dominates every target. The PINN has the lowest learned velocity and pressure errors, FNO gives the smallest vapor-fraction RMSE, and CNN-U-Net has the highest mean one-step vapor IoU. DeepONet remains weaker despite the longer training protocol. Persistence nevertheless achieves the lowest pressure and cavity-volume errors, confirming that a 0.001 s prediction interval makes local one-step scores deceptively easy.")
table(doc, "FINAL-TEST ONE-STEP RESULTS (MEAN ± SAMPLE SD)", ["Model", "U RMSE", "p RMSE\n(kPa)", "α RMSE", "IoU"], [["CNN-U-Net", "0.207", "2.26 ± 1.15", "0.0141", "0.894 ± .019"], ["FNO", "0.180", "1.67 ± 0.07", "0.0137", "0.887 ± .003"], ["DeepONet", "0.916", "5.09 ± 0.53", "0.0445", "0.468 ± .050"], ["PINN", "0.171", "1.08 ± 0.08", "0.0180", "0.878 ± .001"], ["Persistence", "0.183", "0.70", "0.0190", "0.871"]], [0.78, 0.6, 0.8, 0.66, 0.69])
body(doc, "The representative unseen-family case in Fig. 3 clarifies those aggregate scores. CNN-U-Net and FNO recover the large attached cavity but soften or texture pressure structure. The PINN closely preserves pressure and the cavity footprint while slightly underestimating one-step vapor overlap; DeepONet under-resolves both. Table IV separates the dimensionless test residual by equation. The PINN's largest advantage is vapor transport; continuity and momentum differences are smaller and do not uniformly favor it. This componentwise result is more informative than the combined residual alone.")
table(doc, "FINAL-TEST GOVERNING RESIDUALS (MEAN ± SAMPLE SD)", ["Model", "Continuity", "x-mom.", "y-mom.", "Vapor"], [["CNN-U-Net", ".208 ± .049", ".379 ± .114", ".244 ± .101", "70.2 ± 29.9"], ["FNO", ".260 ± .016", ".473 ± .020", ".215 ± .024", "52.5 ± 5.8"], ["DeepONet", ".088 ± .006", ".783 ± .037", ".404 ± .039", "587 ± 205"], ["PINN", ".178 ± .007", ".322 ± .002", ".063 ± .004", "7.86 ± .84"]], [0.78, 0.75, 0.66, 0.66, 0.72])
full_width_figure(doc, "field_comparison.png", "OpenFOAM reference and seed-7 predictions for NACA 2415 at 9 degrees and σ = 1.2. Rows show pressure and vapor fraction on the shared physical domain.")

rollout_heading = heading(doc, "Autoregressive rollout", 2)
body(doc, "The 20-step test changes the ranking (Table V). The PINN provides the best mean learned cavity IoU and markedly lower seed-to-seed dispersion than the other learned models. Its IoU exceeds persistence because an unchanged mask cannot follow cavity motion, although persistence still leads pressure, vapor-fraction, and cavity-area errors. CNN-U-Net and FNO are seed-sensitive, while DeepONet remains unstable despite the longer training protocol. These failures would be hidden by one-step reporting alone, making moving-cavity overlap—not short-horizon pressure—the primary surrogate result.")
table(doc, "TWENTY-STEP AUTOREGRESSIVE RESULTS", ["Model", "p RMSE\n(kPa)", "α RMSE", "IoU", "Cavity-area\nMAE"], [["CNN-U-Net", "13.6 ± 7.7", ".119 ± .113", ".424 ± .217", "50.8 ± 32.6"], ["FNO", "13.5 ± 7.5", ".072 ± .027", ".362 ± .183", "65.0 ± 30.9"], ["DeepONet", "43.5 ± 50.1", ".081 ± .033", ".087 ± .058", "104.2 ± 26.4"], ["PINN", "10.7 ± 4.0", ".0567 ± .0250", ".519 ± .003", "37.1 ± 6.5"], ["Persistence", "3.35", ".0490", ".447", "27.6"]], [0.65, 0.75, 0.67, 0.7, 0.65])
body(doc, "Figure 4 resolves this aggregate by forecast step. FNO initially gives the strongest cavity overlap, with IoU 0.943 at step 1 and 0.834 at step 5, but the PINN becomes strongest by step 10 and retains IoU 0.338 at step 20, compared with 0.242 for CNN-U-Net, 0.148 for FNO, and 0.199 for persistence. Pressure behaves differently: persistence remains best throughout because pressure changes slowly over this 0.02 s horizon. The widening gap between pressure RMSE and cavity IoU shows why neither metric can substitute for the other.")
figure(doc, "horizon_curves.png", "Error growth over the final-test rollout. Lines show means across six NACA 2415 conditions; bands show sample standard deviation across three seeds for learned models.")
body(doc, "Condition-wise results in Table VI show that the ranking also depends on cavity regime. For the strongly cavitating σ = 0.8 cases, the PINN has the highest mean case-level IoU, followed by FNO and persistence. At σ = 1.2, persistence slightly exceeds CNN-U-Net and the PINN, while FNO deteriorates more sharply. All σ = 1.8 reference masks are empty at the 0.1 threshold, so their IoU is undefined in a physical sense and they are excluded from Table VI rather than scored as failed cavity overlap. Across the cavity-positive cases, every method performs better at 9 degrees than at 6 degrees because the larger vapor region makes overlap less sensitive to a few boundary cells. This stratification prevents the aggregate IoU from being mistaken for uniform performance across operating conditions.")
table(doc, "CAVITY-POSITIVE 20-STEP IoU BY TEST CONDITION", ["Model", "σ = 0.8", "σ = 1.2", "6 deg.", "9 deg."], [["CNN-U-Net", ".420", ".229", ".176", ".472"], ["FNO", ".480", ".105", ".194", ".391"], ["DeepONet", ".088", ".002", ".018", ".072"], ["PINN", ".526", ".161", ".201", ".486"], ["Persistence", ".441", ".208", ".211", ".438"]], [0.82, 0.68, 0.68, 0.68, 0.68])
figure(doc, "metric_comparison.png", "One-step and 20-step final-test metrics. Error bars show sample standard deviation over three seeds; persistence is deterministic.")

heading(doc, "Computational cost", 2)
body(doc, "A 30-step CFD trajectory requires 83.4 ± 17.6 s on the test workstation. Batched neural inference requires 4.71–15.16 ms per frame, corresponding to 275x, 450x, 885x, and 770x speedups for 20-step CNN-U-Net, FNO, DeepONet, and PINN forecasts. These are platform-specific workflow measurements, not hardware-neutral algorithmic speedups: CFD ran in an amd64 Docker container under host emulation, whereas neural inference used the host PyTorch backend. DeepONet also illustrates that speed without rollout fidelity is not useful.")

heading(doc, "Discussion")
body(doc, "The results separate three questions that are often collapsed into one score. First, can a model interpolate the next snapshot? Second, does it outperform persistence? Third, does it remain credible after repeated feedback? The answer depends on target and horizon. Persistence wins one-step pressure and remains strongest on several 20-step global errors, so this benchmark does not claim that learning improves slowly varying fields. Its useful distinction is moving-cavity topology: the PINN achieves the strongest learned rollout IoU with little seed variation, while CNN-U-Net and FNO accumulate larger, less consistent errors. DeepONet improves substantially with longer training but its branch–trunk encoding remains poorly matched to repeated feedback.")
body(doc, "The study is intentionally focused: 30 trajectories cover five related four-digit NACA families at one Reynolds number. CFD, not experiment, provides the reference, and the grid study shows non-negligible force and cavity-area sensitivity; surrogate error therefore measures agreement with this numerical workflow rather than physical truth beyond it. The PINN is also data-plus-physics, not equation-only, because its training residual uses saved CFD eddy viscosity. Broader Reynolds-number and geometry coverage, native-platform CFD timing, longer shedding records, and experimental cavity/load measurements are needed before design deployment.")

heading(doc, "Conclusion")
body(doc, "This study benchmarks four neural architectures for actual vapor-fraction forecasting on a new family-separated hydrofoil cavitation dataset. The central result is not a universal winner. The data-plus-physics PINN gives the lowest learned one-step pressure error, the lowest vapor-transport residual, and the strongest mean learned rollout IoU; FNO gives the lowest one-step vapor-fraction RMSE, while CNN-U-Net gives the highest one-step cavity IoU. Persistence remains strongest on slowly varying pressure and global errors, proving that longer-horizon cavity topology must be evaluated explicitly. Platform-specific measurements show substantial workflow acceleration, while the mesh-sensitive CFD reference defines the present accuracy ceiling. Future work will extend the benchmark across Reynolds numbers and geometry classes and validate longer cavity-shedding records against experiments.")

availability = doc.add_paragraph(style="Body Text")
availability.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
availability.paragraph_format.first_line_indent = Inches(0)
availability.add_run("Data and Code Availability—").italic = True
availability.add_run("Processed fields and raw OpenFOAM cases are archived at Zenodo (doi: 10.5281/zenodo.23148648). Generation, training, and evaluation code is available at github.com/aaravvd/transient-hydrofoil-cavitation.")

doc.add_paragraph("References", style="Heading 5")
references = [
    "[1] H. Noda and S. Okabayashi, ‘Development of framework of data-driven cavitation model using CFD data,’ Japanese Journal of Multiphase Flow, vol. 37, no. 1, pp. 94–102, 2023, doi: 10.3811/jjmf.2023.010.",
    "[2] H. Zhang, B. Huang, J. Zhu, and G. Chen, ‘Fast prediction of steady and unsteady hydrofoil cavitation flow fields based on the U-Net enhanced Fourier neural operator,’ Physics of Fluids, vol. 37, 085254, 2025, doi: 10.1063/5.0285357.",
    "[3] J. Clark-Y et al., ‘A reduced-order modeling approach for cavitating flow around a hydrofoil,’ Journal of Marine Science and Engineering, vol. 12, no. 1, p. 148, 2024, doi: 10.3390/jmse12010148.",
    "[4] T. Hino and S. Okabayashi, ‘Super-resolution reconstruction of pressure fields for cavitation-flow prediction,’ Transactions of the JSME, 2024, doi: 10.1299/transjsme.24-00115.",
    "[5] F. R. Menter, ‘Two-equation eddy-viscosity turbulence models for engineering applications,’ AIAA Journal, vol. 32, no. 8, pp. 1598–1605, 1994.",
    "[6] G. H. Schnerr and J. Sauer, ‘Physical and numerical modeling of unsteady cavitation dynamics,’ in Proc. 4th International Conference on Multiphase Flow, New Orleans, LA, USA, 2001.",
    "[7] O. Ronneberger, P. Fischer, and T. Brox, ‘U-Net: Convolutional networks for biomedical image segmentation,’ in Proc. MICCAI, 2015, pp. 234–241.",
    "[8] Z. Li et al., ‘Fourier neural operator for parametric partial differential equations,’ in Proc. ICLR, 2021.",
    "[9] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, ‘Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators,’ Nature Machine Intelligence, vol. 3, pp. 218–229, 2021.",
    "[10] M. Raissi, P. Perdikaris, and G. E. Karniadakis, ‘Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,’ Journal of Computational Physics, vol. 378, pp. 686–707, 2019.",
    "[11] OpenCFD Ltd., ‘OpenFOAM v2312 user guide,’ 2023. [Online]. Available: https://www.openfoam.com/documentation/",
    "[12] C. Li, P. Yuan, Y. Liu, J. Tan, X. Si, S. Wang, and Y. Cao, ‘Fast flow field prediction of hydrofoils based on deep learning,’ Ocean Engineering, vol. 281, 114743, 2023, doi: 10.1016/j.oceaneng.2023.114743.",
    "[13] B. Liu and S. Park, ‘Two-dimensional prediction of transient cavitating flow around hydrofoils using a DeepCFD model,’ Journal of Marine Science and Engineering, vol. 12, no. 11, 2074, 2024, doi: 10.3390/jmse12112074.",
    "[14] H. Ouyang, Z. Zhu, K. Chen, B. Tian, B. Huang, and J. Hao, ‘Reconstruction of hydrofoil cavitation flow based on the chain-style physics-informed neural network,’ Engineering Applications of Artificial Intelligence, vol. 119, 105724, 2023, doi: 10.1016/j.engappai.2022.105724.",
    "[15] R. Gao, S. Heydari, and R. K. Jaiman, ‘Towards spatio-temporal prediction of cavitating fluid flow with graph neural networks,’ International Journal of Multiphase Flow, vol. 177, 104858, 2024, doi: 10.1016/j.ijmultiphaseflow.2024.104858.",
    "[16] K. Chen, X. Zhang, B. Huang, T. Liu, Y. Wu, and F. Gao, ‘Reconstruction and correction of attached cavity evolution on hydrofoil based on physics-informed correction network,’ Ocean Engineering, vol. 355, 125185, 2026, doi: 10.1016/j.oceaneng.2026.125185.",
]
for reference in references:
    doc.add_paragraph(re.sub(r"^\[\d+\]\s*", "", reference), style="references")

set_columns(doc.sections[-1], 2)
for section in doc.sections:
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)
    for paragraph in section.footer.paragraphs:
        paragraph.text = ""
    for node in section.footer._element.xpath(".//w:t"):
        node.text = ""
    for footer in (section.first_page_footer, section.even_page_footer):
        for node in footer._element.xpath(".//w:t"):
            node.text = ""

doc.core_properties.title = "Benchmarking Neural Surrogates for Transient Hydrofoil Cavitation Forecasting"
doc.core_properties.author = "Anonymous"

doc.save(OUTPUT)
print(OUTPUT)
