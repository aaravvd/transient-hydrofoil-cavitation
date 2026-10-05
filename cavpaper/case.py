from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np


RHO_LIQUID = 997.0
RHO_VAPOUR = 0.023
NU_LIQUID = 1.0e-6
NU_VAPOUR = 4.3e-4
P_VAPOUR = 2300.0
PSI_LIQUID = 5.0e-7
PSI_VAPOUR = 2.5e-6
SPAN = 0.001


@dataclass(frozen=True)
class CavitationCase:
    id: str
    geometry: str
    chord_m: float
    angle_deg: float
    reynolds: float
    cavitation_number: float
    end_time_s: float
    write_interval_s: float
    initial_delta_t_s: float
    mesh_size_foil_m: float
    mesh_size_far_m: float
    max_co: float = 0.5

    @property
    def speed(self) -> float:
        return self.reynolds * NU_LIQUID / self.chord_m

    @property
    def outlet_pressure(self) -> float:
        dynamic_pressure = 0.5 * RHO_LIQUID * self.speed**2
        return P_VAPOUR + self.cavitation_number * dynamic_pressure

    @property
    def inlet_total_pressure(self) -> float:
        return self.outlet_pressure + 0.5 * RHO_LIQUID * self.speed**2


def load_cases(path: Path) -> list[CavitationCase]:
    payload = json.loads(path.read_text())
    return [CavitationCase(**record) for record in payload["cases"]]


def write_case(case: CavitationCase, case_dir: Path) -> None:
    for subdir in ("0", "constant", "system"):
        (case_dir / subdir).mkdir(parents=True, exist_ok=True)
    for obsolete in ("p", "rho", "alpha.vapour", "thermodynamicProperties", "momentumTransport"):
        for parent in (case_dir / "0", case_dir / "constant"):
            path = parent / obsolete
            if path.exists():
                path.unlink()
    x, y = naca4_coordinates(case.geometry, chord=case.chord_m)
    _write_metadata(case, case_dir)
    _write_gmsh(case, case_dir, x, y)
    _write_properties(case_dir)
    _write_control(case, case_dir)
    _write_schemes(case_dir)
    _write_solution(case_dir)
    _write_fields(case, case_dir)


def naca4_coordinates(code: str, chord: float, n: int = 201) -> tuple[np.ndarray, np.ndarray]:
    clean = code.upper().replace("NACA", "").strip()
    if len(clean) != 4 or not clean.isdigit():
        raise ValueError(f"Pilot currently supports four-digit NACA geometries, got {code!r}")
    m = int(clean[0]) / 100.0
    p = int(clean[1]) / 10.0
    t = int(clean[2:]) / 100.0
    beta = np.linspace(0.0, math.pi, n)
    xc = 0.5 * (1.0 - np.cos(beta))
    yt = 5.0 * t * (
        0.2969 * np.sqrt(np.maximum(xc, 1e-12))
        - 0.1260 * xc
        - 0.3516 * xc**2
        + 0.2843 * xc**3
        - 0.1015 * xc**4
    )
    yc = np.zeros_like(xc)
    slope = np.zeros_like(xc)
    if m > 0 and p > 0:
        fore = xc < p
        aft = ~fore
        yc[fore] = m / p**2 * (2 * p * xc[fore] - xc[fore] ** 2)
        slope[fore] = 2 * m / p**2 * (p - xc[fore])
        yc[aft] = m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xc[aft] - xc[aft] ** 2)
        slope[aft] = 2 * m / (1 - p) ** 2 * (p - xc[aft])
    theta = np.arctan(slope)
    xu, yu = xc - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = xc + yt * np.sin(theta), yc - yt * np.cos(theta)
    return np.r_[xu[::-1], xl[1:]] * chord, np.r_[yu[::-1], yl[1:]] * chord


def _header(class_name: str, object_name: str, location: str | None = None) -> str:
    location_line = f'    location    "{location}";\n' if location else ""
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
{location_line}    object      {object_name};
}}

"""


def _write_metadata(case: CavitationCase, case_dir: Path) -> None:
    record = {
        **case.__dict__,
        "speed_m_per_s": case.speed,
        "outlet_pressure_pa": case.outlet_pressure,
        "inlet_total_pressure_pa": case.inlet_total_pressure,
        "rho_liquid_kg_per_m3": RHO_LIQUID,
        "rho_vapour_kg_per_m3": RHO_VAPOUR,
        "vapour_pressure_pa": P_VAPOUR,
        "solver": "OpenFOAM v2312 interPhaseChangeFoam",
        "cavitation_model": "Schnerr-Sauer mass transfer model"
    }
    (case_dir / "case.json").write_text(json.dumps(record, indent=2) + "\n")


def _write_gmsh(case: CavitationCase, case_dir: Path, ax: np.ndarray, ay: np.ndarray) -> None:
    c = case.chord_m
    x0, x1, y0, y1 = -2 * c, 5 * c, -2 * c, 2 * c
    lines = [
        'SetFactory("Built-in");',
        "Mesh.MshFileVersion = 2.2;",
        f"lcFar = {case.mesh_size_far_m:.9g};",
        f"lcFoil = {case.mesh_size_foil_m:.9g};",
        f"Point(1) = {{{x0}, {y0}, 0, lcFar}};",
        f"Point(2) = {{{x1}, {y0}, 0, lcFar}};",
        f"Point(3) = {{{x1}, {y1}, 0, lcFar}};",
        f"Point(4) = {{{x0}, {y1}, 0, lcFar}};",
        "Line(1) = {1, 2};", "Line(2) = {2, 3};", "Line(3) = {3, 4};", "Line(4) = {4, 1};"
    ]
    start = 100
    for i, (x, y) in enumerate(zip(ax[:-1], ay[:-1]), start=start):
        lines.append(f"Point({i}) = {{{x:.10g}, {y:.10g}, 0, lcFoil}};")
    ids = list(range(start, start + len(ax) - 1))
    le = int(np.argmin(ax[:-1]))
    upper = ids[: le + 1]
    lower = ids[le:] + [ids[0]]
    lines.extend([
        f"Spline(20) = {{{', '.join(map(str, upper))}}};",
        f"Spline(21) = {{{', '.join(map(str, lower))}}};",
        "Curve Loop(30) = {1, 2, 3, 4};",
        "Curve Loop(31) = {20, 21};",
        "Plane Surface(40) = {30, 31};",
        f"out[] = Extrude {{0, 0, {SPAN}}} {{ Surface{{40}}; Layers{{1}}; Recombine; }};",
        'Physical Surface("frontAndBack") = {40, out[0]};',
        'Physical Surface("topAndBottom") = {out[2], out[4]};',
        'Physical Surface("outlet") = {out[3]};',
        'Physical Surface("inlet") = {out[5]};',
        'Physical Surface("airfoil") = {out[6], out[7]};',
        'Physical Volume("internal") = {out[1]};',
        "Mesh.Algorithm = 6;"
    ])
    (case_dir / "mesh.geo").write_text("\n".join(lines) + "\n")


def _write_properties(case_dir: Path) -> None:
    (case_dir / "constant" / "transportProperties").write_text(_header("dictionary", "transportProperties") + f"""phases (water vapour);

phaseChangeTwoPhaseMixture SchnerrSauer;
pSat            {P_VAPOUR};
sigma           0.07;

water
{{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] {NU_LIQUID};
    rho             [1 -3 0 0 0 0 0] {RHO_LIQUID};
}}

vapour
{{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] {NU_VAPOUR};
    rho             [1 -3 0 0 0 0 0] {RHO_VAPOUR};
}}

SchnerrSauerCoeffs
{{
    n               1.6e13;
    dNuc            2.0e-6;
    Cc              1;
    Cv              1;
}}
""")
    (case_dir / "constant" / "turbulenceProperties").write_text(_header("dictionary", "turbulenceProperties") + """simulationType RAS;
RAS
{
    RASModel        kOmegaSST;
    turbulence      on;
    printCoeffs     on;
}
""")
    (case_dir / "constant" / "g").write_text(_header("uniformDimensionedVectorField", "g") + """dimensions [0 1 -2 0 0 0 0];
value (0 0 0);
""")


def _write_control(case: CavitationCase, case_dir: Path) -> None:
    angle = math.radians(case.angle_deg)
    lift = (-math.sin(angle), math.cos(angle))
    drag = (math.cos(angle), math.sin(angle))
    (case_dir / "system" / "controlDict").write_text(_header("dictionary", "controlDict") + f"""application     interPhaseChangeFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {case.end_time_s};
deltaT          {case.initial_delta_t_s};
writeControl    adjustableRunTime;
writeInterval   {case.write_interval_s};
writeFormat     ascii;
writePrecision  8;
writeCompression off;
runTimeModifiable yes;
adjustTimeStep  on;
maxCo           {case.max_co};

functions
{{
    forceCoeffs
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        patches         (airfoil);
        writeControl    timeStep;
        writeInterval   20;
        p               p;
        U               U;
        rho             rhoInf;
        rhoInf          {RHO_LIQUID};
        pRef            {case.outlet_pressure};
        liftDir         ({lift[0]} {lift[1]} 0);
        dragDir         ({drag[0]} {drag[1]} 0);
        CofR            ({0.25 * case.chord_m} 0 0);
        pitchAxis       (0 0 1);
        magUInf         {case.speed};
        lRef            {case.chord_m};
        Aref            {case.chord_m * SPAN};
    }}
}}
""")


def _write_schemes(case_dir: Path) -> None:
    (case_dir / "system" / "fvSchemes").write_text(_header("dictionary", "fvSchemes") + """ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(rhoPhi,U) Gauss linearUpwind grad(U);
    div(phi,omega) Gauss linearUpwind grad(omega);
    div(phi,k) Gauss linearUpwind grad(k);
    div(phi,alpha) Gauss vanLeer;
    div(phirb,alpha) Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear limited corrected 0.5; }
interpolationSchemes { default linear; }
snGradSchemes { default limited corrected 0.5; }
wallDist { method meshWave; }
""")


def _write_solution(case_dir: Path) -> None:
    (case_dir / "system" / "fvSolution").write_text(_header("dictionary", "fvSolution") + """solvers
{
    "alpha.water.*"
    {
        cAlpha 0; nAlphaCorr 2; nAlphaSubCycles 1;
        MULESCorr yes; nLimiterIter 5;
        solver smoothSolver; smoother symGaussSeidel;
        tolerance 1e-8; relTol 0; maxIter 10;
    }
    "(U|k|omega).*" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0; }
    p_rgh { solver GAMG; tolerance 1e-8; relTol 0.1; smoother DICGaussSeidel; maxIter 50; }
    p_rghFinal
    {
        solver PCG;
        preconditioner { preconditioner GAMG; tolerance 1e-6; relTol 0; nVcycles 2; smoother DICGaussSeidel; }
        tolerance 1e-7; relTol 0; maxIter 50;
    }
    "pcorr.*" { $p_rgh; relTol 0; }
    Phi { $p_rgh; relTol 0; }
}
potentialFlow { nNonOrthogonalCorrectors 3; }
PIMPLE
{
    momentumPredictor no;
    nOuterCorrectors 1;
    nCorrectors 3;
    nNonOrthogonalCorrectors 0;
}
relaxationFactors { equations { "U.*" 1; } }
""")


def _write_fields(case: CavitationCase, case_dir: Path) -> None:
    a = math.radians(case.angle_deg)
    ux, uy = case.speed * math.cos(a), case.speed * math.sin(a)
    p0 = case.outlet_pressure
    patches = {
        "U": f"""inlet {{ type fixedValue; value uniform ({ux} {uy} 0); }}
    outlet {{ type pressureInletOutletVelocity; value uniform ({ux} {uy} 0); }}
    topAndBottom {{ type freestream; freestreamValue uniform ({ux} {uy} 0); }}
    airfoil {{ type noSlip; }}
    frontAndBack {{ type empty; }}""",
        "p_rgh": f"""inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform {p0}; }}
    topAndBottom {{ type freestreamPressure; freestreamValue uniform {p0}; }}
    airfoil {{ type fixedFluxPressure; value uniform {p0}; }}
    frontAndBack {{ type empty; }}""",
        "alpha.water": """inlet { type fixedValue; value uniform 1; }
    outlet { type inletOutlet; inletValue uniform 1; value uniform 1; }
    topAndBottom { type fixedValue; value uniform 1; }
    airfoil { type zeroGradient; }
    frontAndBack { type empty; }""",
        "Phi": """inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    topAndBottom { type zeroGradient; }
    airfoil { type zeroGradient; }
    frontAndBack { type empty; }""",
    }
    vector = _header("volVectorField", "U", "0") + f"dimensions [0 1 -1 0 0 0 0];\ninternalField uniform ({ux} {uy} 0);\nboundaryField\n{{\n    {patches['U']}\n}}\n"
    (case_dir / "0" / "U").write_text(vector)
    for name, dims, value in (
        ("p_rgh", "[1 -1 -2 0 0 0 0]", p0),
        ("alpha.water", "[0 0 0 0 0 0 0]", 1),
        ("Phi", "[0 2 -1 0 0 0 0]", 0),
    ):
        (case_dir / "0" / name).write_text(_header("volScalarField", name, "0") + f"dimensions {dims};\ninternalField uniform {value};\nboundaryField\n{{\n    {patches[name]}\n}}\n")
    k = 1.5 * (0.02 * case.speed) ** 2
    omega = math.sqrt(k) / (0.09**0.25 * 0.007 * case.chord_m)
    scalar_specs = {
        "k": ("[0 2 -2 0 0 0 0]", k, "kqRWallFunction"),
        "omega": ("[0 0 -1 0 0 0 0]", omega, "omegaWallFunction"),
        "nut": ("[0 2 -1 0 0 0 0]", 0.0, "nutkWallFunction"),
    }
    for name, (dims, value, wall_type) in scalar_specs.items():
        inlet = "calculated" if name == "nut" else "fixedValue"
        text = _header("volScalarField", name, "0") + f"""dimensions {dims};
internalField uniform {value};
boundaryField
{{
    inlet {{ type {inlet}; value uniform {value}; }}
    outlet {{ type zeroGradient; }}
    topAndBottom {{ type {inlet}; value uniform {value}; }}
    airfoil {{ type {wall_type}; value uniform {value}; }}
    frontAndBack {{ type empty; }}
}}
"""
        (case_dir / "0" / name).write_text(text)
