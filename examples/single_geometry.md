# single_geometry — HV Asynchronous Generator Stator Example

Generates and visualises a single stator cross-section sized for a large,
4-pole, high-voltage (6–11 kV) asynchronous generator typical of power-station
or industrial power-generation plant.

## Usage

```bash
python examples/single_geometry.py
python examples/single_geometry.py --output /tmp/my_stator --no-plot
python examples/single_geometry.py --slots 48 --lam 1400
```

| Flag | Default | Description |
|---|---|---|
| `--output` | `/tmp/stator_single` | Output directory for JSON metadata and PNG images |
| `--no-plot` | off | Skip gmsh visualisation (2-D cross-section + 3-D stack) |
| `--slots` | 48 | Number of stator slots |
| `--lam` | 1400 | Number of laminations |

## Default geometry

### Radial dimensions

| Parameter | Value | Notes |
|---|---|---|
| Outer radius | 650 mm | Large-frame stator housing |
| Inner radius (bore) | 420 mm | 4-pole rotor clearance |
| Air-gap | 3 mm | Typical for large induction machines; reduces stray-load losses |
| Yoke height | ≈ 110 mm | R_outer − R_inner − slot_depth |

### Slot geometry

| Parameter | Value | Notes |
|---|---|---|
| Number of slots | 48 | Standard for large 4-pole machines (slots/pole/phase = 4) |
| Slot shape | SEMI_CLOSED | Reduces harmonic content while allowing preformed coils |
| Slot depth | 115 mm | Deep to accommodate HV multi-turn coils with thick groundwall |
| Slot width (outer / inner) | 22 / 19 mm | Slight taper aids winding insertion |
| Slot opening | 8 mm | Scaled to slot width |
| Slot opening depth | 6 mm | Shallow wedge region |
| Tooth tip angle | 0.08 rad | Moderate tip to reduce flux fringing |

### Coil and winding

The coil depth must satisfy the validator constraint:

```
coil_depth ≤ slot_depth − slot_opening_depth − 2 × insulation_thickness
           ≤ 0.115 − 0.006 − 0.006 = 0.103 m
```

| Parameter | Value | Notes |
|---|---|---|
| Coil depth | 103 mm | Maximum allowed by constraint above |
| Coil width (outer / inner) | 16 / 13 mm | Groundwall on each side of slot width |
| Insulation thickness | 3 mm | Class-F groundwall required at 6–11 kV |
| Insulation coating | 0.10 mm | Heavier strand coat for HV duty |
| Winding type | DOUBLE_LAYER | Preformed coils, allows short pitching |
| Turns per coil | 6 | Fewer turns at higher voltage (vs low-voltage machines) |
| Coil pitch | 11 | Short-pitched 11/12 for a 48-slot 4-pole machine |
| Wire diameter | 4 mm | Large round conductor for high rated current |
| Slot fill factor | 0.38 | Reduced from typical 0.45 — thick insulation occupies more space |

### Lamination stack

| Parameter | Value | Notes |
|---|---|---|
| Lamination thickness | 0.50 mm | M330-50A grade — standard for large industrial frames |
| Number of laminations | 1400 | Total stack length ≈ 700 mm |
| Material | M330-50A | Higher silicon content than M270-35A; suited to large-frame power generation |

Thicker laminations (0.5 mm vs 0.35 mm for smaller machines) are chosen
because core-loss frequency at 50/60 Hz is low relative to eddy-current losses
that would motivate thinner laminations; thicker laminations reduce tooling cost
and are consistent with industry practice for machines above ~1 MW.

### Mesh sizing

Mesh element sizes are scaled up proportionally to the larger geometry.

| Region | Element size |
|---|---|
| Yoke | 20 mm |
| Slot | 10 mm |
| Coil | 6 mm |
| Insulation | 3 mm |

## Outputs

| File | Description |
|---|---|
| `stator_cross_section.png` | 2-D colour-coded cross-section (yoke, coils A/B/C, insulation) |
| `stator_3d_stack.png` | Isometric view of 3-D lamination stack (up to 8 laminations visualised) |
| `*.json` | Pipeline metadata including SHA-256 fingerprint of the parameter set |
