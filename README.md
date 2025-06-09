# beam_tracing_method Toolkit for AR Waveguides
This code was developed by Yefu Zhang (yefuzhang@ucf.edu) and Yuqiang Ding (yuqiang.ding@ucf.edu) under the supervision of Prof. Shin‑Tson Wu (swu@creol.ucf.edu).

This repository accompanies our paper on **A Framework for optimizing uniformity and efficiency in AR waveguide displays with open-source beam tracing method**. It contains four core scripts:

| Script                   | Purpose                                                                                                                             |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `plot_design.py`         | Visualize the AR layout (in‑coupler, folding couplers, out‑coupler, and **K‑diagram**) and verify that design parameters are valid. |
| `beam_tracing_saving.py` | Trace each beam **once**, save all effective beams, and store them for later optimization.                                          |
| `optimization.py`        | Load the saved beams, run NSGA‑II, and write Pareto‑optimal solutions.                                                              |
| `plot_opti_result.py`    | Plot and inspect optimization results.                                                                                              |

---

## 1  `plot_design.py`

Visualizes the layout and the nine field‑of‑view (FoV) angles used in the paper; you can add more angles for finer sampling.

| Code line*  | Parameter                                     | Description                                                                                                                                       |
| ----------- | --------------------------------------------  | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| 189         | `FOV_num`                                     | Select which of the nine FoVs to show (integer **1 – 9**). Also displays remaining energy after the in‑coupler.                                   |
| 191‑194     | `FoV_x`, `FoV_y`                              | Overall design FoV (°).                                                                                                                           |
| 199         | `lmd`                                         | Design wavelength (nm).                                                                                                                           |
| 203‑207     | `n_g`, `t`, `x`, `y`                          | Refractive index, thickness, and outer dimensions of the waveguide glass.                                                                         |
| 210         | `r`                                           | Radius of the in‑coupler.                                                                                                                         |
| 213‑214     | `x_ic0`, `y_ic0`                              | Centre position of the in‑coupler (input pupil).                                                                                                  |
| 221‑222     | `x_eb`, `y_eb`                                | Eyebox width × height (mm).                                                                                                                       |
| 226‑227     | `x_eb0`, `y_eb0`                              | Eyebox center position.                                                                                                                           |
| 245‑251     | `Lambda_ic`, `Lambda_oc`, `phi_ic`, `phi_oc`  | Period and diffraction direction of in‑ and out‑couplers.                                                                                         |
| 384‑394     | `FoV_X_9c`, `FoV_Y_9c`                         | List of FoV angles to optimize (replace the default nine if desired).                                                                             |
| 417, 571    | `slice_width`                                 | **Slicing** count for folding and out‑couplers. Use a *fractional* placeholder (e.g. `7.001` instead of `7`) to avoid integer‑division artefacts. |

\*Line numbers refer to the current version and may shift if the file is edited.

> **Tip:** Re‑run the script after every parameter change to confirm that couplers do **not** overlap and that the K‑diagram is physically reasonable.

---

## 2  `beam_tracing_saving.py`

Once the design looks correct, run this script to trace beams and store the effective ones on disk.

1. **Sync parameters** – from **line 508 onward**, copy the same values you set in `plot_design.py`.
2. **FoV selection**
   - Line 498: choose which of the nine default FoVs to propagate (integer **1 – 9**).
   - Lines 654‑663: optionally redefine the FoV list to any angles within the design range.
3. **Coupler slicing** – lines 686 (folding) and 839 (out) as above.
4. **Parallelism** – line 1063: set how many CPU cores to use.
5. **Memory control** – line 1104: set the batch size for beams entering the out‑coupler. Use smaller batches if your RAM is limited.

---

## 3  `optimization.py`

Reads the saved beams and performs multi‑objective optimization (default objectives: optical efficiency and eyebox uniformity).

| Code line | What to edit                                                                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------- |
| 46        | `xl`, `xu` – lower/upper limits for each optimization variable (currently coupler efficiencies **0 – 1**).     |
| 65        | `uniformity-0.25` – feasibility threshold (default ≥ 0.75).                                                    |
| 108       | `for FOV_num in [1,2,3,4,5,6,7,8,9]` – FoV angles to include in the objective calculation.                     |
| 138‑139   | `popsize`, `n_gen` – population size and number of generations for NSGA‑II.                                    |

**Outputs**

- `optimize_{FOV_num}_con_uni_0.75_ener` – objective values and corresponding variable sets

---

## 4  `plot_opti_result.py`

Generates plots of the optimisation results. No mandatory edits; run the script and adjust axis labels, colours, or figure size as needed.



*Happy tracing!*
