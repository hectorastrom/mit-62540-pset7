"""Example layout for the 6.2540 inverter prelab: back-gated MoS2 transistors and Au resistors.

Everything on layer 1 becomes metal (Ti 10 nm / Au 15 nm on the MoS2 film, patterned by lift-off).
Layer 10 is for outlines and notes only and is not fabricated. All lengths are in micrometres.

This script draws
  EXAMPLES   one finger (interdigitated) FET, one circular (Corbino) FET and one meander resistor
  FRAME      an empty inverter frame (1.5 mm x 1.5 mm) with its VDD, OUT and GND ports
writes them to inverter_pset_example.gds, and prints W, L, W/L and R of the example devices.

To make your design: copy this file and draw your resistor and FET inside a frame with the functions
below (resistor from VDD to OUT, FET from OUT to GND), following the design rules listed under
"design rules". We check every design against these rules after you submit.

Needs gdspy 1.6 (pip install gdspy, or conda install -c conda-forge gdspy) and numpy.
"""

import os
import numpy as np
import gdspy

METAL, NOTE = 1, 10      # layer 1 is fabricated, layer 10 is not
RSH_AU = 4.0             # ohm per square of the Ti 10 nm / Au 15 nm film (measured on chip A1)

# design rules
FRAME = 1500.0           # each inverter, ports included, fits in a 1.5 mm x 1.5 mm frame
PORT = 200.0             # VDD, OUT, GND: 200 um squares along the top edge, also the probe pads
MIN_L = 5.0              # channel length (gap between source and drain metal) of 5 um or more
MIN_SPACE = 10.0         # any other gap between metal of 10 um or more
MIN_WIDTH = 4.0          # metal lines 4 um wide or more
J_MAX = 1e6              # A/cm^2: keep the current density in the Au below about 1 MA/cm^2
T_AU = 15e-7             # cm, thickness of the Au (15 nm)
VDD = 10.0               # V


def rect(x0, y0, x1, y1, layer=METAL):
    return gdspy.Rectangle((x0, y0), (x1, y1), layer=layer)


def merge(polys, layer=METAL):
    """Union of a list of shapes, so touching rectangles become one polygon."""
    return gdspy.boolean(polys, None, "or", layer=layer)


def note(cell, s, x, y, size=30):
    """Text on the NOTE layer (not fabricated)."""
    cell.add(gdspy.Text(s, size, (x, y), layer=NOTE))


def max_current(w):
    """Largest current (A) a line of width w (um) should carry at J_MAX."""
    return J_MAX * w * 1e-4 * T_AU


# ---------- devices ----------

def finger_fet(cell, x0, y0, L=10.0, n_ch=20, overlap=400.0, wf=6.0, tip=30.0, bus=40.0):
    """Interdigitated FET with its lower-left corner at (x0, y0).

    The top bus is one electrode (drain), the bottom bus the other (source). Fingers of width wf
    alternate between the two buses, a distance L apart, and face each other over `overlap`.
    W = n_ch * overlap. The gap between a finger tip and the opposite bus is `tip`.
    Returns W, L and the width and height of the device.
    """
    n_f = n_ch + 1
    width = n_f * wf + n_ch * L
    y_lb = y0 + bus                       # top edge of the bottom bus
    y_ub = y_lb + tip + overlap + tip     # bottom edge of the top bus
    top = [rect(x0, y_ub, x0 + width, y_ub + bus)]
    bottom = [rect(x0, y0, x0 + width, y_lb)]
    for k in range(n_f):
        fx = x0 + k * (wf + L)
        if k % 2 == 0:
            top.append(rect(fx, y_lb + tip, fx + wf, y_ub))
        else:
            bottom.append(rect(fx, y_lb, fx + wf, y_ub - tip))
    cell.add([merge(top), merge(bottom)])
    return n_ch * overlap, L, width, y_ub + bus - y0


def ring_fet(cell, xc, yc, r_in=150.0, L=10.0, ring=60.0, lead=20.0, slot_gap=30.0, lead_len=60.0):
    """Circular (Corbino) FET centred at (xc, yc).

    Inner disk of radius r_in, channel = annulus of width L, outer electrode = ring of width `ring`.
    With one metal layer the disk can only be wired out through a slot in the ring: its lead
    (width `lead`) leaves upward, slot_gap away from the ring on each side. The ring has a lead
    downward. W/L = 2 pi / ln(r_out / r_in) with r_out = r_in + L; the slot changes it by a few %.
    Returns W/L.
    """
    r_out = r_in + L
    top = r_out + ring
    disk = gdspy.Round((xc, yc), r_in, tolerance=0.1, layer=METAL)
    inner = merge([disk, rect(xc - lead / 2, yc, xc + lead / 2, yc + top + lead_len)])
    outer = gdspy.Round((xc, yc), top, inner_radius=r_out, tolerance=0.1, layer=METAL)
    slot = rect(xc - lead / 2 - slot_gap, yc, xc + lead / 2 + slot_gap, yc + top + 1)
    outer = gdspy.boolean(outer, slot, "not", layer=METAL)
    outer = merge([outer, rect(xc - 20, yc - top - lead_len, xc + 20, yc - top + 1)])
    cell.add([inner, outer])
    return 2 * np.pi / np.log(r_out / r_in)


def meander(cell, x0, y0, w=4.0, space=16.0, leg=500.0, n=40, lead=50.0, lead_w=20.0):
    """Au meander resistor: n vertical legs of line width w, `space` apart, `leg` long (between the
    centre lines of the turns). The first leg starts at (x0, y0); a lead of width lead_w leaves to the
    left from the start and to the right from the end. A 90-degree corner counts as ~0.56 square.
    Returns the number of squares and the width and height of the meander.
    """
    pitch = w + space
    xs = x0 + w / 2 + np.arange(n) * pitch
    polys = [rect(x - w / 2, y0 - w / 2, x + w / 2, y0 + leg + w / 2) for x in xs]
    for k in range(n - 1):
        y = y0 + leg if k % 2 == 0 else y0
        polys.append(rect(xs[k] - w / 2, y - w / 2, xs[k + 1] + w / 2, y + w / 2))
    y_end = y0 + leg if n % 2 == 1 else y0
    polys.append(rect(x0 - lead, y0 - lead_w / 2, x0 + w, y0 + lead_w / 2))
    polys.append(rect(xs[-1] - w / 2, y_end - lead_w / 2, xs[-1] + w / 2 + lead, y_end + lead_w / 2))
    cell.add(merge(polys))
    squares = (n * leg + (n - 1) * pitch) / w - 0.88 * (n - 1)
    return squares, n * pitch - space, leg + w


def inverter_frame(cell, size=FRAME):
    """Empty inverter frame with its lower-left corner at (0, 0).

    The outline is on the NOTE layer. The three ports are metal squares along the top edge, which are
    also the probe pads: VDD at the top-left corner, OUT in the middle, GND at the top-right corner.
    Everything below them (y < size - PORT - 10) is free for the resistor and the transistor.
    There is no gate port: the gate is the silicon substrate, shared by every device on the chip.
    """
    m, y0 = size / 2, size - PORT
    cell.add(rect(0, 0, size, size, NOTE))
    cell.add(rect(0, y0, PORT, size))                                   # VDD
    cell.add(rect(m - PORT / 2, y0, m + PORT / 2, size))                # OUT
    cell.add(rect(size - PORT, y0, size, size))                         # GND
    for s, x in (("VDD", 10), ("OUT", m - PORT / 2 + 10), ("GND", size - PORT + 10)):
        note(cell, s, x, y0 - 40)


# ---------- example ----------

def main():
    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)

    # ex = lib.new_cell("EXAMPLES")
    # W, L, w_f, h_f = finger_fet(ex, 0, 0)
    # note(ex, "finger FET", 0, h_f + 40)
    # note(ex, f"W/L = {W / L:.0f}", 0, -70)
    # print(f"finger FET: L = {L:g} um, W = {W / 1000:g} mm, W/L = {W / L:.0f}, size {w_f:.0f} x {h_f:.0f} um")

    # wl = ring_fet(ex, 750, 270)
    # note(ex, "ring FET", 600, 620)
    # note(ex, f"W/L = {wl:.0f}", 600, -70)
    # print(f"ring FET: r_in = 150 um, L = 10 um, W/L = {wl:.0f}")

    # sq, w_m, h_m = meander(ex, 1150, 20)
    # note(ex, "meander resistor", 1150, h_m + 60)
    # note(ex, f"R = {sq * RSH_AU / 1000:.1f} kOhm", 1150, -70)
    # print(f"meander: 4 um line, 16 um space, 40 legs x 500 um, {sq:.0f} squares, "
    #       f"R = {sq * RSH_AU / 1000:.1f} kOhm at {RSH_AU:g} ohm/sq, size {w_m:.0f} x {h_m:.0f} um, "
    #       f"at most {max_current(4.0) * 1e3:.1f} mA in its 4 um line")

    frame = lib.new_cell("FRAME")
    inverter_frame(frame)

    # -------- RESISTOR (Vdd -> V_out) --------
    rcell = lib.new_cell("RESISTOR")
    R_ORIGIN = (300, 1400) # relative to bottom left of frame
    R_W       = 4    # um
    R_SPACE   = 16   # um
    R_LEG_LEN = 1200 # um = 1.2mm
    R_NUM_LEG = 20   # count

    # Resistance of meander resistor is R = N_sq * R_sqAu
    # Placement within cell
    sq, w_m, h_m = meander(rcell, 0, 0, R_W, R_SPACE, R_LEG_LEN, R_NUM_LEG, lead=150, lead_w=20)      # from the VDD port to the OUT node
    R_res = sq * RSH_AU
    print(f"Meander resistor resistance: sq*RSH_AU={R_res:.1f} Ohm")

    # -------- TRANSISTOR (V_out -> GND) --------
    T_SEPARATION_L = 5   # um; 5um minimum
    T_N_FINGERS    = 50  # count (TODO:)
    T_OVERLAP      = 200 # um (TODO:)
    T_W            = 4  # um; 4um min
    T_TIP          = 20 # um (TODO:)

    W, L, w_f, h_f = finger_fet(frame, 0, 0, T_SEPARATION_L, T_N_FINGERS, T_OVERLAP, T_W, T_TIP)  # from the OUT node to the GND port
    # CONNECT WITH rect() leads
    def R_t(voltage : int):
        res_range = (0, 0) # lo, hi — resistance per square in MOhm
        match voltage: 
            case 30:
                res_range = (10, 50)
            case 20: res_range = (20, 100)
            case 0:  res_range = (1000, 16000)
        # for on threshold, a higher resistance is WORSE as it means more of the
        # voltage is being disappated across the transistor, raising V_out
        worst_case = res_range[1] 

        return worst_case

    # -------- CHECKS --------
    # Trace widths
    _min_trace_width = min(R_W, T_W) # um
    assert _min_trace_width >= 4, f"Trace width {R_W=} or {T_W=} are under 4um"

    # LOW < 25% of VDD
    HIGH_V_TRANSISTOR = 30 # V - what we want to guarantee is logic LOW (after inversion)
    _total_resistance = R_t(HIGH_V_TRANSISTOR) + R_res
    _resistor_r_fraction = R_res / _total_resistance 
    _transistor_r_fraction = R_t(HIGH_V_TRANSISTOR) / _total_resistance
    assert _transistor_r_fraction < 0.25, "Inverter logic LOW exceeds .25*VDD"

    # Current limits
    _overall_current = VDD / _total_resistance
    _current_limit =  J_MAX * _min_trace_width * T_AU*10e-2 * 10e4 # /cm^2 * um * cm * scale
    assert _overall_current < _current_limit, f"Current {_overall_current} exceeds safe limit of {_current_limit}"

    # TODO: more checks

    top = lib.new_cell("TOP")
    # top.add(gdspy.CellReference(ex, (0, 0)))
    top.add(gdspy.CellReference(frame, (0, 0)))
    frame.add(gdspy.CellReference(rcell, origin=R_ORIGIN, x_reflection=True)) # flip

    path = os.path.splitext(os.path.abspath(__file__))[0] + ".gds"
    lib.write_gds(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
