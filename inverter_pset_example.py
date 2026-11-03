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


# ---------- OUR CHOSEN PARAMETERS ----------
# Physical positioning, relative to bottom left of 1.5mm square
R_ORIGIN = (240, 1410)  
T_ORIGIN = (550, 1200)

# Resistor
R_W       = 4    # um
R_SPACE   = 16   # um
R_LEG_LEN = 1350 # um = 1.2mm
R_NUM_LEG = 14   # count

# Transistor
T_L          = 5     # um, channel length (>= MIN_L)
T_N_CHANNELS = 110   # finger_fet is T_N_CHANNELS + 1 fingers
T_OVERLAP    = 800   # um
T_WF         = 4     # um, finger width (>= MIN_WIDTH)
T_TIP        = 15    # um 


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

    frame = lib.new_cell("FRAME")
    inverter_frame(frame)

    # -------- RESISTOR (Vdd -> V_out) --------
    rcell = lib.new_cell("RESISTOR")
    # Resistance of meander resistor is R = N_sq * R_sqAu
    # Placement within cell
    sq, w_m, h_m = meander(rcell, 0, 0, R_W, R_SPACE, R_LEG_LEN, R_NUM_LEG, lead=150, lead_w=40)      # from the VDD port to the OUT node
    R_res = sq * RSH_AU
    print(f"Meander resistor resistance: sq*RSH_AU={R_res:.1f} Ohm")

    # -------- TRANSISTOR (V_out -> GND) --------
    tcell = lib.new_cell("TRANSISTOR")
    W_T, L_T, w_f, h_f = finger_fet(tcell, 0, 0, T_L, T_N_CHANNELS, T_OVERLAP, T_WF, T_TIP)  # from the OUT node to the GND port

    RSQ_MOS2 = {
        30: (10e6, 50e6),
        20: (20e6, 100e6),
        0: (1e9, 6e9),
    }  # ohm/sq: (low, hi)

    def r_fet(v_gate, W, L):
        lo, hi = RSQ_MOS2[v_gate]
        return lo * L / W, hi * L / W

    R_on_lo, R_on_hi = r_fet(30, W_T, L_T)
    print(
        f"FET: W/L = {W_T / L_T:.0f}, R_on = {R_on_lo / 1e3:.1f} to {R_on_hi / 1e3:.1f} kOhm"
    )

    # -------- CHECKS --------
    VDD = 10.0
    assert T_L >= MIN_L, f"channel length {T_L} um < {MIN_L} um"
    assert min(R_W, T_WF) >= MIN_WIDTH, f"line width under {MIN_WIDTH} um"
    assert R_SPACE >= MIN_SPACE and T_TIP >= MIN_SPACE, "gap under 10 um"

    v_low = R_on_hi / (R_on_hi + R_res)  # V_out / VDD at Vin = +30 V
    print(f"Voltage at low = {v_low:.2f} * VDD")
    assert v_low <= 0.25, f"V_out LOW = {v_low:.2f} VDD > 0.25 VDD" # LOW state

    I_worst = VDD / (R_on_lo + R_res)  # highest current: FET at its lowest R
    print(f"Max current: {I_worst*1e3:.3f}mA")
    assert I_worst < max_current(R_W), f"{I_worst * 1e3:.2f} mA > {max_current(R_W) * 1e3:.2f} mA in the {R_W} um meander"

    top = lib.new_cell("TOP")
    frame.add(gdspy.CellReference(rcell, origin=R_ORIGIN, x_reflection=True))  # flip
    # top is DRAIN, bottom is SOURCE; so rotating -90 puts SRC on left and DRAIN
    # on right
    frame.add(gdspy.CellReference(tcell, T_ORIGIN, rotation = -90))
    
    # Wires
    wires = lib.new_cell("WIRES")
    wire_width = 40
    wires.add(rect(550, 700, 550 + wire_width, 1400)) # SRC up
    wires.add(rect(550, 1390, 750, 1390 + wire_width)) # SRC right to OUT
    wires.add(rect(1460-wire_width, 1200, 1460, 1390+wire_width))
    top.add(gdspy.CellReference(wires, (0, 0)))
    
    top.add(gdspy.CellReference(frame, (0, 0)))

    path = os.path.splitext(os.path.abspath(__file__))[0] + ".gds"
    lib.write_gds(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
