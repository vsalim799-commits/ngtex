"""Composite the 3D guide onto the plate (runs in the Higgsfield sandbox).

In linear light: shadow density and tint matched to the real tree shadows on the road,
alpha noise floor removed, and the two traffic-island signs (always nearer than the
animal) kept in front via per-frame template tracking.
Inputs: plate.mp4, guide_fg.mp4 (straight RGB), guide_alpha.mp4 (R = total alpha, G = creature alpha).
"""
import subprocess, numpy as np, cv2, sys
W, H, N = 1920, 1080, 832
out_path = sys.argv[1] if len(sys.argv) > 1 else "guide_comp.mp4"


def reader(path, fmt="rgb24", ch=3):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", fmt, "-"], stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(W * H * ch)
        if len(b) < W * H * ch:
            break
        yield np.frombuffer(b, np.uint8).reshape(H, W, ch)


_c = np.arange(256) / 255.0
LUT_LIN = np.where(_c <= 0.04045, _c / 12.92, ((_c + 0.055) / 1.055) ** 2.4).astype(np.float32)
_x = np.linspace(0, 1, 4096)
LUT_ENC = (np.where(_x <= 0.0031308, 12.92 * _x, 1.055 * _x ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


def lin(x):
    return LUT_LIN[x]                       # uint8 sRGB -> linear (lookup table: fast)


def enc(x):
    return LUT_ENC[(np.clip(x, 0, 1) * 4095 + 0.5).astype(np.int32)]


def ss(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


# pass 1: grey crops around the island (tracking) + the reference frame in full
CY0, CY1, CX0, CX1 = 480, 960, 860, 1640
gray = []
f210 = None
for n, fr in enumerate(reader("plate.mp4")):
    gray.append(cv2.cvtColor(fr[CY0:CY1, CX0:CX1], cv2.COLOR_RGB2GRAY))
    if n == 210:
        f210 = fr.copy()
assert len(gray) == N, len(gray)
# ---------------- sign mattes at frame 210 (sign A: plates+bollard+post+pole; sign B: plate+pole)
R = 210
f = f210.astype(np.float32) / 255
r, g, b = f[..., 0], f[..., 1], f[..., 2]
mx, mn = f.max(-1), f.min(-1)
sat = (mx - mn) / np.maximum(mx, 1e-6)
L = 0.2126 * r + 0.7152 * g + 0.0722 * b
col = (sat > 0.35) & (((r > 0.45) & (g > 0.38) & (b < 0.6 * g)) | ((r > g + 0.15) & (r > b + 0.15)) | ((b > r + 0.12) & (b > g + 0.04)))
hull = np.zeros((H, W), bool)
for (y0, y1, x0, x1) in ((634, 646, 1133, 1143), (646, 675, 1123, 1147), (675, 708, 1112, 1157), (708, 750, 1117, 1151)):
    hull[y0:y1, x0:x1] = True
mA = hull & ((L >= 0.68) | col | ((L >= 0.5) & (sat >= 0.3)))
mA = cv2.morphologyEx(mA.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & hull
refA = mA.astype(np.float32)
refA[750:788, 1134:1140] = 1
refB = np.zeros((H, W), np.float32)
refB[601:634, 995:1031] = 1
refB[633:761, 1010:1015] = 1
# ---------------- track both signs (template matching, frame 210 reference)
T = {"A": (1108, 632, 1162, 752), "B": (990, 598, 1034, 636)}
off = {k: np.zeros((N, 2)) for k in T}
for k, (x0, y0, x1, y1) in T.items():
    tpl = gray[R][y0 - CY0:y1 - CY0, x0 - CX0:x1 - CX0].astype(np.float32)
    th, tw = tpl.shape
    for d in (1, -1):
        pos = np.zeros(2); vel = np.zeros(2); n = R + d
        while 0 <= n < N:
            pred = pos + vel; m = 48
            sx0 = int(round(x0 + pred[0])) - m; sy0 = int(round(y0 + pred[1])) - m
            if sx0 < CX0 or sy0 < CY0 or sx0 + tw + 2 * m > CX1 or sy0 + th + 2 * m > CY1:
                pos = pred
            else:
                cc = cv2.matchTemplate(gray[n][sy0 - CY0:sy0 - CY0 + th + 2 * m, sx0 - CX0:sx0 - CX0 + tw + 2 * m].astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
                _, v, _, (bx, by) = cv2.minMaxLoc(cc)
                sp = lambda a, c, e: 0.0 if abs(a - 2 * c + e) < 1e-9 else 0.5 * (a - e) / (a - 2 * c + e)
                ox = sp(cc[by, bx - 1], cc[by, bx], cc[by, bx + 1]) if 0 < bx < cc.shape[1] - 1 else 0
                oy = sp(cc[by - 1, bx], cc[by, bx], cc[by + 1, bx]) if 0 < by < cc.shape[0] - 1 else 0
                new = np.array([sx0 + bx + ox - x0, sy0 + by + oy - y0]) if v >= 0.5 else pred
                vel = new - pos; pos = new
            off[k][n] = pos; n += d
    # light temporal smoothing (tracker noise ~0.2 px)
    for c in range(2):
        off[k][:, c] = np.convolve(np.pad(off[k][:, c], 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
print("tracked", {k: (round(float(v[:, 0].min()), 1), round(float(v[:, 0].max()), 1)) for k, v in off.items()}, flush=True)
# ---------------- shadow ratio of the real road (sunlit vs tree shadow), per channel, linear light
reg = f210[540:740, 1150:1500]
rf = reg.astype(np.float32) / 255
satr = (rf.max(-1) - rf.min(-1)) / np.maximum(rf.max(-1), 1e-6)
asphalt = (satr < 0.18) & (rf.max(-1) < 0.8)            # grey asphalt only: no paint, signs or foliage
road = lin(reg)[asphalt]
lum = road @ np.array([0.2126, 0.7152, 0.0722])
lo, hi = np.percentile(lum, [10, 90])
shade = road[lum <= lo].mean(0); sun = road[lum >= hi].mean(0)
ratio = np.clip(shade / np.maximum(sun, 1e-6), 0.12, 0.8)
if not 0.12 < float(ratio.mean()) < 0.6:
    ratio = np.array([0.28, 0.30, 0.34])                 # fallback: typical hard-sun / sky-fill ratio
print("asphalt pixels", int(asphalt.sum()), "sun", sun.round(3), "shade", shade.round(3), flush=True)
r_mean = float(ratio @ np.array([0.2126, 0.7152, 0.0722]))
k_c = (1 - ratio) / (1 - r_mean)
print("road shadow ratio per channel", ratio.round(3), "mean", round(r_mean, 3), flush=True)
np.save('sign_offsets.npy', np.stack([off['A'], off['B']], 1))
if '--dry' in sys.argv:
    sys.exit(0)
# ---------------- composite
enc_p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "30", "-i", "-",
                          "-i", "plate.mp4", "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                          "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                          "-c:a", "copy", "-movflags", "+faststart", "-shortest", out_path], stdin=subprocess.PIPE)
HEDGE = (20.19, 1.2444)          # hedge top edge in frame 210: u = a + b v (robust fit, 81/87 inliers)
VV = np.arange(H, dtype=np.float32)[:, None]
UU = np.arange(W, dtype=np.float32)[None, :]
for n, (pl8, fg8, al8) in enumerate(zip(reader("plate.mp4"), reader("guide_fg.mp4"), reader("guide_alpha.mp4"))):
    a = al8[..., 0].astype(np.float32) / 255          # R: total alpha (noise floor already removed)
    bdy = al8[..., 1].astype(np.float32) / 255        # G: creature only
    s = np.clip(a - bdy, 0, 1)                        # shadow part
    fg = lin(fg8)
    # density comes from the plate-matched render lighting; only the tint is taken from the real road
    occ = np.zeros((H, W), np.float32)
    for k, ref in (("A", refA), ("B", refB)):
        M = np.float32([[1, 0, off[k][n, 0]], [0, 1, off[k][n, 1]]])
        occ = np.maximum(occ, cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LINEAR))
    occ = cv2.GaussianBlur(occ, (3, 3), 0.6)
    s *= 1 - occ; bdy *= 1 - occ
    # the camera-side hedge (~2 m) stops the shadow: cut it at the hedge's top edge (measured at frame 210,
    # moved with the camera like sign A) and, in a band around that edge, keep it off the foliage pixels
    dxh, dyh = off["A"][n]
    uline = HEDGE[0] + HEDGE[1] * (VV - dyh) + dxh
    pf = pl8.astype(np.float32) / 255
    satp = (pf.max(-1) - pf.min(-1)) / np.maximum(pf.max(-1), 1e-6)
    green = (pf[..., 1] > pf[..., 0] + 0.015) & (pf[..., 1] > pf[..., 2] + 0.03) & (satp > 0.12)
    allow = np.where(UU > uline + 25, 1.0, np.where(UU > uline - 40, (~green).astype(np.float32), 0.0)).astype(np.float32)
    s *= cv2.GaussianBlur(allow, (0, 0), 1.5)
    pl = lin(pl8)
    out = pl * (1 - s[..., None] * k_c[None, None, :]) * (1 - bdy[..., None]) + fg * bdy[..., None]
    enc_p.stdin.write(enc(out).tobytes())
    if n % 100 == 0:
        print("comp", n, flush=True)
enc_p.stdin.close(); enc_p.wait()
print("done", out_path, flush=True)
