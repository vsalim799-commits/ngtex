"""Final assembly (Higgsfield sandbox): the AI creature back onto the untouched iPhone plate.

1. Time-align the AI render (24 fps, slightly re-timed) to the plate (30 fps) on background pixels.
2. For every output frame (24 fps): colour-match the AI frame to the plate, take from it only what
   differs from the plate inside the guide's (dilated) creature + shadow region, keep the plate
   everywhere else, keep the traffic-island signs in front and the hedge side untouched.
3. Light motion-compensated temporal smoothing of the creature layer (calms skin shimmer).
4. Voice (plate audio) + synced creature sound design, then encode.
Usage: final.py plate.mp4 ai.mp4 guide_alpha.mp4 final.mp4
"""
import subprocess, sys, numpy as np, cv2

plate_p, ai_p, alpha_p, out_p = sys.argv[1:5]
W, H, N = 1920, 1080, 832
SW, SH = 480, 270


def reader(path, fmt="rgb24", ch=3, w=W, h=H, vf=None):
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-vf", vf] if vf else []) + ["-f", "rawvideo", "-pix_fmt", fmt, "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while True:
        b = p.stdout.read(w * h * ch)
        if len(b) < w * h * ch:
            break
        yield np.frombuffer(b, np.uint8).reshape(h, w, ch)


def ss(x, a, b):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def small_gray(path):
    return np.stack([f[..., 0].astype(np.float32) for f in
                     reader(path, "gray", 1, SW, SH, f"scale={SW}:{SH}:flags=area")])


# ---------------- 1. time alignment on the background
Pg = small_gray(plate_p)
Ag = small_gray(ai_p)
K = len(Ag)
Asm = np.stack([f[..., 0] for f in reader(alpha_p, "rgb24", 3, SW, SH, f"scale={SW}:{SH}:flags=area")])
bgm = np.stack([~(cv2.dilate((a > 5).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) for a in Asm])


def norm(x, m):
    v = x[m]
    return (x - v.mean()) / (v.std() + 1e-3)


def cost(k, j):
    m = bgm[j]
    return float(np.abs(norm(Ag[k], m) - norm(Pg[j], m))[m].mean())


def best(k, lo, hi):
    js = list(range(max(0, lo), min(N, hi + 1)))
    c = [cost(k, j) for j in js]
    i = int(np.argmin(c))
    if 0 < i < len(c) - 1:
        d = c[i - 1] - 2 * c[i] + c[i + 1]
        off = 0.5 * (c[i - 1] - c[i + 1]) / d if d > 1e-9 else 0.0
    else:
        off = 0.0
    return js[i] + off, c[i]


anchors = np.linspace(0, K - 1, 9).astype(int)
guess = lambda k: 1.2875 * k + 1.8
A_pts = np.array([(k, best(k, int(guess(k)) - 60, int(guess(k)) + 60)[0]) for k in anchors])
a0, b0 = np.polyfit(A_pts[:, 0], A_pts[:, 1], 1)
J = np.array([best(k, int(a0 * k + b0) - 10, int(a0 * k + b0) + 10)[0] for k in range(K)])
w = np.ones(K)
for _ in range(6):
    A_ = np.stack([np.arange(K), np.ones(K)], 1)
    coef = np.linalg.lstsq(A_ * w[:, None], J * w, rcond=None)[0]
    r = J - A_ @ coef
    s = np.median(np.abs(r)) * 1.4826 + 1e-6
    w = 1 / (1 + (r / (2.5 * s)) ** 2)
alpha, beta = coef
print(f"AI frames {K}; plate index = {alpha:.4f} * k + {beta:.2f}; residual sigma {s:.2f} frames; "
      f"AI covers plate {beta / 30:.2f}-{(alpha * (K - 1) + beta) / 30:.2f} s", flush=True)
del Pg, Ag

# ---------------- sign tracking (same as the guide composite)
CY0, CY1, CX0, CX1 = 480, 960, 860, 1640
gray, f210 = [], None
for n, fr in enumerate(reader(plate_p)):
    gray.append(cv2.cvtColor(fr[CY0:CY1, CX0:CX1], cv2.COLOR_RGB2GRAY))
    if n == 210:
        f210 = fr.copy()
f = f210.astype(np.float32) / 255
r, g, b = f[..., 0], f[..., 1], f[..., 2]
sat = (f.max(-1) - f.min(-1)) / np.maximum(f.max(-1), 1e-6)
L = 0.2126 * r + 0.7152 * g + 0.0722 * b
col = (sat > 0.35) & (((r > 0.45) & (g > 0.38) & (b < 0.6 * g)) | ((r > g + 0.15) & (r > b + 0.15)) | ((b > r + 0.12) & (b > g + 0.04)))
hull = np.zeros((H, W), bool)
for (y0, y1, x0, x1) in ((634, 646, 1133, 1143), (646, 675, 1123, 1147), (675, 708, 1112, 1157), (708, 750, 1117, 1151)):
    hull[y0:y1, x0:x1] = True
mA = hull & ((L >= 0.68) | col | ((L >= 0.5) & (sat >= 0.3)))
mA = cv2.morphologyEx(mA.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool) & hull
refA = mA.astype(np.float32); refA[750:788, 1134:1140] = 1
refB = np.zeros((H, W), np.float32); refB[601:634, 995:1031] = 1; refB[633:761, 1010:1015] = 1
T = {"A": (1108, 632, 1162, 752), "B": (990, 598, 1034, 636)}
off = {k: np.zeros((N, 2)) for k in T}
for k, (x0, y0, x1, y1) in T.items():
    tpl = gray[210][y0 - CY0:y1 - CY0, x0 - CX0:x1 - CX0].astype(np.float32)
    th, tw = tpl.shape
    for d in (1, -1):
        pos = np.zeros(2); vel = np.zeros(2); n = 210 + d
        while 0 <= n < N:
            pred = pos + vel; m = 48
            sx0 = int(round(x0 + pred[0])) - m; sy0 = int(round(y0 + pred[1])) - m
            if sx0 < CX0 or sy0 < CY0 or sx0 + tw + 2 * m > CX1 or sy0 + th + 2 * m > CY1:
                pos = pred
            else:
                cc = cv2.matchTemplate(gray[n][sy0 - CY0:sy0 - CY0 + th + 2 * m, sx0 - CX0:sx0 - CX0 + tw + 2 * m].astype(np.float32), tpl, cv2.TM_CCOEFF_NORMED)
                _, v, _, (bx, by) = cv2.minMaxLoc(cc)
                new = np.array([sx0 + bx - x0, sy0 + by - y0], float) if v >= 0.5 else pred
                vel = new - pos; pos = new
            off[k][n] = pos; n += d
    for c in range(2):
        off[k][:, c] = np.convolve(np.pad(off[k][:, c], 1, mode="edge"), [0.25, 0.5, 0.25], "valid")
del gray

# ---------------- 2-3. per output frame (24 fps on the plate timeline)
end_t = min(N / 30.0, (alpha * (K - 1) + beta) / 30.0)
NO = int(end_t * 24)
enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "24", "-i", "-",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "video_only.mp4"], stdin=subprocess.PIPE)
VV = np.arange(H, dtype=np.float32)[:, None]
UU = np.arange(W, dtype=np.float32)[None, :]
gp, ga, gl = reader(plate_p), reader(ai_p), reader(alpha_p)
pi = ai_i = al_i = -1
pl = ai = al = None
prev_layer = prev_small = None
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
for n in range(NO):
    t = n / 24.0
    p = min(N - 1, int(round(t * 30)))
    k = int(np.clip(round((p - beta) / alpha), 0, K - 1))
    while pi < p:
        pl = next(gp); pi += 1
    while al_i < p:
        al = next(gl); al_i += 1
    while ai_i < k:
        ai = next(ga); ai_i += 1
    P = pl.astype(np.float32); A = ai.astype(np.float32)
    region = cv2.dilate((al[..., 0] > 5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (71, 71))) > 0
    bg = ~region
    # colour-match the AI frame to the plate (per-channel gain/offset fitted on the background)
    idx = np.nonzero(bg[::4, ::4].ravel())[0]
    Ac = np.empty_like(A)
    for c in range(3):
        x = A[::4, ::4, c].ravel()[idx]; y = P[::4, ::4, c].ravel()[idx]
        gsl, off0 = np.polyfit(x, y, 1)
        Ac[..., c] = A[..., c] * gsl + off0
    # what the AI changed (creature + its shadow), inside the region only
    D = np.abs(Ac - P).max(-1)
    md = cv2.GaussianBlur(ss(D, 10, 28).astype(np.float32), (0, 0), 2.5)
    body = cv2.dilate((al[..., 1] > 128).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32)
    m = np.maximum(md, body) * cv2.GaussianBlur(region.astype(np.float32), (0, 0), 8)
    # plate objects in front: island signs; the hedge side stays untouched
    occ = np.zeros((H, W), np.float32)
    for kk, ref in (("A", refA), ("B", refB)):
        M = np.float32([[1, 0, off[kk][p, 0]], [0, 1, off[kk][p, 1]]])
        occ = np.maximum(occ, cv2.warpAffine(ref, M, (W, H), flags=cv2.INTER_LINEAR))
    uline = 20.19 + 1.2444 * (VV - off["A"][p, 1]) + off["A"][p, 0]
    hedge_ok = ss(UU - (uline - 40), 0, 12)
    m = np.clip(m * (1 - cv2.GaussianBlur(occ, (3, 3), 0.6)) * hedge_ok, 0, 1)
    # motion-compensated recursive smoothing of the creature layer (only where consistent)
    cur_small = cv2.resize(cv2.cvtColor(np.clip(Ac, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY), (W // 2, H // 2))
    layer = Ac
    if prev_layer is not None:
        fl = cv2.calcOpticalFlowFarneback(cur_small, prev_small, None, 0.5, 3, 21, 3, 5, 1.1, 0)
        fl = cv2.resize(fl, (W, H)) * 2
        warped = cv2.remap(prev_layer, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        agree = 1 - ss(np.abs(warped - Ac).mean(-1), 6, 18)
        wgt = (0.35 * agree * (m > 0.05))[..., None]
        layer = Ac * (1 - wgt) + warped * wgt
    prev_layer, prev_small = layer, cur_small
    out = P * (1 - m[..., None]) + layer * m[..., None]
    enc.stdin.write(np.clip(out + 0.5, 0, 255).astype(np.uint8).tobytes())
    if n % 60 == 0:
        print("final", n, "/", NO, "plate", p, "ai", k, flush=True)
enc.stdin.close(); enc.wait()
print("video done", NO, "frames at 24 fps,", round(NO / 24, 2), "s", flush=True)
