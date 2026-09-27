"""Procedural creature sound design, synced to the animation (48 kHz stereo).

Footsteps at the animation's foot contacts (distance attenuation, air absorption,
sound travel delay, pan from screen position), breathing, a snort at 11.3 s and a
heavy exhale / low rumble at 17.8 s. Ducked under the voice of the plate audio.
Usage: sfx.py plate_audio.wav sfx.wav
"""
import sys, numpy as np, wave

SR = 48000
DUR = 832 / 30
N = int(DUR * SR)
rng = np.random.default_rng(7)
# foot contacts from the animation: time (s), distance to camera (m), screen x (0..1), speed (m/s)
C = [(0.267, 47.4, 0.652, 0.43), (1.3, 46.0, 0.683, 1.94), (2.233, 44.0, 0.652, 2.11), (3.167, 42.1, 0.655, 2.11),
     (4.067, 40.1, 0.572, 2.12), (5.0, 38.4, 0.556, 2.11), (5.933, 36.2, 0.515, 2.1), (6.867, 34.8, 0.566, 2.1),
     (7.7, 32.7, 0.553, 2.11), (8.6, 32.1, 0.6, 1.62), (9.7, 30.5, 0.577, 1.0), (11.567, 31.8, 0.614, 0.5),
     (13.533, 29.8, 0.599, 2.2), (14.633, 28.3, 0.57, 2.2), (22.067, 27.5, 0.752, 1.57), (23.167, 25.4, 0.711, 1.6),
     (24.267, 24.4, 0.786, 1.6), (25.333, 22.3, 0.761, 1.6), (26.433, 21.3, 0.835, 1.6), (27.567, 20.0, 0.786, 1.6)]
HEAVY = {13.533, 14.633}
LIGHT = {11.567}   # the small repositioning step: a gravel crunch more than a thud


def band(x, lo, hi):
    """Zero-phase FFT band-pass with soft (raised-cosine) edges."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X = np.fft.rfft(x)
    w = np.ones_like(f)
    if lo > 0:
        w *= np.clip((f - lo * 0.7) / (lo * 0.6), 0, 1)
    if hi < SR / 2:
        w *= np.clip((hi * 1.4 - f) / (hi * 0.8), 0, 1)
    return np.fft.irfft(X * w, n)


def env(n, att, dec):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(att, 1e-4)) * np.exp(-np.maximum(0, t - att) / dec)


L = np.zeros(N); R = np.zeros(N)


def place(sig, t0, pan, gain):
    """Constant-power pan (pan -1..1) and add at time t0 (s)."""
    i = int(t0 * SR)
    if i >= N:
        return
    sig = sig[: N - i] * gain
    a = (pan + 1) * np.pi / 4
    L[i:i + len(sig)] += np.cos(a) * sig
    R[i:i + len(sig)] += np.sin(a) * sig


def footstep(dist, speed, heavy):
    n = int(1.1 * SR)
    t = np.arange(n) / SR
    f = 75 + 45 * np.exp(-t / 0.05)                             # pitch drop 120 -> 75 Hz (phone mics cut the infra-bass)
    thump = np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.004, 0.20)
    body = band(rng.standard_normal(n), 90, 260) * env(n, 0.003, 0.07) * 1.6   # audible on phone speakers
    cut = 1600 - 22 * (dist - 18)                                # air absorption with distance
    crunch = band(rng.standard_normal(n), 250, max(500, cut)) * env(n, 0.002, 0.045) * 0.35
    rumble = band(rng.standard_normal(n), 70, 140) * env(n, 0.03, 0.40) * 1.2
    s = thump + body + crunch + rumble
    w = (1.35 if heavy else 1.0) * (0.75 + 0.25 * min(1, speed / 2.2))
    return s / np.abs(s).max() * w


for (t, dist, sx, speed) in C:
    g = min(1.0, 14.0 / dist) * 0.42
    place(footstep(dist, speed, t in HEAVY), t + dist / 343.0, np.clip((sx - 0.5) * 1.4, -0.8, 0.8), g * (0.5 if t in LIGHT else 1.0))


def breath(n, lo, hi, att, dec, grit=0.0):
    x = band(rng.standard_normal(n), lo, hi) * env(n, att, dec)
    if grit:
        am = 1 + grit * np.sin(2 * np.pi * 28 * np.arange(n) / SR)  # nostril flutter
        x *= am
    return x / np.abs(x).max()


# breathing: exhale at each chest maximum of the animation (2.4 s walking, 3.4 s standing)
def speed_at(t):
    if t < 8.3 or 12.3 <= t < 14.4 or t >= 20.8:
        return 1.0
    return 0.0


t = 0.6
while t < DUR - 1:
    T = 2.4 if speed_at(t) else 3.4
    head_d = 45 - 1.0 * min(t, 20) * (0.9 if t < 9.4 else 0.5)
    x = breath(int(1.4 * SR), 110, 750, 0.25, 0.55)
    place(x, t, 0.2, min(1, 14 / max(head_d, 18)) * 0.10)
    t += T
# snort toward the houses (11.3 s): sharp nasal burst + low grunt
n = int(0.9 * SR)
sn = breath(n, 300, 3200, 0.012, 0.16, grit=0.6) + 0.8 * breath(n, 70, 180, 0.02, 0.25)
place(sn / np.abs(sn).max(), 11.40 + 26.2 / 343, 0.2, 0.40)
place(breath(int(0.8 * SR), 150, 900, 0.18, 0.35), 11.95, 0.2, 0.12)       # recovery inhale
# heavy exhale + very low rumble in the silence before "alerter" (17.8 s)
n = int(1.8 * SR)
tt = np.arange(n) / SR
rum = np.sign(np.sin(2 * np.pi * 88 * tt)) * (0.6 + 0.4 * np.sin(2 * np.pi * 3.1 * tt))
rum = band(rum, 80, 320) * env(n, 0.25, 0.7)
ex = breath(n, 100, 900, 0.3, 0.6) + 0.9 * rum / np.abs(rum).max()
place(ex / np.abs(ex).max(), 18.05 + 24.1 / 343, 0.45, 0.34)
# faint low rumble while it sniffs the asphalt (9.9 s)
n = int(1.8 * SR)
place(band(rng.standard_normal(n), 75, 150) * env(n, 0.5, 0.8) * 0.9, 9.9, 0.15, 0.20)
# short nasal sniffs in the rhythm of the head nudges (animation: 1.7 Hz)
for t0, t1, dist, pan in ((9.9, 10.8, 30.0, 0.15), (18.45, 20.3, 24.1, 0.45)):
    k = 0
    while True:
        ts = (0.25 + k) / 1.7
        k += 1
        if ts > t1:
            break
        if ts >= t0:
            place(breath(int(0.25 * SR), 700, 4200, 0.02, 0.06), ts + dist / 343, pan, min(1, 14 / dist) * 0.13)

# duck under the voice
src = wave.open(sys.argv[1], "rb")
va = np.frombuffer(src.readframes(src.getnframes()), np.int16).astype(np.float32) / 32768
if src.getnchannels() == 2:
    va = va.reshape(-1, 2).mean(1)
if src.getframerate() != SR:
    va = np.interp(np.arange(N) / SR, np.arange(len(va)) / src.getframerate(), va)
va = np.pad(va, (0, max(0, N - len(va))))[:N]
win = int(0.05 * SR)
rms = np.sqrt(np.convolve(va ** 2, np.ones(win) / win, "same"))
act = np.clip((20 * np.log10(rms + 1e-6) + 42) / 12, 0, 1)            # 0 below -42 dBFS, 1 above -30 dBFS
k = np.exp(-np.arange(int(0.25 * SR)) / (0.08 * SR)); k /= k.sum()
act = np.convolve(act, k, "same")
duck = 1 - 0.45 * act
out = np.stack([band(L, 70, 12000) * duck, band(R, 70, 12000) * duck], 1)   # phone-mic band
peak = np.abs(out).max()
out *= min(1.0, 0.5 / peak)                                             # SFX bus peaks at -6 dBFS max
print("sfx peak dBFS", round(20 * np.log10(np.abs(out).max() + 1e-9), 1), "rms dBFS", round(20 * np.log10(np.sqrt((out ** 2).mean()) + 1e-9), 1))
w = wave.open(sys.argv[2], "wb")
w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((np.clip(out, -1, 1) * 32767).astype(np.int16).tobytes())
w.close()
