# =========================
# Lab 1 — ECE 491
# Intro to Python & 1-D Convolution
# =========================
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# ---------- Part A.1: u[n] and indexed version ----------
x = np.array([0.,0.,1.,1.,1.,1.,1.,1.,1.,1.])  # 10 samples (2 zeros then 8 ones)
print("A.1 vector x:", x)

plt.figure("A1: u[n]")
plt.stem(x, use_line_collection=True)
plt.title("A1: u[n] (stem over sample index n)")
plt.xlabel("n"); plt.ylabel("amplitude"); plt.pause(0.0001)

k = np.arange(-2., 8., 1.)
plt.figure("A1: k vs x")
plt.stem(k, x, use_line_collection=True)
plt.title("A1: u[n] with index k in [-2,7]")
plt.xlabel("k"); plt.ylabel("amplitude"); plt.pause(0.0001)

# ---------- Part A.2: u[n] and u[n-5], n ∈ [-10,20] ----------
n = np.arange(-10, 21)
u = (n >= 0).astype(float)
u_shift = (n-5 >= 0).astype(float)

plt.figure("A2: u[n] and u[n-5]")
plt.stem(n, u, linefmt='C0-', markerfmt='C0o', basefmt=' ', use_line_collection=True, label='u[n]')
plt.stem(n, u_shift, linefmt='C1-', markerfmt='C1s', basefmt=' ', use_line_collection=True, label='u[n-5]')
plt.legend(); plt.title("A2: u[n] vs u[n-5]"); plt.xlabel("n"); plt.ylabel("amplitude"); plt.pause(0.0001)

# ---------- Part A.3: x[n] = 7cos(0.1n) + cos(0.95n) and x[n-20] ----------
n = np.arange(-40, 81)
x_mix = 7*np.cos(0.1*n) + np.cos(0.95*n)
x_mix_shift = 7*np.cos(0.1*(n-20)) + np.cos(0.95*(n-20))

plt.figure("A3: x[n] and x[n-20]")
plt.plot(n, x_mix, label='x[n]')
plt.plot(n, x_mix_shift, label='x[n-20]', alpha=0.8)
plt.legend(); plt.title("A3: Cosine Mixture and Shift"); plt.xlabel("n"); plt.ylabel("amplitude"); plt.pause(0.0001)

# ---------- Part B: 1-D Convolution ----------
h_avg  = (1./5.)*np.array([1.,1.,1.,1.,1.])  # moving average (length 5)
h_diff = np.array([1., -1.])                 # difference

def plot_conv(n_axis, x_in, h, h_name, fig_title):
    y = np.convolve(x_in, h, mode='full')
    # Axis for 'full' length (rough alignment for viz):
    m = np.arange(n_axis[0], n_axis[0] + len(y))
    plt.figure(fig_title); plt.plot(m, y)
    plt.title(f"{fig_title}: y = x * {h_name}")
    plt.xlabel("n"); plt.ylabel("amplitude"); plt.pause(0.0001)

# (a) x = 1 for k = 0..20
k_a = np.arange(0, 21); x_a = np.ones_like(k_a, dtype=float)
plot_conv(k_a, x_a, h_avg,  "h_avg",  "B.2(a) Moving Average on x=1, k∈[0,20]")
plot_conv(k_a, x_a, h_diff, "h_diff", "B.3(a) Difference on x=1, k∈[0,20]")

# (b) x = cos(0.1k), k e= -40..80
k_b = np.arange(-40, 81); x_b = np.cos(0.1*k_b)
plot_conv(k_b, x_b, h_avg,  "h_avg",  "B.2(b) Moving Average on cos(0.1k)")
plot_conv(k_b, x_b, h_diff, "h_diff", "B.3(b) Difference on cos(0.1k)")

# (c) x = cos(0.95k), k = -40..80
k_c = np.arange(-40, 81); x_c = np.cos(0.95*k_c)
plot_conv(k_c, x_c, h_avg,  "h_avg",  "B.2(c) Moving Average on cos(0.95k)")
plot_conv(k_c, x_c, h_diff, "h_diff", "B.3(c) Difference on cos(0.95k)")

# (d) x is the signal in Part A.3
plot_conv(np.arange(-40,81), x_mix, h_avg,  "h_avg",  "B.2(d) Moving Average on x_mix")
plot_conv(np.arange(-40,81), x_mix, h_diff, "h_diff", "B.3(d) Difference on x_mix")

print("""
B.4 Quick observations:
- Moving average (low-pass) smooths signals, reducing high-frequency content/noise.
- Difference (high-pass) emphasizes changes/edges; it attenuates constants/slow trends.
""")

plt.show()
