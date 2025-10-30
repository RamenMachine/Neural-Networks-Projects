import numpy as np
import matplotlib.pyplot as plt             
from mpl_toolkits.mplot3d import Axes3D
# Question 1 part B
blue = np.array([[-1, -1], [0, -1], [-1, 0]])
red  = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
X = np.vstack([blue, red])
y = np.hstack([-np.ones(len(blue)), np.ones(len(red))])

def perceptron(X, y, w=None, b=0.0, max_epochs=1000, lr=1.0, verbose=True):
    w = np.zeros(X.shape[1]) if w is None else np.array(w, dtype=float)
    for epoch in range(1, max_epochs + 1):
        mistakes = 0
        for xi, yi in zip(X, y):
            if yi * (w.dot(xi) + b) <= 0:
                w += lr * yi * xi
                b += lr * yi
                mistakes += 1
        if verbose:
            print(f"Epoch {epoch:02d} | mistakes: {mistakes}")
        if mistakes == 0:
            break
    return w, b

if __name__ == "__main__":
    w, b = perceptron(X, y, w=[-1.0, 1.0], verbose=True)
    print("Final w, b =", w, b)
    if abs(w[1]) > 1e-12:
        print("Decision boundary: x2 = {:.6g} * x1 + {:.6g}".format(-w[0]/w[1], -b/w[1]))
    else:
        print("Decision boundary: vertical line x1 =", -b/w[0])


# Question 2 part B

def z_from_xy(x, y):
    return (4 - 2*x - 3*y) / 4.0

x1 = np.linspace(-0.5, 2.5, 50)
x2 = np.linspace(-0.5, 1.8, 50)
X1, X2 = np.meshgrid(x1, x2)
X3 = z_from_xy(X1, X2)

p1 = np.array([2.0, 0.0, 0.0])
p2 = np.array([0.0, 4.0/3.0, 0.0])
p3 = np.array([0.0, 0.0, 1.0])

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, X3, alpha=0.5)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
ax.plot([p2[0], p3[0]], [p2[1], p3[1]], [p2[2], p3[2]])
ax.plot([p3[0], p1[0]], [p3[1], p1[1]], [p3[2], p1[2]])

ax.scatter([p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], [p1[2], p2[2], p3[2]], s=40)
ax.text(*p1, " (2,0,0)")
ax.text(*p2, " (0,4/3,0)")
ax.text(*p3, " (0,0,1)")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("Plane: 2x1 + 3x2 + 4x3 - 4 = 0")
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(-0.5, 1.8)
ax.set_zlim(-0.5, 1.3)

plt.show()

w = np.array([2, 3, 4])
b = -4
print("w =", w, "b =", b)
print("Intercepts:", tuple(p1), tuple(p2), tuple(p3))
