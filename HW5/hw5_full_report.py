
import os, io, math, time, datetime, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Preformatted, PageBreak
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import utils
from reportlab.lib.units import inch

# ---------------- Helpers ----------------
def img_for_report(path, max_width=6*inch):
    img = utils.ImageReader(path)
    iw, ih = img.getSize()
    scale = min(1.0, max_width/iw)
    return Image(path, width=iw*scale, height=ih*scale)

# ---------------- Q1: Single tanh neuron (2 epochs) ----------------
def q1_train_and_log(lr=0.1, epochs=2):
    X = np.array([[0.,1.],[1.,2.],[0.,-1.],[-1.,0.]], float)
    y = np.array([1.,1.,-1.,-1.], float)
    w = np.array([-1., 1.], float)
    log_lines = []
    log_lines.append(f"Initial w = {w.tolist()}, learning rate = {lr}, epochs = {epochs}\n")
    log_lines.append("Update log (per sample):")
    def forward(w,x):
        u = float(w@x); o = math.tanh(u); return u,o
    for ep in range(1, epochs+1):
        for i,(xi,yi) in enumerate(zip(X,y), start=1):
            u,o = forward(w, xi)
            grad = (o-yi)*(1.0-o*o)*xi
            w = w - lr*grad
            log_lines.append(f" epoch {ep}, sample {i}, x={xi.tolist()}, y={yi:+.0f}: u={u:+.6f}, o=tanh(u)={o:+.6f}")
            log_lines.append(f"   grad=(o-y)*(1-o^2)*x = {grad}")
            log_lines.append(f"   new w = {w}")
    def predict_label(w,x): return 1.0 if math.tanh(float(w@x))>=0 else -1.0
    preds = np.array([predict_label(w, xi) for xi in X], float)
    acc = float((preds==y).mean())
    log_lines.append(f"\nFinal w after {epochs} epochs: {w}")
    log_lines.append(f"Predicted labels: {preds.tolist()}  |  True labels: {y.tolist()}")
    log_lines.append(f"Training accuracy: {acc*100:.2f}%")
    return w, acc, "\n".join(log_lines)

# ---------------- Q2/Q3: Fashion MNIST ----------------
def load_fashion_mnist():
    # Tries keras.datasets first; if offline, raises and instructs user.
    try:
        import tensorflow as tf
        (x_train,y_train),(x_test,y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype("float32")/255.0
        x_test  = x_test.astype("float32")/255.0
        x_train = x_train.reshape((-1, 28*28))
        x_test  = x_test.reshape((-1, 28*28))
        return (x_train,y_train),(x_test,y_test)
    except Exception as e:
        raise RuntimeError("Could not load Fashion MNIST. Ensure internet access or provide a local copy via tf.keras.datasets cache.") from e

def build_model_3layers(input_dim=784, num_classes=10):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    logits = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, logits, name="fashion_mnist_3layer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model

def build_model_5layers(input_dim=784, num_classes=10):
    import tensorflow as tf
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(512, activation="relu")(inputs)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    logits = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, logits, name="fashion_mnist_5layer")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    return model

def capture_model_summary(model):
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + "\n"))
    return buf.getvalue()

def confusion_matrix_fig(y_true, y_pred, out_png):
    from sklearn.metrics import confusion_matrix
    import numpy as np, matplotlib.pyplot as plt
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    fontsize=7, color=("white" if cm[i,j]>cm.max()/2 else "black"))
    plt.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
    return out_png, cm

def train_eval_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    import numpy as np
    history = model.fit(x_train, y_train, validation_split=0.1,
                        epochs=epochs, batch_size=batch_size, verbose=2)
    logits = model.predict(x_test, batch_size=512, verbose=0)
    y_pred = np.argmax(logits, axis=1)
    acc = float((y_pred == y_test).mean())
    return acc, y_pred, history.history

def gpu_name_or_none():
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        return gpus[0].name if gpus else None
    except Exception:
        return None

# ---------------- Build Report ----------------
def main():
    # Prepare report
    doc = SimpleDocTemplate("HW5_ECE491_Report.pdf", pagesize=LETTER,
                            leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=8, leading=10))
    story = []
    story.append(Paragraph("ECE 491 — Homework 5 Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyText"]))
    story.append(Spacer(1,12))

    # Q1
    story.append(Paragraph("<b>Q1: Single tanh neuron (2 epochs)</b>", styles["Heading2"]))
    w, acc, q1_log = q1_train_and_log(lr=0.1, epochs=2)
    story.append(Paragraph(f"Final w = {w}, Training accuracy = {acc*100:.2f}%", styles["BodyText"]))
    story.append(Paragraph("<i>Derivation</i>: For o=tanh(u), u=w^T x, "
                           "dE/dw = (o - y)*(1 - o^2)*x; update w <- w - η dE/dw.", styles["BodyText"]))
    story.append(Spacer(1,6))
    story.append(Preformatted(q1_log, styles["Mono"]))

    # Q2/Q3
    story.append(PageBreak())
    story.append(Paragraph("<b>Q2 & Q3: Fashion MNIST (3-layer vs 5-layer)</b>", styles["Heading2"]))
    story.append(Paragraph("Loss: Sparse Categorical Cross-Entropy (from logits=True). Optimizer: Adam, LR=1e-3. "
                           "Training: epochs=10, batch_size=128.", styles["BodyText"]))
    gname = gpu_name_or_none()
    story.append(Paragraph(f"Detected GPU: {gname if gname else 'None'}", styles["BodyText"]))
    story.append(Spacer(1,6))

    try:
        (x_train,y_train),(x_test,y_test) = load_fashion_mnist()

        # 3-layer
        import tensorflow as tf
        m3 = build_model_3layers()
        sum3 = capture_model_summary(m3)
        t0 = time.time()
        acc3, ypred3, hist3 = train_eval_model(m3, x_train, y_train, x_test, y_test, epochs=10, batch_size=128)
        t3 = time.time()-t0
        png3, cm3 = confusion_matrix_fig(y_test, ypred3, "cm_3layer.png")
        story.append(Paragraph("<b>3-layer network</b>", styles["Heading3"]))
        story.append(Preformatted(sum3, styles["Mono"]))
        story.append(Paragraph(f"Test accuracy: {acc3:.4f}  |  Train time: {t3:.1f} s", styles["BodyText"]))
        story.append(Image(png3, width=4.5*inch, height=4.5*inch))
        story.append(Spacer(1,6))

        story.append(PageBreak())

        # 5-layer (same loss/optimizer/params)
        m5 = build_model_5layers()
        sum5 = capture_model_summary(m5)
        t0 = time.time()
        acc5, ypred5, hist5 = train_eval_model(m5, x_train, y_train, x_test, y_test, epochs=10, batch_size=128)
        t5 = time.time()-t0
        png5, cm5 = confusion_matrix_fig(y_test, ypred5, "cm_5layer.png")
        story.append(Paragraph("<b>5-layer network</b>", styles["Heading3"]))
        story.append(Preformatted(sum5, styles["Mono"]))
        story.append(Paragraph(f"Test accuracy: {acc5:.4f}  |  Train time: {t5:.1f} s", styles["BodyText"]))
        story.append(Image(png5, width=4.5*inch, height=4.5*inch))
        story.append(Spacer(1,6))

        story.append(Paragraph("<b>Did accuracy improve?</b>", styles["Heading3"]))
        story.append(Paragraph(
            "Typically the 5-layer model can improve accuracy because it has greater capacity to learn complex patterns. "
            "However, gains depend on regularization and optimization; deeper models may also overfit.", styles["BodyText"]))

    except Exception as e:
        story.append(Paragraph("<b>Note:</b> Could not load Fashion MNIST in this environment.", styles["BodyText"]))
        story.append(Paragraph("To complete Q2/Q3 locally: ensure internet access for tf.keras.datasets.fashion_mnist "
                               "or pre-download the dataset into Keras cache. Then re-run this script to populate "
                               "accuracy, confusion matrices, and model summaries.", styles["BodyText"]))
        story.append(Preformatted(str(e), styles["Mono"]))

    doc.build(story)
    print("Wrote HW5_ECE491_Report.pdf")

if __name__ == "__main__":
    main()
