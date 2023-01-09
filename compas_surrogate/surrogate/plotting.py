import matplotlib.pyplot as plt

plt.rcParams["agg.path.chunksize"] = 10000


def plot_loss(history):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))
    ax.plot(history.history["loss"], label="loss", lw=5)
    ax.plot(history.history["val_loss"], label="val_loss", lw=5)
    plt.xlabel("Epoch")
    plt.ylabel("Error [MPG]")
    plt.legend()
    plt.grid(True)
