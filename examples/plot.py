import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

sb.set()

misc = "ep100_step1000_seq200_period1_"
alg = "legacy"
adjust = misc + alg


def moving_avg(x, window_size=3):
    i = 0
    moving_averages = []
    while i < len(x) - window_size + 1:
        this_window = x[i : i + window_size]
        # print(this_window)
        window_average = sum(this_window) / window_size
        moving_averages.append(window_average)
        i += 1
    return moving_averages


def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


legacy = []

with open(adjust + ".txt") as f:
    print(adjust + ".txt")
    for line in f:
        temp = float(line.strip())
        # print(temp)
        legacy.append(temp)
    # lines = f.readlines()

lim = len(legacy)  # - 1  # LAST IS TOTAL
y_leg = legacy[:lim]
x_leg = np.arange(lim)

alg = "trajectory"
adjust = misc + alg

trajectory = []

with open(adjust + ".txt") as f:
    for line in f:
        temp = float(line.strip())
        # print(temp)
        trajectory.append(temp)
    # lines = f.readlines()

lim = len(trajectory)  # - 1
y_traj = trajectory[:lim]
x_traj = np.arange(lim)
window = 100
smoothing = 0.99

alg = "sequence"
adjust = misc + alg

seq = []

with open(adjust + ".txt") as f:
    for line in f:
        temp = float(line.strip())
        # print(temp)
        seq.append(temp)
    # lines = f.readlines()

lim = len(seq)  # - 1
y_seq = seq[:lim]
x_seq = np.arange(lim)
window = 100
smoothing = 0.99

print(y_leg[-5:])

start = 0
plot_lim = len(y_leg) - 1  # 150

plt.plot(y_leg[start : plot_lim + start], color="r", label="Legacy")  # , alpha=0.5)
plt.plot(
    y_traj[start : plot_lim + start], color="g", label="Trajectory"
)  # , alpha=0.5)
plt.plot(y_seq[start : plot_lim + start], color="b", label="Sequence")  # , alpha=0.5)

# plt.scatter(
#     np.arange(len(y_leg[start : plot_lim + start])),
#     y_leg[start : plot_lim + start],
#     color="b",
#     label="Legacy",
# )  # , alpha=0.5)
# plt.scatter(
#     np.arange(len(y_traj[start : plot_lim + start])),
#     y_traj[start : plot_lim + start],
#     color="g",
#     label="Trajectory",
# )  # , alpha=0.5)

# plt.yscale("log")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Time (s)")
plt.title("Legacy vs Trajectory Writer")
# plt.grid()
plt.savefig(misc + "_plot.png")
plt.show()

zipped_lists = zip(y_leg, y_traj)
y_ratio_legtraj = [x / (x + y) for (x, y) in zipped_lists]

zipped_lists = zip(y_seq, y_traj)
y_ratio_seqtraj = [x / (x + y) for (x, y) in zipped_lists]


plt.clf()
plt.plot(
    y_ratio_legtraj[start : plot_lim + start],
    color="m",
    label="Ratio Legacy vs Trajectory",
    linewidth=1.0,
    alpha=0.5,
)
plt.plot(
    y_ratio_seqtraj[start : plot_lim + start],
    color="c",
    label="Ratio Sequence vs Trajectory",
    linewidth=1.0,
    alpha=0.5,
)
# y_ratio_legtraj = np.array(y_ratio_legtraj)
# col = np.where(y_ratio_legtraj < 0.5, "g", "r")[start : plot_lim + start]
# plt.scatter(
#     np.arange(len(y_ratio_legtraj[start : plot_lim + start])),
#     y_ratio_legtraj[start : plot_lim + start],
#     color=col,
#     label="Executor",
# )
plt.plot(
    np.convolve(
        y_ratio_legtraj[start : plot_lim + start],
        np.ones(window) / window,
        mode="valid",
    ),
    color="m",
)
plt.plot(
    np.convolve(
        y_ratio_seqtraj[start : plot_lim + start],
        np.ones(window) / window,
        mode="valid",
    ),
    color="c",
)
# plt.yscale("log")
# plt.scatter(np.arange(len(y_ratio_legtraj[:])), y_ratio_legtraj[:], color="g", label="Trainer")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Ratio")
plt.title("Legacy vs Trajectory Writer")
plt.yticks(np.arange(0.1, 1.0, 0.1))
plt.axhline(y=0.5, color="r", linestyle="-")
# plt.grid()
# print(min(y_ratio_legtraj), max(y_ratio_legtraj))
plt.savefig(misc + "_ratio.png")  # dpi=1000)
plt.show()


y_leg_cumsum = np.cumsum(y_leg)
y_seq_cumsum = np.cumsum(y_seq)
y_traj_cumsum = np.cumsum(y_traj)
print(len(y_traj) * y_traj[0])
print(np.sum(y_traj), y_traj_cumsum[-1])
plt.clf()
plt.plot(
    y_leg_cumsum[start : plot_lim + start],
    color="r",
    label="Legacy",
    linewidth=1.0,
)
plt.plot(
    y_traj_cumsum[start : plot_lim + start],
    color="g",
    label="Trajectory",
    linewidth=1.0,
)
plt.plot(
    y_seq_cumsum[start : plot_lim + start],
    color="b",
    label="Sequence",
    linewidth=1.0,
)

# plt.yscale("log")
# plt.scatter(np.arange(len(y_ratio_legtraj[:])), y_ratio_legtraj[:], color="g", label="Trainer")
plt.legend()
plt.xlabel("Step")
plt.ylabel("Time (s)")
plt.title("Cumulative time writers")
# plt.yticks(np.arange(0.1, 1.0, 0.1))
# plt.grid()
# print(min(y_ratio_legtraj), max(y_ratio_legtraj))
plt.savefig(misc + "_cumtime.png")  # dpi=1000)
plt.show()


plt.clf()
labels = ["Legacy", "Trajectory", "Sequence"]
bars = [y_leg[-1] / 60, y_traj[-1] / 60, y_seq[-1] / 60]
plt.bar(labels, bars, color=["r", "g", "b"])
# plt.yscale("log")
# plt.scatter(np.arange(len(y_ratio_legtraj[:])), y_ratio_legtraj[:], color="g", label="Trainer")
# plt.legend()
plt.xlabel("Step")
plt.ylabel("Time (m)")
plt.title("Total time writers")
# plt.yticks(np.arange(0.1, 1.0, 0.1))
# plt.grid()
# print(min(y_ratio_legtraj), max(y_ratio_legtraj))
plt.savefig(misc + "_tottime.png")  # dpi=1000)
plt.show()
