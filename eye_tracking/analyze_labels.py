import platform
import matplotlib
import matplotlib.pyplot as plt
import os

import numpy as np

import constants
import pandas as pd


# for interactive plots
if platform.system() == "Darwin":
    matplotlib.use('QtAgg')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    plt.rcParams.update({'font.size': 22})
elif platform.system() == "Linux":
    matplotlib.use('TkAgg')

# in theory 600 settings
labels = pd.read_csv("/Volumes/Data/chronopilot/Julia_study/features/all_labels.csv")
labels.dropna(inplace=True)

l, counts = np.unique(labels["ppot"], return_counts=True)
# print(l, counts, np.sum(counts))
l = ["very slow", "slow", "medium", "fast", "very fast"]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].bar(l, counts)
axs[0].set_title("PPOT distribution")

# over under estimation
counts = np.zeros(3)
l = ["under", "correct", "over"]
for i, row in labels.iterrows():
    if row["duration_estimate"] < row["time"] * 0.75:  # account for slight underestimation in all settings
        counts[0] += 1
    elif row["duration_estimate"] > row["time"] * 1.05:
        counts[2] += 1
    else:
        counts[1] += 1

axs[1].bar(l, counts)
axs[1].set_title("Over/Underestimation distribution")


l, counts = np.unique(labels["ppot"], return_counts=True)
counts_2 = [counts[0] + counts[1] + counts[2], counts[3] + counts[4]]
l = ["slow", "fast"]
print(l, counts_2, np.sum(counts_2))
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].bar(l, counts_2)
axs[0].set_title("PPOT distribution 2 classes")

counts_3 = [counts[0] + counts[1], counts[2], counts[3] + counts[4]]
l = ["slow", "medium", "fast"]
print(l, counts_3, np.sum(counts_3))
axs[1].bar(l, counts_3)
axs[1].set_title("PPOT distribution 3 classes")


# over under estimation
counts = np.zeros(2)
l = ["under", "over"]
for i, row in labels.iterrows():
    if row["duration_estimate"] < row["time"] * 0.9:  # account for slight underestimation in all settings
        counts[0] += 1
    else:
        counts[1] += 1

fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].bar(l, counts)
axs[0].set_title("Over/Underestimation 2 classes")

counts = np.zeros(3)
l = ["under", "correct", "over"]
for i, row in labels.iterrows():
    if row["duration_estimate"] < row["time"] * 0.75:  # account for slight underestimation in all settings
        counts[0] += 1
    elif row["duration_estimate"] > row["time"] * 1.05:
        counts[2] += 1
    else:
        counts[1] += 1

axs[1].bar(l, counts)
axs[1].set_title("Over/Underestimation 3 classes")


plt.show()



plt.figure()
plt.hist(labels[labels["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="time 1", color="blue")
plt.vlines(1, 0, 37, color="blue")
plt.hist(labels[labels["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="time 3", color="orange")
plt.vlines(3, 0, 37, color="orange")
plt.hist(labels[labels["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="time 5", color="green")
plt.vlines(5, 0, 37, color="green")
plt.legend()
plt.title("Duration estimates")


# ppot distribution for time
ppot_time_1 = labels[labels["time"] == 1]["ppot"]
ppot_time_3 = labels[labels["time"] == 3]["ppot"]
ppot_time_5 = labels[labels["time"] == 5]["ppot"]

l, counts = np.unique(ppot_time_1, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
label = ["very slow", "slow", "medium", "fast", "very fast"]
_, axs = plt.subplots(1, 3, figsize=(12, 5))

axs[0].bar(label, counts)
axs[0].set_title("PPOT distribution time 1")
axs[0].tick_params(axis='x', rotation=90)
l, counts = np.unique(ppot_time_3, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[1].bar(label, counts)
axs[1].set_title("PPOT distribution time 3")
axs[1].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_time_5, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[2].bar(label, counts)
axs[2].set_title("PPOT distribution time 5")
axs[2].tick_params(axis='x', rotation=90)

plt.tight_layout()

# ppot distribution for robot
ppot_robot_1 = labels[labels["robot"] == 1]["ppot"]
ppot_robot_3 = labels[labels["robot"] == 3]["ppot"]
ppot_robot_5 = labels[labels["robot"] == 5]["ppot"]
ppot_robot_7 = labels[labels["robot"] == 7]["ppot"]
ppot_robot_9 = labels[labels["robot"] == 9]["ppot"]
ppot_robot_11 = labels[labels["robot"] == 11]["ppot"]
ppot_robot_13 = labels[labels["robot"] == 13]["ppot"]
ppot_robot_15 = labels[labels["robot"] == 15]["ppot"]

label = ["very slow", "slow", "medium", "fast", "very fast"]
fig, axs = plt.subplots(2, 4, figsize=(12, 5))
fig.suptitle("PPOT distribution for robots")

l, counts = np.unique(ppot_robot_1, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[0, 0].bar(label, counts)
axs[0, 0].set_title("1 robot")
axs[0, 0].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_robot_3, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[0, 1].bar(label, counts)
axs[0, 1].set_title("3 robot")
axs[0, 1].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_robot_5, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[0, 2].bar(label, counts)
axs[0, 2].set_title("5 robot")
axs[0, 2].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_robot_7, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[0, 3].bar(label, counts)
axs[0, 3].set_title("7 robot")
axs[0, 3].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_robot_9, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[1, 0].bar(label, counts)
axs[1, 0].set_title("9 robot")
axs[1, 0].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_robot_11, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[1, 1].bar(label, counts)
axs[1, 1].set_title("11 robot")
axs[1, 1].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_robot_13, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[1, 2].bar(label, counts)
axs[1, 2].set_title("13 robot")
axs[1, 2].tick_params(axis='x', rotation=90)

l, counts = np.unique(ppot_robot_15, return_counts=True)
c = np.zeros((5,))
c[l.astype(int)] = counts
counts = c
axs[1, 3].bar(label, counts)
axs[1, 3].set_title("15 robot")
axs[1, 3].tick_params(axis='x', rotation=90)

plt.tight_layout()

# duration estimation robots
duration_robot_1 = labels[labels["robot"] == 1][["duration_estimate", "time"]]
duration_robot_3 = labels[labels["robot"] == 3][["duration_estimate", "time"]]
duration_robot_5 = labels[labels["robot"] == 5][["duration_estimate", "time"]]
duration_robot_7 = labels[labels["robot"] == 7][["duration_estimate", "time"]]
duration_robot_9 = labels[labels["robot"] == 9][["duration_estimate", "time"]]
duration_robot_11 = labels[labels["robot"] == 11][["duration_estimate", "time"]]
duration_robot_13 = labels[labels["robot"] == 13][["duration_estimate", "time"]]
duration_robot_15 = labels[labels["robot"] == 15][["duration_estimate", "time"]]

fig, axs = plt.subplots(2, 4, figsize=(12, 5))
fig.suptitle("Duration estimation for robots")

axs[0, 0].hist(duration_robot_1[duration_robot_1["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[0, 0].vlines(1, 0, 10, color="blue")
axs[0, 0].hist(duration_robot_1[duration_robot_1["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[0, 0].vlines(3, 0, 10, color="orange")
axs[0, 0].hist(duration_robot_1[duration_robot_1["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[0, 0].vlines(5, 0, 10, color="green")
axs[0, 0].set_title("1 robot")

axs[0, 1].hist(duration_robot_3[duration_robot_3["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[0, 1].vlines(1, 0, 10, color="blue")
axs[0, 1].hist(duration_robot_3[duration_robot_3["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[0, 1].vlines(3, 0, 10, color="orange")
axs[0, 1].hist(duration_robot_3[duration_robot_3["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[0, 1].vlines(5, 0, 10, color="green")
axs[0, 1].set_title("3 robot")

axs[0, 2].hist(duration_robot_5[duration_robot_5["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[0, 2].vlines(1, 0, 10, color="blue")
axs[0, 2].hist(duration_robot_5[duration_robot_5["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[0, 2].vlines(3, 0, 10, color="orange")
axs[0, 2].hist(duration_robot_5[duration_robot_5["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[0, 2].vlines(5, 0, 10, color="green")
axs[0, 2].set_title("5 robot")

axs[0, 3].hist(duration_robot_7[duration_robot_7["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[0, 3].vlines(1, 0, 10, color="blue")
axs[0, 3].hist(duration_robot_7[duration_robot_7["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[0, 3].vlines(3, 0, 10, color="orange")
axs[0, 3].hist(duration_robot_7[duration_robot_7["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[0, 3].vlines(5, 0, 10, color="green")
axs[0, 3].set_title("7 robot")

axs[1, 0].hist(duration_robot_9[duration_robot_9["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[1, 0].vlines(1, 0, 10, color="blue")
axs[1, 0].hist(duration_robot_9[duration_robot_9["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[1, 0].vlines(3, 0, 10, color="orange")
axs[1, 0].hist(duration_robot_9[duration_robot_9["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[1, 0].vlines(5, 0, 10, color="green")
axs[1, 0].set_title("9 robot")

axs[1, 1].hist(duration_robot_11[duration_robot_11["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[1, 1].vlines(1, 0, 10, color="blue")
axs[1, 1].hist(duration_robot_11[duration_robot_11["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[1, 1].vlines(3, 0, 10, color="orange")
axs[1, 1].hist(duration_robot_11[duration_robot_11["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[1, 1].vlines(5, 0, 10, color="green")
axs[1, 1].set_title("11 robot")

axs[1, 2].hist(duration_robot_13[duration_robot_13["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[1, 2].vlines(1, 0, 10, color="blue")
axs[1, 2].hist(duration_robot_13[duration_robot_13["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[1, 2].vlines(3, 0, 10, color="orange")
axs[1, 2].hist(duration_robot_13[duration_robot_13["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[1, 2].vlines(5, 0, 10, color="green")
axs[1, 2].set_title("13 robot")

axs[1, 3].hist(duration_robot_15[duration_robot_15["time"] == 1]["duration_estimate"], bins=20, alpha=0.5, label="1 min", color="blue")
axs[1, 3].vlines(1, 0, 10, color="blue")
axs[1, 3].hist(duration_robot_15[duration_robot_15["time"] == 3]["duration_estimate"], bins=20, alpha=0.5, label="3 min", color="orange")
axs[1, 3].vlines(3, 0, 10, color="orange")
axs[1, 3].hist(duration_robot_15[duration_robot_15["time"] == 5]["duration_estimate"], bins=20, alpha=0.5, label="5 min", color="green")
axs[1, 3].vlines(5, 0, 10, color="green")
axs[1, 3].set_title("15 robot")

plt.tight_layout()
plt.show()