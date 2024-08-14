import pandas as pd
import constants


data = pd.read_csv(constants.HELICOPTER_DATA_PATH + "timings.csv")

labels = data["ParticipantID", "Session", "PassageOfTimeSlowFast"]
for i in range(labels.shape[0]):
    if labels["PassageOfTimeSlowFast"][i] < 3:
        labels["PassageOfTimeSlowFast"][i] = 0
    else:
        labels["PassageOfTimeSlowFast"][i] = 1


labels.to_csv(constants.HELICOPTER_DATA_PATH + "labels.csv", index=False)