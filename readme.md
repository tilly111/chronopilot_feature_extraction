# Feature extraction pipeline for the ChronoPilot project

## How to run 
Please see ``main.py`` for arguments and how to run the feature extraction pipeline.

## Helicopter study

The `helicopter_featues` folder contains all preprocessing scripts used for the helicopter study.

You can read about it here: [Helicopter study](https://arxiv.org/pdf/2404.15213).

Cite as:

```
@InProceedings {Aust2024,
    title        = {Automatic classification of subjective time perception using multi-modal physiological data of air traffic controllers},
    author       = {Aust, Till and Balta, Eirini and Vatakis, Argiro and Hamann, Heiko},
    booktitle    = {2024 IEEE Int. Conf. on Systems, Man, and Cybernetics (SMC)},
    year         = {2024},
    pages        = {3962--3967},
    doi          = {10.1109/SMC54092.2024.10831642},
}
```

## Robot behavior study

The `robot_behavior_features` folder contains all preprocessing scripts used for the robot behavior study.

## Eye tracking data

[Under development]

The `eye_tracking` folder contains all preprocessing scripts used for the eye tracking data.

The file `calc_eye_tracking_features.py` contains the preprocessing pipeline for the eye tracking data.

## Label encoding

Here we shortly define how the labels are encoded in the different studies.

### Perceived passage of time (PPOT)

The PPOT labels are encoded as follows:

|     Study      | n_classes | Class |                       Explanation                       |
|:--------------:|:---------:|:-----:|:-------------------------------------------------------:|
|   Helicopter   |     2     |   0   |  Slow PPOT; **Threshold:** ≤ 3; **Scale:** Likert 1–5   |
|                |           |   1   |  Fast PPOT; **Threshold:** > 3; **Scale:** Likert 1–5   |
|                |     3     |   0   |  Slow PPOT; **Threshold:** ≤ 3; **Scale:** Likert 1–5   |
|                |           |   1   | Medium PPOT; **Threshold:** == 3; **Scale:** Likert 1–5 |
|                |           |   2   |  Fast PPOT; **Threshold:** > 3; **Scale:** Likert 1–5   |
| Robot behavior |     2     |   0   |  Slow PPOT; **Threshold:** ≤ 3; **Scale:** Likert 1–5   |
|                |           |   1   |  Fast PPOT; **Threshold:** > 3; **Scale:** Likert 1–5   |
|                |     3     |   0   |  Slow PPOT; **Threshold:** ≤ 3; **Scale:** Likert 1–5   |
|                |           |   1   | Medium PPOT; **Threshold:** == 3; **Scale:** Likert 1–5 |
|                |           |   2   |  Fast PPOT; **Threshold:** > 3; **Scale:** Likert 1–5   |

### Duration estimate

The duration estimate is calculated as the difference between the estimated time (ET) and the actual time (AT): <br>
`duration_estimate = (ET - AT)/AT`

The duration estimate labels are encoded as follows:

|     Study      | n_classes | Class |                              Explanation                               |
|:--------------:|:---------:|:-----:|:----------------------------------------------------------------------:|
|   Helicopter   |     2     |   0   |      Under estimation; **Threshold:** ≤ 0.9 ; **Scale:** relative      |
|                |           |   1   |       Over estimation; **Threshold:** > 0.9; **Scale:** relative       |
|                |     3     |   0   |      Under estimation; **Threshold:** ≤ 0.75; **Scale:** relative      |
|                |           |   1   | Correct estimation; **Threshold:** ≤ 0.75, ≤ 1.05; **Scale:** relative |
|                |           |   2   |       Overestimation; **Threshold:** > 1.05; **Scale:** relative       |
| Robot behavior |     2     |   0   |      Under estimation; **Threshold:** ≤ 0.9 ; **Scale:** relative      |
|                |           |   1   |       Over estimation; **Threshold:** > 0.9; **Scale:** relative       |
|                |     3     |   0   |      Under estimation; **Threshold:** ≤ 0.75; **Scale:** relative      |
|                |           |   1   | Correct estimation; **Threshold:** ≤ 0.75, ≤ 1.05; **Scale:** relative |
|                |           |   2   |       Overestimation; **Threshold:** > 1.05; **Scale:** relative       |
|  Scream study  |     2     |   0   |               Under estimation; Temporal bisection task                |
|                |           |   1   |                Over estimation; Temporal bisection task                |
