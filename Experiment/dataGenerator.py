import imp
import random
import numpy as np

periods = [7, 14, 28, 30]


def generate_time_series(dataframe):
    clip_val = random.uniform(0.3, 1)
    period = random.choice(periods)
    phase = random.randint(-1000, 1000)
    dataframe["views"] = dataframe.apply(
        lambda x: np.clip(
            np.cos(x["index"] * 2 * np.pi / period + phase), -
            clip_val, clip_val
        )
        * x["amplitude"]
        + x["offset"],
        axis=1,
    ) + np.random.normal(
        0, dataframe["amplitude"].abs().max() / 10, size=(dataframe.shape[0],)
    )
    return dataframe
