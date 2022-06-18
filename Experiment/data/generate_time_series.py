import random
import pandas as pd
import numpy as np
from uuid import uuid4
from statsmodels.tsa.seasonal import seasonal_decompose
# 显示进度条
from tqdm import tqdm


# 生成具有不同周期性，偏移和模式的非平凡时间序列
# 周期性时间序列数据可以使用由余弦函数来生成，单纯的余弦函数与现实相差很大，因为现实中的时序数据具有大量的噪声，因此需要加上随机振幅和随机偏移来生存具有噪声的时间序列数据


# 生成时间序列索引
def get_init_df():
    date_rng = pd.date_range(start="2021-06-01", end="2022-06-01", freq="H")
    # 时间戳
    dataframe = pd.DataFrame(date_rng, columns=["timestamp"])
    # 行数索引
    dataframe["index"] = range(dataframe.shape[0])
    dataframe["article"] = uuid4().hex
    return dataframe


# 生成随机振幅
def set_amplitude(dataframe):
    # 最大步幅
    max_step = random.randint(90, 365)
    # 最大振幅
    max_amplitude = random.uniform(0.1, 1)
    # 偏移
    offset = random.uniform(-1, 1)
    phase = random.randint(-1000, 1000)
    amplitude = (
        dataframe["index"]
        .apply(lambda x: max_amplitude * (x % max_step + phase) / max_step + offset)
        .values
    )

    # 增加随机性，每次生成，都有50%的概率正序或倒序排列   
    if random.random() < 0.5:
        amplitude = amplitude[::-1]
    dataframe["amplitude"] = amplitude
    return dataframe


# 生成随机偏移
def set_offset(dataframe):
    # 最大步幅
    max_step = random.randint(15, 45)
    # 最大偏移
    max_offset = random.uniform(-1, 1)
    # 基础偏移
    base_offset = random.uniform(-1, 1)
    phase = random.randint(-1000, 1000)
    offset = (
        dataframe["index"]
        .apply(
            lambda x: max_offset * np.cos(x * 2 * np.pi / max_step + phase)
            + base_offset
        )
        .values
    )

    if random.random() < 0.5:
        offset = offset[::-1]

    dataframe["offset"] = offset

    return dataframe


# 使用余弦函数，生成具有噪声的时序数据
def generate_time_series(dataframe):
    # 周期
    periods = [7, 14, 28, 30]
    period = random.choice(periods)
    # 初相位
    phase = random.randint(-1000, 1000)
    clip_val = random.uniform(0.3, 1)
    # 包含生成的随机振幅的随机偏移
    # 在整个函数上加上一系列常数，使得每次生成的数据有一定的差别，该系列常数分布满足是从0到最大振幅之间生成的正态分布
    dataframe["value"] = dataframe.apply(
                                        lambda x: np.clip(
                                            np.cos(x["index"] * 2 * np.pi / period + phase), -clip_val, clip_val
                                        ) * x["amplitude"] + x["offset"],
                                        axis=1,
    ) + np.random.normal(
        0, dataframe["amplitude"].abs().max() / 10, size=(dataframe.shape[0],)
    )

    return dataframe


def generate_df():
    dataframe = get_init_df()
    dataframe = set_amplitude(dataframe)
    dataframe = set_offset(dataframe)
    dataframe = generate_time_series(dataframe)
    return dataframe




# 使用时间序列季节性分解，查看分解结果
def season_view():   
    df = generate_df()
    df = df.set_index(['timestamp'])
    result_mul = seasonal_decompose(df['views'], 
                model='additive', 
                extrapolate_trend='freq')
    plt.rcParams.update({'figure.figsize': (10, 10)})
    result_mul.plot().suptitle('Additive Decompose')
    plt.show()


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    dataframes = []

    # 多次生成，并合并数据
    # 显示进度条
    # for _ in tqdm(range(20000)):
    #     df = generate_df()
    #     # fig = plt.figure()
    #     # plt.plot(df[-120:]["index"], df[-120:]["value"])
    #     # plt.show()
    #     dataframes.append(df)
    df = generate_df()
    dataframes.append(df)
    all_data = pd.concat(dataframes, ignore_index=True)
    all_data.to_csv("/home/yinke/TimeSeries/Experiment/data/generate_time_data.csv", index=False)
