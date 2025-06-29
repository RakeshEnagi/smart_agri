def generate_data(n=100):
    import pandas as pd
    import numpy as np
    ozone = np.random.uniform(30, 100, n)
    temp = np.random.uniform(25, 35, n)
    rain = np.random.uniform(200, 800, n)
    soil = np.random.uniform(0.1, 0.4, n)
    yield_ = 6 - 0.02 * ozone + 0.01 * rain - 0.05 * (temp - 30) + 4 * soil + np.random.normal(0, 0.3, n)
    return pd.DataFrame({
        "ozone": ozone,
        "temp": temp,
        "rain": rain,
        "soil": soil,
        "yield": yield_
    })
