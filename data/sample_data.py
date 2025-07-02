def generate_data(n=100):
    import pandas as pd
    import numpy as np
    ozone = np.random.uniform(30, 100, n)
    temp = np.random.uniform(15, 30, n)  # Potato prefers 15–25°C
    rain = np.random.uniform(200, 600, n)
    soil = np.random.uniform(0.2, 0.4, n)
    yield_ = 30 - 0.05 * ozone + 0.02 * rain - 0.4 * abs(temp - 20) + 25 * soil + np.random.normal(0, 2, n)
    return pd.DataFrame({
        "ozone": ozone,
        "temp": temp,
        "rain": rain,
        "soil": soil,
        "yield": yield_
    })
