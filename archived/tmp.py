import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import pytz


x = np.linspace(0, np.pi * 8, num=1000)
y = np.sin(x) + np.random.randint(0, 100)
dataset = pd.DataFrame(data={'x': x, 'xlabel': f"Hello you", 'target': y, 'prediction': y+(np.random.random()*1.5)})
now = pytz.utc.localize(datetime.now())
dataset.index = dataset.index.map(lambda i: now+timedelta(minutes=i))

dataset.to_csv('data/firehose.csv')
