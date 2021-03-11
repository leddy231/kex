from datetime import datetime
from datetime import timedelta

i = 155
print(datetime(2020, 1, 1).timetuple().tm_yday)
print(datetime(2020, 1, 1) + timedelta(i - 1))