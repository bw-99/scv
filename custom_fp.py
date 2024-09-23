import numpy as np
from fuxictr.preprocess import FeatureProcessor
import polars as pl
from datetime import datetime, date

class CustomizedFeatureProcessor(FeatureProcessor):
    def convert_to_bucket(self, col_name):
        def _convert_to_bucket(value):
            if value > 2:
                value = int(np.floor(np.log(value) ** 2))
            else:
                value = int(value)
            return value
        return pl.col(col_name).map_elements(_convert_to_bucket).cast(pl.Int32)
    
    def extract_country_code(self, col_name):
        return pl.col(col_name).map_elements(lambda isrc: isrc[0:2] if isrc else "")

    def bucketize_age(self, col_name):
        def _bucketize(age):
            try:
                age = float(age)
                if age < 1 or age > 95:
                    return ""
                elif age <= 10:
                    return "1"
                elif age <=20:
                    return "2"
                elif age <=30:
                    return "3"
                elif age <=40:
                    return "4"
                elif age <=50:
                    return "5"
                elif age <=60:
                    return "6"
                else:
                    return "7"
            except:
                return ""
        return pl.col(col_name).map_elements(_bucketize)

    def convert_weekday(self, col_name=None):
        def _convert_weekday(timestamp):
            dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
            return int(dt.strftime('%w'))
        return pl.col("hour").map_elements(_convert_weekday)

    def convert_weekend(self, col_name=None):
        def _convert_weekend(timestamp):
            dt = date(int('20' + timestamp[0:2]), int(timestamp[2:4]), int(timestamp[4:6]))
            return 1 if dt.strftime('%w') in ['6', '0'] else 0
        return pl.col("hour").map_elements(_convert_weekend)

    def convert_hour(self, col_name=None):
        return pl.col("hour").map_elements(lambda x: int(x[6:8]))


