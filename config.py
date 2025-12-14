from qlib.contrib.data.loader import Alpha158DL

market = "csi300"
start_time = "2020-01-01"
end_time = "2025-01-01"

# factor_fields, factor_names = Alpha158DL.get_feature_config()
factor_fields = ["Ref($close, 60)/$close"]
factor_names = ["ROC60"]
periods = {"1d": 1, "1w": 5, "1m": 20}
ret_map = {f"ret_{label}": f"Ref($close, -{lag})/$close - 1" for label, lag in periods.items()}