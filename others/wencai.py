import pywencai
import pandas as pd


def query_stocks(condition: str) -> pd.DataFrame:
    """根据条件字符串查询选股结果并返回DataFrame。"""
    return pywencai.get(question=condition, query_type="stock", loop=True)


if __name__ == "__main__":
    sample_condition = "非ST，非退市，股东人数连续3年减少，股价在233日均线之上，股价在377日均线之上，股价在610日均线之上，股价在987日均线之上"
    result = query_stocks(sample_condition)
    print(result)
