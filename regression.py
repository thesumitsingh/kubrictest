import requests
import pandas
import scipy
from scipy import stats
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION GOES HERE
    testdata=requests.get(TEST_DATA_URL)
    response = requests.get(TRAIN_DATA_URL).content
    df=pd.read_csv(io.StringIO(response.decode('utf-8')), header=None)
    x=df.iloc[0,1:].values.astype('float32')
    y=df.iloc[1,1:].values.astype('float32')
    r=stats.linregress(x,y)
    ans=[]
    for x in areas:
        ans.append(x*r.slope+r.intercept)
    return ans


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
