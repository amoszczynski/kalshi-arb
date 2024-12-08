import pandas as pd
import requests
import datetime

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

def load_private_key_from_file(file_path):
    with open(file_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,  # or provide a password if your key is encrypted
            backend=default_backend()
        )
    return private_key


import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.exceptions import InvalidSignature

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    # Before signing, we need to hash our message.
    # The hash is what we actually sign.
    # Convert the text to bytes
    message = text.encode('utf-8')

    try:
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode('utf-8')
    except InvalidSignature as e:
        raise ValueError("RSA sign PSS failed") from e
    

def auth_headers(path):
    # Get the current time
    current_time = datetime.datetime.now()
    
    # Convert the time to a timestamp (seconds since the epoch)
    timestamp = current_time.timestamp()
    
    # Convert the timestamp to milliseconds
    current_time_milliseconds = int(timestamp * 1000)
    timestampt_str = str(current_time_milliseconds)
    
    # Load the RSA private key
    private_key = load_private_key_from_file('kalshiprivatekey.key')
    
    method = "GET"
    
    msg_string = timestampt_str + method + path
    sig = sign_pss_text(private_key, msg_string)

    return (sig, timestampt_str)


def GetMarketsFromEvent(*, event_ticker: str):
    base_url = "https://api.elections.kalshi.com"
    path = "/trade-api/v2/events/" + event_ticker

    sig, timestampt_str = auth_headers(path)

    headers = {
        "accept": "application/json",
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": timestampt_str
    }

    # Send the request
    response = requests.get(base_url + path, headers=headers)

    if response.status_code == 200:
        data = response.json()
        tickers = []
        for dt in data["markets"]:
            tickers.append(dt["ticker"])
        return tickers
    else:
        print(f"Error: {response.text}")
        return []



def GetMarketCandlesticks(*, market_ticker: str, series_ticker: str, start_ts: int, end_ts: int, period_interval: int):
    base_url = "https://api.elections.kalshi.com"
    path = "/trade-api/v2/series/" + series_ticker + "/markets/" + market_ticker + "/candlesticks"

    sig, timestampt_str = auth_headers(path)

    headers = {
        "accept": "application/json",
        "KALSHI-ACCESS-KEY": API_KEY,
        "KALSHI-ACCESS-SIGNATURE": sig,
        "KALSHI-ACCESS-TIMESTAMP": timestampt_str
    }

    params = {
        "start_ts": start_ts,
        "end_ts": end_ts,
        "period_interval": period_interval
    }

    # Send the request
    response = requests.get(base_url + path, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        return data

    else:
        print(f"Error: {response.text}")
        return []


API_KEY = ""  #TODO: Add your API key here
SERIES_TICKER = "KXINXY"
EVENT_TICKERS = {2022 : "INXY-22DEC30", 2023 : "INXY-23DEC29", 2024 : "INXY-24DEC31"}
period_interval = 1440

master_data = {}

for year, EVENT_TICKER in EVENT_TICKERS.items():

    market_tickers = GetMarketsFromEvent(event_ticker=EVENT_TICKER)[1:-1]  # Remove lower and upper bound markets

    start_date = datetime.datetime(year, 1, 1, 16, 0, 0)
    end_date = datetime.datetime(year, 12, 31, 16, 0, 0)

    start_ts = int(start_date.timestamp())
    end_ts = int(end_date.timestamp())

    temp = pd.DataFrame(columns=market_tickers, index=pd.date_range(start=start_date.date(), end=end_date.date(), freq="D"))

    for MARKET_TICKER in market_tickers:
        # Only get data for same year market
        data = GetMarketCandlesticks(market_ticker=MARKET_TICKER, series_ticker=SERIES_TICKER, start_ts=start_ts, end_ts=end_ts, period_interval=period_interval)
        data = pd.DataFrame(data["candlesticks"])
        if data.empty: continue
        data.index = pd.to_datetime(data["end_period_ts"], unit="s")
        data.index = data.index.normalize()
    
        prices = data[["yes_bid", "yes_ask", "price"]].apply(lambda x : x.apply(lambda y : y["close"]))
        temp[MARKET_TICKER] = prices["price"]

    # Column names are of this format "INXD-24DEC31-B3300" I want to extract the price from this and change the column name to be a string that represents the range (price-100, price+99.99)
    temp.columns = [str(float(col.split("-")[2][1:]) - 100) + "-" + str(float(col.split("-")[2][1:]) + 99.99) for col in temp.columns.tolist()]
    master_data[year] = temp

print(master_data)
