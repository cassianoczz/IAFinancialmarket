#https://www.youtube.com/watch?v=JrXkRJlVYiU
import requests, bs4

res = requests.get("https://www.tradingview.com/chart/?symbol=BITFINEX%3ABTCUSD")

try:
    res.raise_for_status()
    objectSoup = bs4.BeautifulSoup(res.text, features="lxml")
    listPrice = objectSoup.select(#valueItem-3JDGGSt_-)

    print(listPrice)
except Exception as exc:
    print("Nao deu nao agora %s" % (exc))