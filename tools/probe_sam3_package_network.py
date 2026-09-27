"""Read-only network probe for the official PyTorch wheel, never disables TLS."""
import concurrent.futures
import time
import requests

URL='https://download.pytorch.org/whl/cu126/torch-2.7.1%2Bcu126-cp312-cp312-win_amd64.whl'

def probe(proxy):
    session=requests.Session()
    session.trust_env=False
    start=time.monotonic()
    try:
        with session.get(URL,headers={'Range':'bytes=0-2097151'},proxies=proxy,timeout=15,stream=True) as response:
            size=0
            for part in response.iter_content(262144):
                size+=len(part)
                if size>=2097152: break
            return dict(proxy=bool(proxy),status=response.status_code,range=response.headers.get('Content-Range'),
                        bytes=size,seconds=round(time.monotonic()-start,2))
    except requests.RequestException as error:
        return dict(proxy=bool(proxy),error=type(error).__name__)

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(2) as executor:
        for result in executor.map(probe,[{}, {'https':'http://127.0.0.1:7890'}]):
            print(result,flush=True)
