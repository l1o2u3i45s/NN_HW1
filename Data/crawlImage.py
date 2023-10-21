import requests
from bs4 import BeautifulSoup
import os
import json



def GetPokemonImage():
    pokemonPath = 'pokemon'
    if not os.path.isdir(pokemonPath):
        os.mkdir(pokemonPath)

    for i in range(1,97):
        pokemonUrl = "https://tw.portal-pokemon.com/play/pokedex/" + str(i).zfill(4)

        print(pokemonUrl)
        response = requests.get(pokemonUrl)
        soup = BeautifulSoup(response.text, "html.parser")
        imageElement = soup.find_all("img", {"class": "pokemon-img__front"})
        imgUrl = "https://tw.portal-pokemon.com" + imageElement[0]['src']

        imgData = requests.get(imgUrl).content

        if i < 80:
            with open(pokemonPath + '/'+ str(i).zfill(4) + '.jpg', 'wb') as handler:
                handler.write(imgData)
        else:
            with open('testData/pokemon_'+ str(i).zfill(4) + '.jpg', 'wb') as handler:
                handler.write(imgData)

        print(imgUrl)

def GetDigimonImage():
    digimonPath = 'digimon'
    if not os.path.isdir(digimonPath):
        os.mkdir(digimonPath)

    digimonUrl = 'https://digimon.net/reference_zh-CHT/request.php?digimon_name=&digimon_level=&attribute=&type=&next=0'
 
    digimonHeaders = {
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
    "Cookie": "_gcl_au=1.1.1404146890.1697427455; _ga=GA1.1.1925168787.1697427455; OptanonConsent=isGpcEnabled=0&datestamp=Sat+Oct+21+2023+11%3A05%3A06+GMT%2B0800+(%E5%8F%B0%E5%8C%97%E6%A8%99%E6%BA%96%E6%99%82%E9%96%93)&version=202307.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&genVendors=&consentId=31a40f26-03ae-4a66-9081-f17d165d4643&interactionCount=1&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0004%3A0&AwaitingReconsent=false; digimongRUHdad6=cgs07drimk1afbj6g1ban5t5j1; _ga_7SHTJYC01X=GS1.1.1697865800.3.0.1697865800.60.0.0; _ga_HB6GQBXK33=GS1.1.1697865800.3.0.1697865800.0.0.0",
    "DNT": "1",
    "Host": "digimon.net",
    "Referer": "https://digimon.net/reference_zh-CHT/index.php?next=96",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "sec-ch-ua": '"Chromium";v="118", "Google Chrome";v="118", "Not=A?Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "Windows"
    }

    index = 1
    response = requests.get(digimonUrl, headers=digimonHeaders)  
    digimonDTOList = json.loads(response.text)
    for digimon in digimonDTOList['rows']:
        digiminImgPath = 'https://digimon.net/cimages/digimon/' + digimon['directory_name'] + '.jpg'
        print(digimon) 

        imgData = requests.get(digiminImgPath).content

        if index < 80:
            with open(digimonPath + '/'+ str(index).zfill(4) + '.jpg', 'wb') as handler:
                handler.write(imgData)
        else:
            with open('testData/digimon_'+ str(index).zfill(4) + '.jpg', 'wb') as handler:
                handler.write(imgData)

        index+=1
     
if __name__ == '__main__':
    testDataPath = 'testData'
    if not os.path.isdir(testDataPath):
        os.mkdir(testDataPath)
    GetPokemonImage()
    GetDigimonImage()
