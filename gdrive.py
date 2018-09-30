# import httplib2
# import os

# from datetime import tzinfo, timedelta, date
# from dateutil.relativedelta import relativedelta

import cv2
import numpy as np
# import ConfigParser

# # Gdrive
# from apiclient.discovery import build
# from oauth2client.client import OAuth2WebServerFlow
# from oauth2client.file import Storage

# from models import getLastRecognizedImage, mylogger

import io
import requests
from PIL import Image


# class TZ(tzinfo):
#     def utcoffset(self, dt): return timedelta(hours=+4)


def getImagesFromS3():
    
    response = requests.get('https://i-met.s3.amazonaws.com/alex@gmail.com/gas/3663434534/image.bmp')
    image = Image.open(io.BytesIO(response.content))
    image.save("image.bmp")
    image = np.asarray(image)
      
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

# def downloadImageFromGDrive (downloadUrl, http=None, file_id=None):
#     return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# def createImageFromGDriveObject (img_info, http=None):
# return downloadImageFromGDrive(img_info['downloadUrl'], http)