import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
from datetime import date, datetime


OPERATORS= {
    "Turkcell": "turkcell",
    "Türk Telekom": "turk-telekom",
    "Vodafone": "vodafone"
}

DATE_START= date(2025, 11, 1)
DATE_END= date(2026, 1, 31)

OUTPUT_DIR= "output2"

TEST_MODE= False
TEST_LIMIT= 10

#---------------------------
DELAY_LIST= (1.5, 3.0)
DELAY_DETAIL= (1.0, 2.0)



