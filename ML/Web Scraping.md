# Introduction to Web Scraping with Python
```python
#The Web Scraping Pipeline
'''
"
1.Downloading: Downloading the HTML web-page
2.Parsing: Parsing the HTML and retrieving data we're interested in
3.Storing: Storing the retrieved data in our local machine in a specific format
"
'''

#Downloading HTML
import requests
result = requests.get('http://quotes.toscrape.com/')
page = result.text

#Parsing HTML and Extracting Data
"parsing is the process of analyzing a string so that we can understand its contents and thus access data within it easily."

from bs4 import BeautifulSoup
soup = BeautifulSoup(page, 'html.parser') #create a parsed version of the page by passing it to the BeautifulSoup
#html.parser is the parser that Beautiful Soup is using to parse the string

#extract all the div tags in the page containing class="quote"
quotes = soup.find_all('div', class_='quote')

scraped = []
for quote in quotes:
    text = quote.find('span', class_='text').text
    author = quote.find('small', class_='author').text
    scraped.append([text, author])

#Storing the Retrieved Data
import csv

with open('quotes.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',') # create a CSV writer using the opened quotes.csv file
    for quote in scraped:
        writer.writerow(quote) #writing the quotes one at a time with the writerow function
        #the parameter that writerow accepts is a list and then it writes that to the CSV as a row.
```
# Resources
https://stackabuse.com/introduction-to-web-scraping-with-python/



