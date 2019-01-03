from	bs4	import	BeautifulSoup
import	requests

#download	one	of	those	pages	and	feed	it	to Beautiful	Soup
url	=	"https://apod.nasa.gov/htmltest/gifcity/e.1mil"
soup	=	BeautifulSoup(requests.get(url).text,	'html5lib')
digits = soup.prettify()
len(digits)
print(digits)

strdigits=str(digits)
print(strdigits)
import re
newstr=strdigits.replace("</head>", "").replace("<html>", "").replace(" <head>","").replace('</body>','').replace('</html>','').replace('<body>','')
newstr
r=''.join(c for c in newstr if c.isdigit())
len(r[14:])
print(r[13:])
newstr=r[13:]
len(newstr)
print(newstr)

dig=newstr.strip()
print(dig)
len(dig)
#counting each number occurences in a list

list=[]
list.append(dig.count('0'))
list.append(dig.count('1'))
list.append(dig.count('2'))
list.append(dig.count('3'))
list.append(dig.count('4'))
list.append(dig.count('5'))
list.append(dig.count('6'))
list.append(dig.count('7'))
list.append(dig.count('8'))
list.append(dig.count('9'))

x=[0,1,2,3,4,5,6,7,8,9]
y=list
y
x
#histogram
import matplotlib.pyplot as plt
import numpy as np

index = np.arange(len(x))

def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(x))
    plt.bar(index, y)
    plt.xlabel('Digits', fontsize=8)
    plt.ylabel('Counts', fontsize=8)
    plt.xticks(index, x, fontsize=8, rotation=30)
    plt.title('Counts for each digit in e')
    plt.show()
