
import sys

sample = input("Strings: ")
list= []


for x in sample:
    if x in list:
        continue
    else:
        list.append(x)


for i in list:
    sys.stdout.write(i)


