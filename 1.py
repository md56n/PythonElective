#import library
import sys

#get input from user
sample = input("Strings: ")
list= []

#if the character in the string is already listed then skip, else add it to the list
for x in sample:
    if x in list:
        continue
    else:
        list.append(x)

#print the list
for i in list:
    sys.stdout.write(i)


