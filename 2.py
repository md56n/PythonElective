names = []
keys = []
myDict = {}

list = [( 'John', ('Physics', 80)) , ( 'Daniel', ('Science', 90)), ('John', ('Science', 95)), ('Mark',('Maths', 100)),
        ('Daniel', ('History', 75)),('Mark', ('Social', 95))]

for (key, value) in list:
    if key in myDict:
        myDict[key].append(value)
    else:
        myDict[key] = [value]

print(myDict)
