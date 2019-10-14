class Library:      #library class
    student = ""    #student variable
    book = ""       #book variable
    faculty = ""    #faculty variable
    department = "" #department variable

    def __init__(self, student, book, faculty, department): #Define variables
        self.student = student
        self.book = book
        self.faculty = faculty
        self.department = department

l=Library("Chris", "Help", "Jennings", "Science")   #library object
print(l.student)        #print student
print(l.book)           #print book
print(l.faculty)        #print faculty
print(l.department)     #print deprtment