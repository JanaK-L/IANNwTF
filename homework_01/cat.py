class Cat:

    def __init__(self, name:str):
        self.name = name
        self.whoAmI = "Hello! I am " + self.name + "!"
    
    def intro(self):
        print(self.whoAmI)
    
    def greet(self, nameToGreet:str):
        print("I see you are also a cool fluffy kitty " + nameToGreet + ", let’s together purr at the human, so that they shall give us food.")
    
    def getName(self):
        return self.name