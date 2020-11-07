class Agent:
    def __init__(self,foodStart,moneyStart,homePos):
        self.food=foodStart
        self.money=moneyStart
        self.home=homePos
    def chooseAction(self,state):
        pass
    def nextPos(self, action):
        pass
    def getFood(self,price,food_):
        self.money-=price
        self.food+=food_
    def work(self, work_):
        self.money+=work_
    def display(self):
        pass
    def __str__(self):
        
