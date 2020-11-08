from typing import Iterator
import numpy as np
from agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import time


class Person:
    def __init__(self, home,sfood,smoney):
        self.home=home
        self.pos=home.copy()
        self.food=sfood
        self.money=smoney
        self.inf=False
        self.gottenFood=False
        self.gottenWork=False
    def getPos(self):
        return self.pos.copy()
    def getHome(self):
        return self.home.copy()
    def setPos(self,pos):
        self.pos=pos.copy()
    def addFood(self,amount,price):
        self.food+=amount
        self.money-=price
    def addMoney(self, amount):
        self.money+=amount
    def hunger(self,rate):
        self.food-=rate
    def infect(self):
        self.inf=True
class World:
    directions=[[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
    up = 0
    down = 1
    left = 2
    right = 3
    stay = 4
    def __init__(self, size, workList,foodList,homeList,startFood,startMoney,inf_,death_,salary_,buyFood_,price_,hunger_):
        self.size=size
        self.lboard=np.zeros((size,size))
        self.pboardInit=np.zeros((size,size))
        self.pLoc=[]
        self.homeListInit=[x.copy() for x in homeList]
        self.pList={}
        self.infected=[]
        self.startM=startMoney
        self.startF=startFood
        self.inf=inf_
        self.death=death_
        self.salary=salary_
        self.food=buyFood_
        self.price=price_
        self.hunger=hunger_
        for i in range(size):
            temp=[]
            for _ in range(size):
                temp.append([])
            self.pLoc.append(temp)
        for i in workList:
            self.lboard[i[0]][i[1]]=1
        for i in foodList:
            self.lboard[i[0]][i[1]]=2
        for i in range(len(homeList)):
            self.lboard[homeList[i][0]][homeList[i][1]]=3
            self.pboardInit[homeList[i][0]][homeList[i][1]]=1
            self.pList[i]=Person(homeList[i],self.startF,self.startM)
            self.pLoc[homeList[i][0]][homeList[i][1]].append(i)
        self.pboard=self.pboardInit.copy()
        self.keys=[]
        self.pos=0
    def display(self):
        np.set_printoptions(threshold=10000)
        print("loc board")
        for row in self.lboard:
            print(row)
        print("pNum board")
        for row in self.pboard:
            print(row)
        print("people board")
        for row in self.pLoc:
            for col in row:
                print(col,end='')
            print()
        print("infected: \n", self.infected)
    def getSingleState(self,index):
        pos=self.pList[index].pos
        #print("pos: ",pos)
        stateLoc=np.zeros((3,3))
        statePeep=np.zeros((3,3))
        stateRes=np.zeros((3,3))
        for d in self.directions:
            tempx=pos[0]+d[0]
            tempy=pos[1]+d[1]
            if (tempx>-1 and tempx<self.size) and (tempy>-1 and tempy<self.size):
                stateLoc[d[0]+1][d[1]+1]=self.lboard[tempx][tempy]
                #if self.lboard[tempx][tempy]!=3:
                statePeep[d[0]+1][d[1]+1]=self.pboard[tempx][tempy]
            else:
                stateLoc[d[0]+1][d[1]+1]=-1
                statePeep[d[0]+1][d[1]+1]=-1
        stateLoc[1][1]=self.lboard[pos[0]][pos[1]]
        statePeep[1][1]= 0 if self.pboard[pos[0]][pos[1]]<2 else self.pboard[pos[0]][pos[1]]-1
        stateRes[0][0]=self.pList[index].food
        stateRes[0][1]=self.pList[index].money
        return (stateLoc,statePeep,stateRes)
    def __iter__(self):
        self.keys=[*self.pList]
        self.pos=0
        return self
    def __next__(self):
        if(self.pos<len(self.keys)):
            x=self.getSingleState(self.keys[self.pos])
            self.pos+=1
            return x
        else:
            raise StopIteration
    def newDay(self):
        for x in self.pList:
                current=self.pList[x].getPos()
                home=self.pList[x].getHome()
                self.pLoc[current[0]][current[1]].remove(x)
                self.pLoc[home[0]][home[1]].append(x)
                self.pList[x].setPos(home)
                self.pList[x].hunger(self.hunger)
                self.pList[x].gottenFood=False
                self.pList[x].gottenWork=False
        newInf=[]
        for x in self.infected:
            if self.pList[x].inf and np.random.random_sample()<self.death:
                current=self.pList[x].getPos()
                self.pLoc[current[0]][current[1]].remove(x)
                del self.pList[x]
            else:
                newInf.append(x)
        self.infected=newInf
        self.pboard=[]
        for row in self.pLoc:
            temp=[]
            for col in row:
                temp.append(len(col))
            self.pboard.append(temp)
    def reset(self):
        self.pList=[x.copy() for x in self.homeListInit]
        self.homeList=[x.copy() for x in self.homeListInit]
        self.foodList=[]
        self.moneyList=[]
        self.infected=[]
        for _ in self.homeList:
            self.foodList.append(self.startF)
            self.moneyList.append(self.startM)
            self.infected.append(False)
        self.pboard=self.pboardInit.copy()
    #move, infect update loop
    def update(self,actions):
        temp=[*self.pList]
        for i in range(len(temp)):
            k=temp[i]
            current=self.pList[k].getPos()
            if actions[i]==self.up:
                self.posUpdate(current,k,-1,0)
            if actions[i]==self.down:
                self.posUpdate(current,k,1,0)
            if actions[i]==self.left:
                self.posUpdate(current,k,0,-1)
            if actions[i]==self.right:
                self.posUpdate(current,k,0,+1)
            newPos=self.pList[k].getPos()
            if not self.pList[k].gottenWork and self.lboard[newPos[0]][newPos[1]]==1:
                self.pList[k].addMoney(self.salary)
                self.pList[k].gottenWork=True
            elif not self.pList[k].gottenFood and self.lboard[newPos[0]][newPos[1]]==2:
                self.pList[k].addFood(self.food,self.price)
                self.pList[k].gottenFood=True
        newInf=[]
        for i in self.infected:
            current=self.pList[i].getPos()
            for key in self.pLoc[current[0]][current[1]]:
                if not self.pList[key].inf and (np.random.random_sample()<self.inf):
                    self.pList[key].infect()
                    newInf.append(key)
        self.infected=self.infected+newInf
        return [1 for _ in range(len(temp))]


    def posUpdate(self,current,k,v,h):
        if current[0] + v >= self.size or current[0] + v < 0 or current[1] + h >= self.size or current[1] + h < 0:
            return
        self.pboard[current[0]][current[1]]-=1
        self.pboard[current[0]+v][current[1]+h]+=1
        self.pList[k].setPos([current[0]+v,current[1]+h])
        self.pLoc[current[0]][current[1]].remove(k)
        self.pLoc[current[0]+v][current[1]+h].append(k)
    def infection(self, num):
        keys=[*self.pList]
        while len(self.infected)<num:
            key=np.random.choice(keys)
            if not self.pList[key].inf:
                self.pList[key].infect()
                self.infected.append(key)
def main():
    size=5
    startF=10
    startM=10
    deathRate=0.01
    infRate=1
    salary=10
    amount=5
    price=5
    food=[[1,1]]
    home=[[3,3]]
    #,[0,0],[3,2],[1,1],[5,5],[7,7],[9,9]]
    work=[[4,4]]
    hunger=2
    world=World(size,work,food,home,startF,startM,infRate,deathRate,salary,amount,price, hunger)
    world.infection(1)
    #world.update(['r','l','s'])
    #world.update(['d','s','s'])
    #world.display()
    days = 100
    iterations_per_day = 4 * world.size
    agent = Agent()
    iterations = 1
    days = 2
    reward_iter = [[] * len(home)]
    best_avg_reward = -np.inf
    for iter in range(iterations):
        for d in range(days):
            for i in range(iterations_per_day):
                #world.display()
                actions = []
                states = []
                for x in world:
                    test = np.array([x[0], x[1], x[2]])
                    states.append(test)
                    actions.append(agent.act(test))
                rewards = world.update(actions)
                next_states = []
                for count, x in enumerate(world):
                    reward_iter[count].append(rewards[count])
                    test = np.array([x[0], x[1], x[2]])
                    next_states.append(test)
                for j in range(len(states)):
                    agent.remember(states[j], actions[j], rewards[j], next_states[j], 0)
                if len(agent.memory) > agent.batch:
                    agent.learn()
                
            world.newDay()
        world.reset()
        world.display()

    for i in range(len(home)):
        plt.plot(reward_iter[i], label="Person Number {}".format(i))
    plt.legend()
    plt.ylabel('Reward')
    plt.xlabel('Actions Taken')
    plt.show()
        

        
if __name__ == "__main__":
    main()
