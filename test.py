from random import uniform
from typing import Iterator
import numpy as np
from agent import Agent
from collections import deque
import matplotlib.pyplot as plt
import time
import random


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
        self.food = max(0, self.food - rate)
    def infect(self):
        self.inf=True
class World:
    directions=[[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
    up = 0
    down = 1
    left = 2
    right = 3
    stay = 4
    def __init__(self, size, workList,foodList,homeList,startFood,startMoney,inf_,death_,salary_,buyFood_,price_,hunger_, rewardFood_, rewardLive_,rewardDeath_,rewardInfect_,rewardWork_,rewardCrowd_):
        self.size=size
        self.lboard=np.zeros((size,size))
        self.pboardInit=np.zeros((size,size))
        self.pLoc=[]
        self.homeListInit=[x.copy() for x in homeList]
        self.pList={}
        self.infected=[]
        self.infectedInit=[]
        self.startM=startMoney
        self.startF=startFood
        self.inf=inf_
        self.death=death_
        self.salary=salary_
        self.food=buyFood_
        self.price=price_
        self.hunger=hunger_
        self.rewardFood=rewardFood_
        self.rewardLive=rewardLive_
        self.rewardDeath=rewardDeath_
        self.rewardInfect=rewardInfect_
        self.rewardWork=rewardWork_
        self.rewardCrowd=rewardCrowd_
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
        kill=[]
        rewards=[]
        for x in self.pList:
                current=self.pList[x].getPos()
                home=self.pList[x].getHome()
                self.pLoc[current[0]][current[1]].remove(x)
                self.pLoc[home[0]][home[1]].append(x)
                self.pList[x].setPos(home)
                self.pList[x].hunger(self.hunger)
                self.pList[x].gottenFood=False
                self.pList[x].gottenWork=False
                if self.pList[x].inf and np.random.random_sample()<self.death:
                    kill.append(x)
                    rewards.append(self.rewardDeath)
                else:
                    rewards.append(self.rewardLive)
        for x in kill:
            current=self.pList[x].getPos()
            self.pLoc[current[0]][current[1]].remove(x)
            del self.pList[x]
            self.infected.remove(x)
        self.pboard=[]
        for row in self.pLoc:
            temp=[]
            for col in row:
                temp.append(len(col))
            self.pboard.append(temp)
        return rewards

    def reset(self):
        self.pLoc=[]
        for i in range(self.size):
            temp=[]
            for _ in range(self.size):
                temp.append([])
            self.pLoc.append(temp)
        self.pList={}
        for i in range(len(self.homeListInit)):
            self.pList[i]=Person(self.homeListInit[i],self.startF,self.startM)
            self.pLoc[self.homeListInit[i][0]][self.homeListInit[i][1]].append(i)
        self.pboard=self.pboardInit.copy()
        self.infected=self.infectedInit.copy()
        for x in self.infected:
            self.pList[x].inf=True
    #move, infect update loop
    def update(self,actions):
        temp=[*self.pList]
        rewards=[]
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
                rewards.append(self.rewardWork)
            elif not self.pList[k].gottenFood and self.lboard[newPos[0]][newPos[1]]==2:
                if(self.pList[k].money>=self.price):
                    self.pList[k].addFood(self.food,self.price)
                    self.pList[k].gottenFood=True
                    rewards.append(self.rewardFood)
                else:
                    rewards.append(-1)
            else:
                if(self.pList[k].food==0):
                    rewards.append(-2)
                else:
                    rewards.append(0)
        newInf=[]
        for i in self.infected:
            current=self.pList[i].getPos()
            for key in self.pLoc[current[0]][current[1]]:
                if not self.pList[key].inf and (np.random.random_sample()<self.inf):
                    self.pList[key].infect()
                    newInf.append(key)
                    rewards[temp.index(k)]=self.rewardInfect
        self.infected=self.infected+newInf
        for i in range(self.size):
            for j in range(self.size):
                for k in self.pLoc[i][j]:
                    if len(self.pLoc[i][j])>1 and rewards[temp.index(k)]>-10:
                        rewards[temp.index(k)]=self.rewardCrowd
        return rewards
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
        self.infectedInit=self.infected.copy()


def main():
    size = 75
    startF=10
    startM=10
    deathRate=0.05
    infRate=0.5
    salary=10
    amount=5
    price=5
    N = 10
    food=[[random.randint(0, size-1), random.randint(0, size-1)] for i in range(N)]
    home = []
    work = []
    for i in range(N):
        x = [random.randint(0, size-1), random.randint(0, size-1)]
        if x not in food:
            work.append(x)
        else:
            i -= 1
    for i in range(15):
        x = [random.randint(0, size-1), random.randint(0, size-1)]
        if x not in food and x not in work:
            home.append(x)
        else:
            i -= 1
    hunger=2
    rewardFood,rewardLive,rewardDeath,rewardInfect,rewardWork,rewardCrowd= 5, 1,-10,-1,10,-5
    world=World(size,work,food,home,startF,startM,infRate,deathRate,salary,amount,price, hunger, rewardFood, rewardLive,rewardDeath,rewardInfect,rewardWork,rewardCrowd)
    world.infection(2)
    days = 30
    iterations_per_day = int(1.2 * world.size)
    agent = Agent()
    iterations = 2
    reward_iter = [[] for _ in range(len(home))]
    dead = [i for i in range(len(home))]
    day_rewards = [[] for _ in range(len(home))]
    avg_iter = []
    avg_day = []
    for iter in range(iterations):
        for d in range(days):
            print(iter, d, len(world.infected))
            start = time.time()
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
                temp_reward = [0 for i in range(len(rewards))]
                reset = False
                for count, x in enumerate(world):
                    #print(rewards, count)
                    temp_reward.append(rewards[count])
                    avg_iter.append(sum(temp_reward)/len(temp_reward))
                    test = np.array([x[0], x[1], x[2]])
                    next_states.append(test)
                counter = 0
                if len(temp_reward) > 0:
                    for j in range(len(dead)):
                        if dead[j] == -1:
                            reward_iter[j].append(0)
                            counter += 1
                        else:
                            reward_iter[j].append(temp_reward[j-counter])
                else: 
                    reset = True
                for j in range(len(states)):
                    agent.remember(states[j], actions[j], rewards[j], next_states[j], 0)
                if len(agent.memory) > agent.batch:
                    agent.learn()
                if reset:
                    break
            actions = []
            states = []
            for x in world:
                test = np.array([x[0], x[1], x[2]])
                states.append(test)
                actions.append(agent.act(test))
            rewards = world.newDay()
            next_states = []
            temp_reward = [0 for i in range(len(rewards))]
            for count, x in enumerate(world):
                temp_reward.append(rewards[count])
                if rewards[count] < 0:
                    dead[count] = -1
                avg_iter.append(sum(temp_reward)/len(temp_reward))
                test = np.array([x[0], x[1], x[2]])
                next_states.append(test)
            counter = 0
            reset = False
            if len(temp_reward) > 0:
                for j in range(len(dead)):
                    if dead[j] == -1:
                        reward_iter[j].append(0)
                        counter += 1
                    else:
                        reward_iter[j].append(temp_reward[j-counter])
            else:
                reset = True
            for j in range(len(states)):
                day_rewards[j].append(sum(reward_iter[j]))
                avg_day.append(sum(day_rewards[j])/len(day_rewards[j]))
                next_1 = 0
                if j < len(next_states):
                    next_1 = next_states[j]
                else:
                    next_1 = np.zeros(shape=(3,3,3))
                agent.remember(states[j], actions[j], rewards[j], next_1, 1)
            if len(agent.memory) > agent.batch:
                agent.learn()
            if reset:
                world.reset()
                break
            #print(time.time() - start)
        world.reset()
        #world.display()
        #agent.q_network.save("model.h5")

    #np.save("all_reward", np.asarray(reward_iter))
    #np.save("avg_reward", np.asarray(avg_iter))
    #np.save("day_reward", np.asarray(day_rewards))
    #np.save("avg_day_reward", np.asarray(avg_day))

    for i in range(len(home)):
        plt.plot(reward_iter[i], label="Person Number {}".format(i))
    #plt.legend()
    plt.ylabel('Iteration Reward')
    plt.xlabel('Actions Taken')
    plt.show()
    
    plt.plot(avg_iter, label="Average Rewards")
    plt.legend()
    plt.ylabel('Iteration Reward')
    plt.xlabel('Actions Taken')
    plt.show()

    for i in range(len(home)):
        plt.plot(day_rewards[i], label="Person Number {}".format(i))
    #plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('Days')
    plt.show()
        
    plt.plot(avg_day, label="Average Day Rewards")
    plt.legend()
    plt.ylabel("Rewards")
    plt.xlabel("Days")
    plt.show()

        
if __name__ == "__main__":
    main()
