import numpy as np
global iRate
global price
global food
global work
class World:
    directions=[[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
    def __init__(self, size, foodList,homeList,workList):
        self.size=size
        self.lboardInit=np.zeros((size,size))
        self.pboardInit=np.zeros((size,size))
        self.plist=[x.copy() for x in homeList]
        for i in workList:
            self.lboardInit[i[0]][i[1]]=1
        for i in foodList:
            self.lboardInit[i[0]][i[1]]=2
        for i in homeList:
            self.lboardInit[i[0]][i[1]]=3
            self.pboardInit[i[0]][i[1]]=1
        self.lboard=self.lboardInit.copy()
        self.pboard=self.pboardInit.copy()
    def display(self):
        pass
    def getSingleState(self,pos):
        stateLoc=np.zeros((3,3))
        statePeep=np.zeros((3,3))
        for d in self.directions:
            tempx=pos[0]+d[0]
            tempy=pos[1]+pos[1]
            if (tempx>-1 and tempx<self.size) and (tempy>-1 and tempy<self.size):
                stateLoc[d[0]][d[1]]=self.lboard[tempx][tempy]
                statePeep[d[0]][d[1]]=self.pboard[tempx][tempy]
            else:
                stateLoc[d[0]][d[1]]=-1
                statePeep[d[0]][d[1]]=-1
        return stateLoc,statePeep
    def getState(self):

        for i in self.plist:
            sLoc,pLoc=self.getSingleState(i);
def main():
    iRate=0.01
    price=10
if __name__ == "__main__":
    main()
