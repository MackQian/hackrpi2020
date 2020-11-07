import numpy as np
import agent.py as a
global iRate
global price
global food
global work

class World:
    def __init__(self, size):
        self.board=np.zeros((size,size))
    def display(self):
        pass

def main():
    iRate=0.01
    price=10
    board=np.array([[0,0,0],[0,1,0],[0,-1,0]])

if __name__ == "__main__":
    main()
