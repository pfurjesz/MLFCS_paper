import tme   # sample import from other files i import for example the tme 

class ensamble(data,train,val,test):
    def __init__(self, data):
        super().__init__(data)
        self.data = data

    def train(self):
        return self.data

    def predict(self):
        return self.data
    
    def evaluate(self):
        return self.data