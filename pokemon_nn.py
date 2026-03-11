import torch 
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class Model(torch.nn.Module):
    def __init__(self, in_features=3, out_features = 2, h1=5,h2=5,h3=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, h1)
        self.fc2 = torch.nn.Linear(h1, h2)
        self.fc3 = torch.nn.Linear(h2, h3)
        self.out = torch.nn.Linear(h3, out_features)

    def forward(self, X):
        # linear --> relu --> linear --> relu --> linear --> relu --> output
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.out(X)
        return X
        

    def train(self, X_train, y_train):
        epochs = 1000
        optimizer = optim.Adam(self.parameters(), lr=0.03)

        criterion = torch.nn.CrossEntropyLoss()

        for i in range(epochs):
            print(f"Epoch: {i+1}")
            y_pred = self.forward(X_train)
            
            optimizer.zero_grad()

            loss = criterion(y_pred, y_train)
            loss.backward()
        
            optimizer.step()

def main():
    df = pd.read_csv("pokemon.csv")

    X = df[["Attack", "Defense", "Sp. Atk"]].astype(float)
    y = df["Legendary"].astype(float)

    X = X.values
    y = y.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    model = Model()

    model.train(X_train, y_train)

    pred = model.forward(X_test).argmax(axis=1)
    

    plt.scatter(X_test[:, 0], X_test[:, 1], c=pred, cmap="coolwarm", s = X_test[:, 2], marker="x")
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", s = X_test[:, 2])

    plt.title(f"Accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")

    plt.show()


main()



