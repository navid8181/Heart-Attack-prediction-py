# %% Packages
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# %%
df = pd.read_csv('cvFiles/heart.csv')
df.head()
# %%
X = np.array(df.loc[:, df.columns != 'output'])
Y = np.array(df['output'])


print(X.shape, end=",")
print(Y.shape)
# print(f"X: {X.shape()}, Y: {Y.shape()}")

# %%

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.2, random_state=123)

# %%
scaler = StandardScaler()
X_TrainScale = scaler.fit_transform(X_Train)
X_tetsScale = scaler.transform(X_Test)
# %%


class NeuralNetworkFromScratch:
    def __init__(self, LR, X_Train,
                 X_Test, Y_Train, Y_Test):
        self.w = np.random.randn(X_TrainScale.shape[1])
        self.b = np.random.randn()

        self.LR = LR
        self.X_Train = X_Train
        self.X_Test = X_Test
        self.Y_Train = Y_Train
        self.y_Test = Y_Test
        self.L_Train = []
        self.L_Test = []

    def activation(self, x):
        return 1/(1 + np.exp(-x))

    def deactivation(self, x):
        return self.activation(x) * (1 - self.activation(x))

    def forward(self, X):
        hidden_1 = np.dot(X, self.w) + self.b
        activate_1 = self.activation(hidden_1)
        return activate_1

    def backward(self, X, y_true):
        # calc gradients
        hidden_1 = np.dot(X, self.w) + self.b
        y_pred = self.forward(X)
        dL_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.deactivation(hidden_1)
        dhidden1_db = 1
        dhidden1_dw = X

        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw
        return dL_db, dL_dw

    def optimizer(self, dL_db, dL_dw):
        self.b = self.b - dL_db * self.LR
        self.w = self.w - dL_dw * self.LR

    def Train(self, ITERATIONS):

        for i in range(ITERATIONS):
            rand_pos = np.random.randint(len(self.X_Train))

            yTrain_true = self.Y_Train[rand_pos]
            yTrain_Pred = self.forward(self.X_Train[rand_pos])

            L = np.square(yTrain_Pred - yTrain_true)

            self.L_Train.append(L)

            dL_db, dL_dw = self.backward(
                self.X_Train[rand_pos], self.Y_Train[rand_pos])

            self.optimizer(dL_db, dL_dw)

            L_Sum = 0

            for j in range(len(self.X_Test)):
                y_true = self.y_Test[j]
                y_pred = self.forward(self.X_Test[j])

                L_Sum += np.square(y_pred - y_true)

            self.L_Test.append(L_Sum)

        print("Training is success full")


# %%

LR = 0.3
ITERATIONS = 10000


nn = NeuralNetworkFromScratch(LR=LR, X_Train=X_TrainScale,
                              Y_Train=Y_Train, X_Test=X_tetsScale, Y_Test=Y_Test)

nn.Train(ITERATIONS=ITERATIONS)
# %%

sb.lineplot(x=list(range(len(nn.L_Test))),
            y=nn.L_Test)


# %%

total = X_tetsScale.shape[0]

y_preds = []
correct = 0
for i in range(total):
    y_true = Y_Test[i]
    y_pred = np.round(nn.forward(X_tetsScale[i]))

    y_preds.append(y_pred)
    correct += 1 if y_pred == y_true else 0

print(correct/total)


# %%

confusion_matrix(y_true=Y_Test,y_pred=y_preds)
# %%
