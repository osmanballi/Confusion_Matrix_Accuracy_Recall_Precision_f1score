import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
def confusion(y_true,y_pred):
    fig = plt.figure(figsize=(8,8)) # Set Figure
    mat = confusion_matrix(y_true, y_pred) # Confusion matrix

    # Plot Confusion matrix
    sns.set(font_scale=2)
    sns.heatmap(mat.T, square=True, annot=True, cbar=False, cmap=plt.cm.YlGnBu,fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted Values')
    plt.ylabel('True Values');
    plt.show();
    test_accuracy = accuracy_score(y_true,y_pred)
    print(round(test_accuracy,3))

    precision = precision_score(y_true,y_pred, average='weighted')
    print(round(precision,3))

    recall = recall_score(y_true,y_pred, average='weighted')
    print(round(recall,3))

    f1score = f1_score(y_true, y_pred, average='weighted')
    print(round(f1score,3))