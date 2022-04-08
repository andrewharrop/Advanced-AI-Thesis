from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def plot_confusion_matrix(confusion_mtx):

    # creates a graph template
    fig, ax = plt.subplots(dpi = 120)
    # plot the confusion matrix
    sns.heatmap(confusion_mtx , annot=True)
    # x-axis label
    ax.set_xlabel('Predicted labels') 
    # y-axis label  
    ax.set_ylabel('True labels')     
    # title
    ax.set_title('Confusion Matrix')    
    plt.show()