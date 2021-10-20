import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_confusion_matrix



def visualizeScale(df_col, col, scale_list):
    plt.subplots(figsize=(6, 4))
    # df_col.plot(kind='density')
    # df_col.plot.hist(bins=list(np.arange(1.1, 4.1, 0.1)))
    df_col.plot.hist()
    plt.rcParams['font.family'] = 'tahoma'
    plt.title(col)
    if len(scale_list) > 1:
        for l in range(len(scale_list)-1):
            plt.axvline(scale_list[l], color="r")
    plt.show()
    
def showAllConfusion(list_conf):
    plt.rcParams["figure.figsize"] = (10,3)
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
    # fig.suptitle('Horizontally stacked subplots')
    plot_confusion_matrix(conf_mat=list_conf[0], axis=ax1)
    plot_confusion_matrix(conf_mat=list_conf[1], axis=ax2)  
    plot_confusion_matrix(conf_mat=list_conf[2], axis=ax3)  
    plot_confusion_matrix(conf_mat=list_conf[3], axis=ax4)  
    plot_confusion_matrix(conf_mat=list_conf[4], axis=ax5)
    plt.ylabel('')
    plt.xlabel('')
    plt.show()
    
def howFitting(train_list, validate_list):
    plt.subplots(figsize=(6, 4))
    x = np.arange(1, len(train_list)+1, 1)   
    plt.plot(x, train_list, "-o", label= "Train score")
    plt.plot(x, validate_list, "--o", label= "Validation score")

    plt.title("Model score")
    plt.xlabel("Folds")
    plt.ylabel("Score")
    plt.legend()
    
    # find peak
    for yi in [train_list, validate_list]: 
        for i in range(len(yi)):
            plt.text(x[i], yi[i], np.round(yi[i],3))
                   
    plt.xticks(x)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.show()
