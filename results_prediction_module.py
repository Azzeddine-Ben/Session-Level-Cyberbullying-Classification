# -*- coding: utf-8 -*-
"""

@author: BENAISSA-Azzeddine
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.metrics import confusion_matrix, classification_report

  
def print_confusion_matrix(confusion_matrix, class_names, figsize = (5,3), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def make_predictions(classifier, X_test_):
    test_predictions = classifier.predict(X_test_, batch_size=32)
    return test_predictions

def visualize_predictions(y_test, predictions, nn = False):
  # if nn == True:
  y = list(y_test)
  y_pred = (predictions > 0.5)
  cm = confusion_matrix(y, y_pred)
  # else:
  #   cm = confusion_matrix(y_test, predictions)
  sns.heatmap(cm, annot=True)
  return cm

def visualize_report(y_test, predictions): 
  clf_report = classification_report(y_test, predictions)
  print(clf_report)
  clf_report = classification_report(y_test, predictions, output_dict=True)
  return clf_report
  
def predict_and_visualize(classifier, X_test, y_test):
  predictions = make_predictions(classifier, X_test)
  y_test = list(y_test)
  predictions = (predictions > 0.5)
  cm = confusion_matrix(y_test, predictions)
  fig = print_confusion_matrix(cm, ['0', '1'])
  clf_report = visualize_report(y_test, predictions)
  return clf_report, fig
# =============================================================================
#   if return_recall == True:
#     return MinorityClass_recall(y_test, predictions)
# =============================================================================

def print_learning_curves(history, saving_folder_path):
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.grid()
  # plt.xlim(12.5,15)
  plt.savefig(saving_folder_path + '/Train learning curves')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.grid()
  # plt.xlim(5,8)
  plt.savefig(saving_folder_path + '/Validation learning curves')
  # plt.show()
  