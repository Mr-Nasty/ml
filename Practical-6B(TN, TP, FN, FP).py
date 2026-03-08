from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix , accuracy_score , precision_score , recall_score, f1_score
import matplotlib.pyplot as plt

y_true=[1,0,1,1,0,1,0,0]
y_pred=[1,0,0,1,0,1,1,0]

cm=confusion_matrix(y_true,y_pred)
print(f"Confusion Matrix : \n{cm}")

disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['Class 0','Class 1'])
disp.plot(cmap=plt.cm.Blues)
plt.show()

TN,FP,FN,TP=cm.ravel()
print("True Positive:",TP)
print("True Negative:",TN)
print("False Positive:",FP)
print("False Negative:",FN)

precision=TP/(TP+FP)
recall=TP/(TP+FN)
f1=2* (precision*recall)/(precision+recall)
accuarcy=(TP+TN)/(TP+TN+FP+FN)
print ("accuarcy", accuarcy)
print("precision", precision)
print("recall", recall)
print("f1 score", f1)

accuarcy=accuracy_score(y_true, y_pred)
print("accuarcy", accuarcy)

precision=precision_score(y_true,y_pred)
print("precision", precision)

recall=recall_score(y_true, y_pred)
print("recall", recall)

f1=f1_score(y_true,y_pred)
print("f1 score", f1)
