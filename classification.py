import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report

def classifyFoodInspectionReports():
    my_data = pd.read_csv('Food_Inspections_Prepared_Data_Prepared.csv')
    X, y = my_data.iloc[:,3:-1], my_data['Results']
    y = y.to_frame()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 2)

    s = (X.dtypes == 'object')
    object_cols = list(s[s].index)

    label_X_train = X_train.copy()
    label_X_test = X_test.copy()
    label_X = X.copy()
    # Applying ordinal encoder to each column with categorical data
    ordinal_encoder = OrdinalEncoder()
    label_X[object_cols] = ordinal_encoder.fit_transform(X[object_cols])
    label_X_train[object_cols] = ordinal_encoder.transform(X_train[object_cols])
    label_X_test[object_cols] = ordinal_encoder.transform(X_test[object_cols])

    #TRAIN and PREDICT
    model1 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=18)
    model1 = model1.fit(label_X_train, y_train)
    print(type((model1.predict(label_X_test))))
    preds1 = pd.Series(list(model1.predict(label_X_test)))
    preds1 = preds1.to_frame()


    model2 = LogisticRegression(random_state=2, max_iter=200)
    model2.fit(label_X_train, y_train)
    preds2 = model2.predict(label_X_test)

    #print(my_data.corr(numeric_only=True))
    print("\n\n********DECISION TREE******")
    print('Mean abs', mean_absolute_error(y_test, preds1))
    print('Train clf Score', model1.score(label_X_train, y_train))
    print('Test clf Score', model1.score(label_X_test, y_test))
    print('Accuracy Score', accuracy_score(y_test, preds1))
    print('Precision Score ', precision_score(y_test, preds1, average = None))

    cnf_matrix = metrics.confusion_matrix(y_test, preds1)
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Decision Trees - Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    matplotlib.pyplot.show()

    target_names = ['Inspection Failed', 'Inspection Passed']
    print("\n", classification_report(y_test, preds1, target_names=target_names))


    print("\n********LOGISTIC REGRESSION******")
    print('Mean abs', mean_absolute_error(y_test, preds2))
    print('Train clf Score', model2.score(label_X_train, y_train))
    print('Test clf Score', model2.score(label_X_test, y_test))
    print('Accuracy Score', accuracy_score(y_test, preds2))
    print('Precision Score ', precision_score(y_test, preds2, average = None))

    cnf_matrix = metrics.confusion_matrix(y_test, preds2)
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Logistic Regression - Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    matplotlib.pyplot.show()

    target_names = ['Inspection Failed', 'Inspection Passed']
    print("\n", classification_report(y_test, preds2, target_names=target_names))
