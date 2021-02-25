from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from datetime import datetime
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

callback = EarlyStopping(monitor='val_loss', patience=5)

def train_model(classifier, train_x, train_lables, validation_x, validation_labels,
                test_vectors = None, neural_network = False, epochs = None,
                submissions_data = None, submissions_file_prefix="submission"):
        
    # fitting the model
    if not neural_network:
        classifier.fit(train_x, train_lables)
    else:
        classifier.fit(train_x, train_lables, validation_data = (validation_x, validation_labels),epochs=epochs, callbacks = callback)
    
    # validation
    predictions = classifier.predict(validation_x)
    
    if neural_network:
        predictions = np.where(predictions>0.5,1,0)
        
    
    # reporting important metrics
    classification_report_matrix = classification_report(validation_labels,predictions,labels=[1,0])
    print('Classification report : \n')
    print(classification_report_matrix)
    
    # competition submissions data
    if (submissions_data is not None) & (test_vectors is not None):
        test_preds = classifier.predict(test_vectors)
        if neural_network:
#             test_preds = test_preds.argmax(axis = -1)
            test_preds = np.where(test_preds>0.5,1,0)
    
        submissions_data['target'] = test_preds
        now = datetime.now()
        now = now.strftime("%Y%m%d%H%M%S")
        file_export_path = "data/{}_{}.csv".format(submissions_file_prefix, now)
        print("Exporting data to: \n")
        print("\t {}".format(file_export_path))
        submissions_data.to_csv(file_export_path, index = False)
        