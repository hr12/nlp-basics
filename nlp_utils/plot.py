import matplotlib.pyplot as plt

def plot_metrics(history, metrics=['loss', 'accuracy']):
    
    num_metrics = len(metrics)
    
    
    for i,metric in enumerate(metrics):
        epochs = range(1, (len(history[metric])+1))
        plt.subplot(num_metrics, 1,(i+1))
        plt.plot(epochs, history[metric], label = 'training_'+metric)
        plt.plot(epochs, history['val_'+metric], label = 'validation_'+metric)
        plt.title("Training and validation "+metric)
        plt.ylabel(metric)
        plt.xlabel('epochs')
        plt.legend()
        plt.show()