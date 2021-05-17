#################################################################PLOT LOSS AND ACCURACY AND TRAINING AND VAL SET #######################################################

def plot_loss_accuracy(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='best')
    plt.show()


############################################################# CROSS VALIDATION DEEP LEARNING FUNC  #######################################################################"


def crossval__for_deep(X_train, X_test, y_train, y_test):

    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler


    kf = KFold(n_splits=10)
    kf.get_n_splits(X)

    results = []

    for train_index, test_index in kf.split(X):
        # Split the data into train and test
        X_train, X_test, y_train, y_test = train_test_split(X,y_cat, test_size=0.3)
        
        # Use the standard scaler
        scaler = StandardScaler()
        
        scaler.fit(X_train)
        
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize the model
        model = initialize_model()
        
        # Fit the model on the train data
        history = model.fit(X_train_scaled, y_train, epochs=150, verbose = 0)
        
        # Evaluate the model on the test data and append the result in the `results` variable
        results.append(model.evaluate(X_test, y_test, verbose = 0))