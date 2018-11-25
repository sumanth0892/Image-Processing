from keras import layers,models
CatOrDog = models.Sequential()
CatOrDog.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Conv2D(64,(3,3),activation='relu'))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Conv2D(128,(3,3),activation='relu'))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Conv2D(128,(3,3),activation='relu'))
CatOrDog.add(layers.MaxPooling2D((2,2)))
CatOrDog.add(layers.Flatten())
CatOrDog.add(layers.Dense(512,activation='relu'))
CatOrDog.add(layers.Dense(1,activation='sigmoid'))
from keras import optimizers
CatOrDog.compile(loss='binary_crossentropy',optimizer = optimizers.RMSprop(lr=1e-4),metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator
training_data = ImageDataGenerator(rescale=1./255)
testing_data = ImageDataGenerator(rescale=1./255)
training_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),batch_size=20,class_mode='binary')
validating_generator = test_datagen.flow_from_directory(validating_dir,target_size=(150,150),batch_size=20,class_mode='binary')
history = model.fit_generator(training_generator,steps_per_epoch=100,epochs=30,validation_data=validating_generator,validation_steps=50)
model.save('CatsOrDogsmicro.h5')

#Plotting the display curves of loss and accuracy
import matplotlib.pyplot as plt
accuracy=history.history['acc']
validation_acc=history.history['val_acc']
loss=history_history['loss']
validation_loss = history.history['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'ro',label='Training Accuracy')
plt.plot(epochs,val_acc,'r',label='Validation Accuracy')
plt.title('Training and Validation accuracy')

plt.figure()

plt.plot(epochs,loss,'bo',label='Training Loss')
plt.plot(epochs,val_loss,'b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
