import  tensorflow  as  tf
from tensorflow  import  keras
from  tensorflow.keras import  layers


fashion_mnist =  keras.datasets.fashion_mnist
(train_images,train_labels) , (test_images,test_labels) = fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([
    layers.Flatten(input_shape= (28,28)),
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
    
    
])

model.compile(optimizer= 'adam', loss = 'sparse_categorical_crossentropy')
model.fit(train_images,train_labels,epochs= 5)