# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1: Import the required libraries, including PyTorch, NumPy, Matplotlib, and Scikit-learn. Load the MNIST dataset and preprocess the images by converting them into tensors and normalizing pixel values.

### STEP 2: Split the dataset into training and testing sets, and create data loaders to process images in batches.

### STEP 3: Design the CNN model with two convolutional layers containing 16 and 32 filters, respectively. Apply ReLU activation and max-pooling layers to extract important image features and reduce spatial dimensions.

### STEP 4: Flatten the extracted features and pass them through a fully connected layer with 128 neurons. Use an output layer with 10 neurons to classify the handwritten digits from 0 to 9.

### STEP 5: Define the cross-entropy loss function and Adam optimizer with a learning rate of 0.001. Train the model for 3 epochs using forward propagation, loss calculation, backpropagation, and weight updates.

### STEP 6: Evaluate the trained model using the test dataset. Plot the training loss, generate a confusion matrix and classification report, and predict the digit classes of new sample images.


## PROGRAM

### Name: Vignesh S

### Register Number: 212224110061

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs=3):

    model.train()
    for epoch in range(num_epochs):
      running_loss = 0.0
      for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

```

### OUTPUT

## Training Loss per Epoch

<img width="1772" height="772" alt="image" src="https://github.com/user-attachments/assets/88dd5870-923f-471a-b1e7-054e883f7381" />


## Confusion Matrix

<img width="1050" height="796" alt="image" src="https://github.com/user-attachments/assets/d366b5f6-c218-4e4a-8921-33d10dd3a948" />


## Classification Report

<img width="1731" height="427" alt="image" src="https://github.com/user-attachments/assets/bb852ce5-a687-4b62-bade-a4d4591efff2" />


### New Sample Data Prediction

<img width="932" height="692" alt="image" src="https://github.com/user-attachments/assets/f08ecbf3-cf5d-4fea-96c5-e6a3d51d07a4" />

## RESULT 
Thus, a Convolutional Neural Network (CNN) classification model was successfully developed and trained using the MNIST handwritten digit dataset. The model learned image features through convolutional and pooling layers and classified handwritten digits into 10 categories. Its performance was evaluated using the training loss graph, confusion matrix, classification report, and predictions on new sample images.

