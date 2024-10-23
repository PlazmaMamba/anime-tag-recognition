import sys
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QSlider
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from torchvision import transforms, models
from PIL import Image
import json

class TagPredictionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.loadModel()

    def initUI(self):
        self.setWindowTitle('Anime Tag Prediction')
        self.setGeometry(100, 100, 400, 600)

        layout = QVBoxLayout()

        self.selectButton = QPushButton('Select Image')
        self.selectButton.clicked.connect(self.selectImage)
        layout.addWidget(self.selectButton)

        self.imageLabel = QLabel()
        self.imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.imageLabel)

        self.sliderLabel = QLabel('Number of tags to display: 5')
        layout.addWidget(self.sliderLabel)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)
        self.slider.setMaximum(20)
        self.slider.setValue(5)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.updateSliderLabel)
        layout.addWidget(self.slider)

        self.resultLabel = QLabel('Predicted Tags:')
        layout.addWidget(self.resultLabel)

        self.setLayout(layout)

    def updateSliderLabel(self, value):
        self.sliderLabel.setText(f'Number of tags to display: {value}')
        if hasattr(self, 'last_prediction'):
            self.updatePredictionDisplay(self.last_prediction)

    def loadModel(self):
        # Load the trained model
        self.model = models.resnet50(pretrained=False)
        num_ftrs = self.model.fc.in_features
        
        # Load tag mapping
        with open('processed_data/tag_mapping.json', 'r') as f:
            self.tag_mapping = json.load(f)
        
        self.model.fc = torch.nn.Linear(num_ftrs, len(self.tag_mapping))
        self.model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
        self.model.eval()

        # Define image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def selectImage(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Select Image', '', 'Image files (*.jpg *.jpeg *.png)')
        if fname:
            pixmap = QPixmap(fname)
            self.imageLabel.setPixmap(pixmap.scaled(300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            # Predict tags
            image = Image.open(fname).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.sigmoid(outputs)
            
            self.last_prediction = probabilities
            self.updatePredictionDisplay(probabilities)

    def updatePredictionDisplay(self, probabilities):
        num_tags = self.slider.value()
        tag_probs = [(tag, probabilities[0][idx].item()) for tag, idx in self.tag_mapping.items()]
        tag_probs.sort(key=lambda x: x[1], reverse=True)
        
        top_tags = tag_probs[:num_tags]
        result_text = 'Predicted Tags:\n'
        for tag, prob in top_tags:
            result_text += f'{tag}: {prob:.2f}\n'
        
        self.resultLabel.setText(result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TagPredictionUI()
    ex.show()
    sys.exit(app.exec_())