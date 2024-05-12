from collections import defaultdict
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QHBoxLayout, QLabel, QLineEdit, QFormLayout


class RazrezWindow(QWidget):
    def __init__(self, dataFrame):
        super().__init__()
        self.dataFrame = dataFrame
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Построение Разреза')
        self.setGeometry(100, 100, 800, 600)

        mainLayout = QHBoxLayout(self)

        # Left panel for input
        leftPanel = QVBoxLayout()
        titleLabel = QLabel("Формирование разреза по горным выработкам")
        leftPanel.addWidget(titleLabel)

        self.numberInput = QLineEdit()
        self.numberInput.setPlaceholderText("Введите количество выработок")
        leftPanel.addWidget(self.numberInput)

        okButton = QPushButton("Ок")
        okButton.clicked.connect(self.createInputFields)
        leftPanel.addWidget(okButton)

        self.formLayout = QFormLayout()
        leftPanel.addLayout(self.formLayout)

        buildButton = QPushButton("Построить разрез")
        buildButton.clicked.connect(self.buildPlot)
        leftPanel.addWidget(buildButton)

        cancelButton = QPushButton("Отмена")
        cancelButton.clicked.connect(self.resetInputs)
        leftPanel.addWidget(cancelButton)

        mainLayout.addLayout(leftPanel)

        # Right panel for plot
        rightPanel = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        rightPanel.addWidget(self.canvas)
        mainLayout.addLayout(rightPanel)
        self.setLayout(mainLayout)

    def createInputFields(self):
        # Clear old fields
        for i in reversed(range(self.formLayout.count())):
            self.formLayout.itemAt(i).widget().deleteLater()

        try:
            count = int(self.numberInput.text())
            self.nameInputs = []
            for i in range(count):
                lineEdit = QLineEdit()
                self.nameInputs.append(lineEdit)
                self.formLayout.addRow(f"Введите название выработки {i+1}:", lineEdit)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Введите корректное количество выработок", QMessageBox.Ok)

    def buildPlot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        points = []
        unique_names = set()
        for nameInput in self.nameInputs:
            point_name = nameInput.text()
            if point_name in self.dataFrame['Name'].values and point_name not in unique_names:
                point_data = self.dataFrame[self.dataFrame['Name'] == point_name].iloc[0]
                points.append((point_data['X'], point_data['Y'], point_data['Height'], point_name))
                unique_names.add(point_name)
            elif point_name in unique_names:
                QMessageBox.warning(self, "Ошибка", f"Дублирование названий точек: {point_name}", QMessageBox.Ok)
                return

        if not points:
            QMessageBox.warning(self, "Ошибка", "Нет данных для введенных названий", QMessageBox.Ok)
            return

        x, y = [20], [points[0][2]]  # Start at x = 20 with the first point's height
        labels = [f"{points[0][3]}\n{points[0][2]}"]
        
        # Calculate subsequent points based on sorted order
        for i in range(1, len(points)):
            dx = np.sqrt((points[i][0] - points[i-1][0])**2 + (points[i][1] - points[i-1][1])**2)
            new_x = x[-1] + dx
            x.append(new_x)
            y.append(points[i][2])
            labels.append(f"{points[i][3]}\n{points[i][2]}")
            # Annotate distances between points just above the x-axis
            ax.text((x[i-1] + x[i])/2, min(y) - 19, f'{dx:.2f}', ha='center', va='top', fontsize=9)

        for i in range(len(points)):
            points[i] = (x[i], points[i][2], points[i][2], points[i][3])

        
        ax.plot(x, y, marker='o', color='blue', linestyle='-')  # Plot a blue line
        
        new_dots = self.get_addon_points(points)
        
        ax.set_xlim(0, x[-1] + 20)
        ax.set_ylim(int(min(y)-20), int(max(y))+3)
        # Draw horizontal lines connecting to the graph edges
        ax.hlines(y[0], 0, x[0], color='blue', linestyle='-')  # Extend to the left
        ax.hlines(y[-1], x[-1], x[-1] + 20, color='blue', linestyle='-')  # Extend to the right
        # Draw vertical lines to the x-axis
        ax.vlines(x, int(min(y)-20), y, color='black', linestyle='-')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(int(min(y)-20), int(max(y))+3, 1.0))  # Set y-ticks correctly using ax


        grouped_points = defaultdict(list)
        x_coords = set()
        for x_val, depth, y_val, name in new_dots:
            grouped_points[(name, x_val)].append((y_val, depth))
            x_coords.add(x_val)

        
        sorted_x_coords = sorted(x_coords)

       # Сортировка точек по x для каждого имени и рисование точек
        last_connected = defaultdict(dict)
        for (name, x_val), points in grouped_points.items():
            y_vals, depths = zip(*sorted(points))
            ax.scatter([x_val] * len(y_vals), y_vals, color='red', s=5)
            for y_val, depth in zip(y_vals, depths):
                ax.text(x_val + 3, y_val, name, verticalalignment='center', color='black')
                ax.text(x_val - 10, y_val, f"{depth}", verticalalignment='center', horizontalalignment='left', color='black')

        # Словарь для следующих X
        next_x_map = {sorted_x_coords[i]: sorted_x_coords[i + 1] for i in range(len(sorted_x_coords) - 1)}

        # Инициализация словаря для отслеживания соединений по именам и осям X
        connected_points = defaultdict(lambda: defaultdict(set))

        # Соединение точек с одинаковыми именами
        for (name, x_val), points in grouped_points.items():
            if name and x_val in next_x_map and (name, next_x_map[x_val]) in grouped_points:
                next_x = next_x_map[x_val]
                for y_val, depth in sorted(points):
                    next_points = grouped_points[(name, next_x)]
                    # Поиск точки с минимальным расстоянием по Y, но с учетом возможной корректировки по вашим требованиям
                    closest_y = min(next_points, key=lambda p: (abs(p[0] - y_val), p[1]))  # второй критерий сортировки может быть добавлен при необходимости
                    ax.plot([x_val, next_x], [y_val, closest_y[0]], color='blue')
                    connected_points[name][x_val].add(y_val)
                    connected_points[name][next_x].add(closest_y[0])



        self.canvas.draw()


    def get_addon_points(self, well_names):
        addon_points = []
        try:
            for name in well_names:
                # Filter the dataframe for the specific well name
                well_data = pd.read_excel('data.xlsx', sheet_name=name[3], skiprows=3, usecols='C,G,H', converters={'C': str, 'G': str, 'H': str}).dropna()
                well_data.columns = ['C', 'G', 'H']
                
                # Process each row in the filtered dataframe
                for index, row in well_data.iterrows():
                    g = float(row['G'])
                    depth = row['H']
                    c_value = row['C']
                    
                    # Attempt to extract a number within parentheses from the 'C' column
                    match = re.search(r'\((\d+)\)', c_value)
                    label = match.group(1) if match else ""
                    
                    # Calculate new Y coordinate
                    new_y = name[2] - depth
                    
                    # Append the tuple (X, new_Y, label)
                    addon_points.append((name[0], depth, new_y, label))
            
            return addon_points
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", "Невозможно загрузить данные: " + str(e), QMessageBox.Ok)

    def resetInputs(self):
        self.numberInput.clear()
        for i in reversed(range(self.formLayout.count())):
            self.formLayout.itemAt(i).widget().deleteLater()


class MapWindow(QWidget):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.initUI()

    def initUI(self):

        self.setWindowTitle('Карта')
        layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)  # Добавление канвы в layout

        # Кнопка для активации функции "Разрез"
        self.btn_cut = QPushButton('Разрез', self)
        self.btn_cut.clicked.connect(self.cut_action)
        layout.addWidget(self.btn_cut)

        self.setLayout(layout)
        self.showMap()

    def showMap(self):
        img = mpimg.imread("karta.png")
        ax = self.figure.add_subplot(111)
        
        # Расчет новых пределов с учетом буфера
        y_buffer = (self.data['Y'].max() - self.data['Y'].min()) * 0.05
        x_buffer = (self.data['X'].max() - self.data['X'].min()) * 0.05
        y_min, y_max = self.data['Y'].min() - y_buffer, self.data['Y'].max() + y_buffer
        x_min, x_max = self.data['X'].min() - x_buffer, self.data['X'].max() + x_buffer
        
        # Отображение изображения с новыми пределами
        ax.imshow(img, extent=[y_min, y_max, x_min, x_max], aspect='auto')
        ax.scatter(self.data['Y'], self.data['X'], color='blue', s=10)

        # Аннотация каждой точки
        for i in range(len(self.data)):
            ax.annotate(f"{self.data['Name'][i]}\n{self.data['Height'][i]}", (self.data['Y'][i], self.data['X'][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        ax.set_title('Расположение точек на карте')
        ax.set_ylabel('X координата')
        ax.set_xlabel('Y координата')
        
        ax.set_xlim([y_min, y_max])
        ax.set_ylim([x_min, x_max])
        
        self.canvas.draw()

    def cut_action(self):
        # Здесь может быть логика для изменения отображения или данных
        self.rarzWindow = RazrezWindow(self.data)
        self.rarzWindow.show()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.filePath = None
        self.dataFrame = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Формирование схемы расположения горных выработок')
        self.setGeometry(100, 100, 850, 400)
        layout = QVBoxLayout()

        self.loadButton = QPushButton('Загрузить файл', self)
        self.loadButton.clicked.connect(self.loadFile)
        layout.addWidget(self.loadButton)

        self.generateButton = QPushButton('Сформировать схему', self)
        self.generateButton.clicked.connect(self.generateScheme)
        layout.addWidget(self.generateButton)

        centralWidget = QWidget(self)
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def loadFile(self):
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, 'Выберите файл xlsx', '', 'Excel Files (*.xlsx)', options=options)
        if filePath:
            self.filePath = filePath
            self.dataFrame = pd.read_excel(filePath, skiprows=3, sheet_name='Каталог горных выработок', usecols="B,C,D,E", converters={'B': str, 'C': str, 'D': str, 'E': str})
            self.dataFrame.columns = ['Name','X', 'Y', 'Height'] 
            print("File loaded:", filePath)

    def generateScheme(self):
        if self.filePath and self.dataFrame is not None:
            self.mapWindow = MapWindow(self.dataFrame)
            self.mapWindow.show()
        else:
            QMessageBox.warning(self, 'Error', 'Файл не загружен', QMessageBox.Ok)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    ex.show()
    sys.exit(app.exec_())
