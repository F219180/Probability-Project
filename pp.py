import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel,QLineEdit, QTextEdit, QVBoxLayout, QWidget, QPushButton, QComboBox, QDialog, QStackedWidget,QTableWidget,QTableWidgetItem
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from scipy import stats
from PyQt5.QtGui import QPixmap
from statsmodels.graphics.tsaplots import plot_acf

class GraphSelectionWindow(QDialog):
    def __init__(self, stacked_widget):
        super().__init__()


        # Set up the window
        self.setWindowTitle('Graph Options')
        self.setGeometry(200, 200, 400, 200)
        self.setStyleSheet("background-color: lightgoldenrodyellow;")

        self.stacked_widget = stacked_widget  # Access to stacked widget

        # Create combo box for graph selection
        self.graph_combo = QComboBox(self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.graph_combo)
        self.setLayout(self.layout)
   # Add a figure to display graphs
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        # List of available graphs
        self.graph_options = [ 
            'Histogram of Literacy Rate',
            'Histogram of Year-wise Distribution of Literacy Rate',
            'Bar Chart of Literacy Rates by Country',
            'Bar Chart of Literacy Rates by Region',
            'Bar Chart of Literacy Rates by Gender',
            'Multiple-Bar Chart of Average Literacy Rates by Country and Gender',
            'Multiple-Bar Chart of Literacy Rates by Region and Gender',
            'Multiple-Bar Chart of Literacy Rates by Region and Country',
            'Component-Bar Chart of Literacy Rates by Country and Gender',
            'Component-Bar Chart of Literacy Rates by Region and Gender',
            'Component-Bar Chart of Literacy Rates by Region and Country',
            'Pie Chart of Literacy Rates by Region',
            'Pie Chart of Literacy Rates by Country',
            'Pie Chart of Literacy Rates by Gender',
            'Boxplot of Literacy Rates by Region',
            'Boxplot of Literacy Rates by Gender',
            'Boxplot of Literacy Rates by Country',
            'Scatter Plot of Literacy Rate vs Year with Linear Regression'
            # Add more options as needed
        ]
        self.graph_combo.addItems(self.graph_options)
        
        self.graph_combo.currentIndexChanged.connect(self.plot_selected_graph)
        # Back button
        self.back_button = QPushButton('Back', self)
        self.back_button.clicked.connect(self.back_to_main)
        self.layout.addWidget(self.back_button)


    def plot_selected_graph(self):
        self.figure.clear()
        selected_graph = self.graph_combo.currentText()


        # Read the CSV file (change the path to your LR.csv file)
        file_path = '/Users/syedadaniya/Desktop/LR.csv'
        df = pd.read_csv(file_path)

        # Perform different plots based on the selected graph option
        if selected_graph == 'Histogram of Literacy Rate':
            ax = self.figure.add_subplot(111)
            ax.hist(df['Literacy rate'], bins='auto', color='skyblue', edgecolor='black')
            ax.set_xlabel('Literacy Rate')
            ax.set_ylabel('Frequency')
            ax.set_title('Histogram of Literacy Rate')
            ax.grid(True)


        if selected_graph == 'Histogram of Year-wise Distribution of Literacy Rate':
                 ax = self.figure.add_subplot(111)
                 for year, group_data in df.groupby('Year')['Literacy rate']:
                     ax.hist(group_data, bins='auto', alpha=0.5, label=str(year))

                     ax.set_xlabel('Literacy Rate')
                     ax.set_ylabel('Frequency')
                     ax.set_title('Year-wise Distribution of Literacy Rate')
                     ax.legend(title='Year')
                     ax.grid(True)

        # Add other plot types similarly

        self.canvas.draw()

        
        if selected_graph == 'Bar Chart of Literacy Rates by Country':
             ax = self.figure.add_subplot(111)
             countries = df['Country']
             literacy_rates = df['Literacy rate']
             ax.bar(countries, literacy_rates, color='skyblue')
             ax.set_xlabel('Country')
             ax.set_ylabel('Literacy rate')
             ax.set_title('Literacy Rates by Country')
             ax.set_xticklabels(countries, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Bar Chart of Literacy Rates by Region':
             ax = self.figure.add_subplot(111)
             region = df['Region']
             literacy_rates = df['Literacy rate']
             ax.bar(region, literacy_rates, color='skyblue')
             ax.set_xlabel('Region')
             ax.set_ylabel('Literacy rate')
             ax.set_title('Literacy Rates by Region')
             ax.set_xticklabels(region, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Bar Chart of Literacy Rates by Gender':
             ax = self.figure.add_subplot(111)
             gender = df['Gender']
             literacy_rates = df['Literacy rate']
             ax.bar(gender, literacy_rates, color='skyblue')
             ax.set_xlabel('Gender')
             ax.set_ylabel('Literacy rate')
             ax.set_title('Literacy Rates by Gender')
             ax.set_xticklabels(gender, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Multiple-Bar Chart of Average Literacy Rates by Country and Gender':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby(['Country', 'Gender'])['Literacy rate'].mean().unstack()
             grouped_data.plot(kind='bar', figsize=(10, 6), ax=ax)
             ax.set_xlabel('Country')
             ax.set_ylabel('Average Literacy Rate')
             ax.set_title('Average Literacy Rate by Country and Gender')
             ax.legend(title='Gender')
             ax.set_xticklabels(grouped_data.index, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Multiple-Bar Chart of Literacy Rates by Region and Gender':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby(['Region', 'Gender'])['Literacy rate'].mean().unstack()
             grouped_data.plot(kind='bar', figsize=(10, 6), ax=ax)
             ax.set_xlabel('Region')
             ax.set_ylabel('Average Literacy Rate')
             ax.set_title('Average Literacy Rate by Region and Gender')
             ax.legend(title='Gender')
             ax.set_xticklabels(grouped_data.index, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Multiple-Bar Chart of Literacy Rates by Region and Country':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby(['Region', 'Country'])['Literacy rate'].mean().unstack()
             grouped_data.plot(kind='bar', figsize=(10, 6), ax=ax)
             ax.set_xlabel('Region')
             ax.set_ylabel('Average Literacy Rate')
             ax.set_title('Average Literacy Rate by Country and Region')
             ax.legend(title='Country')
             ax.set_xticklabels(grouped_data.index, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Component-Bar Chart of Literacy Rates by Country and Gender':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby(['Country', 'Gender'])['Literacy rate'].sum().unstack()
             grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6), ax=ax)
             ax.set_xlabel('Country')
             ax.set_ylabel('Literacy Rate')
             ax.set_title('Component Bar Chart of Literacy Rate by Country and Gender')
             ax.legend(title='Gender')
             ax.set_xticklabels(grouped_data.index, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Component-Bar Chart of Literacy Rates by Region and Gender':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby(['Region', 'Gender'])['Literacy rate'].sum().unstack()
             grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6), ax=ax)
             ax.set_xlabel('Region')
             ax.set_ylabel('Literacy Rate')
             ax.set_title('Component Bar Chart of Literacy Rate by Region and Gender')
             ax.legend(title='Gender')
             ax.set_xticklabels(grouped_data.index, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Component-Bar Chart of Literacy Rates by Region and Country':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby(['Region', 'Country'])['Literacy rate'].sum().unstack()
             grouped_data.plot(kind='bar', stacked=True, figsize=(10, 6), ax=ax)
             ax.set_xlabel('Region')
             ax.set_ylabel('Literacy Rate')
             ax.set_title('Component Bar Chart of Literacy Rate by Region and Country')
             ax.legend(title='Gender')
             ax.set_xticklabels(grouped_data.index, rotation=90)
             self.canvas.draw()

        if selected_graph == 'Pie Chart of Literacy Rates by Region':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby('Region')['Literacy rate'].mean().reset_index()
             colors = plt.get_cmap('tab20c').colors
             ax.pie(grouped_data['Literacy rate'], labels=grouped_data['Region'], autopct='%1.1f%%', colors=colors, startangle=90)
             ax.axis('equal')
             ax.set_title('Distribution of Literacy Rate by Region')
             self.canvas.draw()

        if selected_graph == 'Pie Chart of Literacy Rates by Country':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby('Country')['Literacy rate'].mean().reset_index()
             colors = plt.get_cmap('tab20c').colors
             ax.pie(grouped_data['Literacy rate'], labels=grouped_data['Country'], autopct='%1.1f%%', colors=colors, startangle=90)
             ax.axis('equal')
             ax.set_title('Distribution of Literacy Rate by Country')
             self.canvas.draw()

        if selected_graph == 'Pie Chart of Literacy Rates by Gender':
             ax = self.figure.add_subplot(111)
             grouped_data = df.groupby('Gender')['Literacy rate'].mean().reset_index()
             colors = plt.get_cmap('tab20c').colors
             ax.pie(grouped_data['Literacy rate'], labels=grouped_data['Gender'], autopct='%1.1f%%', colors=colors, startangle=90)
             ax.axis('equal')
             ax.set_title('Distribution of Literacy Rate by Gender')
             self.canvas.draw()

        if selected_graph == 'Boxplot of Literacy Rates by Region':
             ax = self.figure.add_subplot(111)
             df.boxplot(column='Literacy rate', by='Region', ax=ax)
             ax.set_xlabel('Region')
             ax.set_ylabel('Literacy Rate')
             self.canvas.draw()

        if selected_graph == 'Boxplot of Literacy Rates by Gender':
             ax = self.figure.add_subplot(111)
             df.boxplot(column='Literacy rate', by='Gender', ax=ax)
             ax.set_xlabel('Gender')
             ax.set_ylabel('Literacy Rate')
             ax.grid(True)
             self.canvas.draw()

        if selected_graph == 'Boxplot of Literacy Rates by Country':
             ax = self.figure.add_subplot(111)
             df.boxplot(column='Literacy rate', by='Country', ax=ax)
             ax.set_xlabel('Country')
             ax.set_ylabel('Literacy Rate')
             plt.xticks(rotation=90)
             ax.grid(True)
             self.canvas.draw() 

        if selected_graph == 'Scatter Plot of Literacy Rate vs Year with Linear Regression':
             x = df['Year']
             y = df['Literacy rate']
             slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

             # Plotting scatter plot and linear regression line
             ax = self.figure.add_subplot(111)
             ax.scatter(x, y, alpha=0.5, label='Data')
             ax.plot(x, slope * x + intercept, color='red', label='Linear Regression')

             ax.set_title('Scatter Plot of Literacy Rate vs Year with Linear Regression')
             ax.set_xlabel('Year')
             ax.set_ylabel('Literacy Rate')
             ax.legend()
             ax.grid(True)
             self.canvas.draw()






    def back_to_main(self):
        self.stacked_widget.setCurrentIndex(0)  # Show the main window



class TabularWindow(QWidget):
    def __init__(self, data_frame, stacked_widget):
        super().__init__()

        self.setWindowTitle('Tabular Representation')
        self.setGeometry(200, 200, 600, 400)
        self.setStyleSheet("background-color: lightgoldenrodyellow;")

        self.stacked_widget = stacked_widget

        layout = QVBoxLayout()

        # Create QTableWidget to display tabular data
        self.table = QTableWidget()
        self.populate_table(data_frame)
        layout.addWidget(self.table)

        # Back button to return to the previous window
        self.back_button = QPushButton('Back', self)
        self.back_button.clicked.connect(self.back_to_graph_options)
        layout.addWidget(self.back_button)

        self.setLayout(layout)

    def populate_table(self, data_frame):
        self.table.setRowCount(data_frame.shape[0])
        self.table.setColumnCount(data_frame.shape[1])
        self.table.setHorizontalHeaderLabels(data_frame.columns)

        for i in range(data_frame.shape[0]):
            for j in range(data_frame.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(data_frame.iloc[i, j])))

        self.table.resizeColumnsToContents()
    
    def back_to_graph_options(self):
        self.stacked_widget.setCurrentIndex(0)  # Show the graph selection window

    def populate_table(self, data_frame):
        data_frame['Literacy rate'] = data_frame['Literacy rate'].astype(float)  # Ensure 'Literacy rate' column is numeric

        data_frame = data_frame.sort_values('Literacy rate')

        frequency = data_frame['Literacy rate'].value_counts().sort_index()
        cum_frequency = frequency.cumsum()

        data_frame['Frequency'] = data_frame['Literacy rate'].map(frequency)
        data_frame['Cumulative Frequency'] = data_frame['Literacy rate'].map(cum_frequency)

        self.table.setRowCount(data_frame.shape[0])
        self.table.setColumnCount(data_frame.shape[1])
        self.table.setHorizontalHeaderLabels(data_frame.columns)

        for i in range(data_frame.shape[0]):
            for j in range(data_frame.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(data_frame.iloc[i, j])))

        self.table.resizeColumnsToContents()

    def back_to_main(self):
        self.stacked_widget.setCurrentIndex(0) 

class DescriptiveStatsWindow(QWidget):
    def __init__(self, data):
        super().__init__()
        layout = QVBoxLayout(self)
        self.main_window = None  # Placeholder for the main window reference

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)

        # Calculate Descriptive Statistics
        mean = data['Literacy rate'].mean()
        median = data['Literacy rate'].median()
        data_min = data['Literacy rate'].min()
        data_max = data['Literacy rate'].max()
        data_range = data_max - data_min
        variance = data['Literacy rate'].var()
        std_deviation = data['Literacy rate'].std()
        q1 = data['Literacy rate'].quantile(0.25)
        q3 = data['Literacy rate'].quantile(0.75)
        iqr = q3 - q1
        skewness = data['Literacy rate'].skew()
        frequency_distribution = data['Literacy rate'].value_counts()

        # Measures of Position
        quartiles = np.percentile(data['Literacy rate'], [25, 50, 75])  # Quartiles
        deciles = np.percentile(data['Literacy rate'], np.arange(10, 100, 10))  # Deciles
        percentiles_100 = np.percentile(data['Literacy rate'], np.arange(1, 100, 1))  # Percentiles (1 to 99)
        file_path = '/Users/syedadaniya/Desktop/LR.csv'
        df = pd.read_csv(file_path)
# Assuming 'Year' and 'Literacy rate' columns exist in your DataFrame
        x = df['Year']
        y = df['Literacy rate']
# Calculate the correlation coefficient
        correlation = np.corrcoef(x, y)[0, 1]

       
        # Displaying the calculated values
        text_edit.append(f"Mean: {mean}")
        text_edit.append(f"Median: {median}")
        text_edit.append(f"Range: {data_range}")
        text_edit.append(f"Variance: {variance}")
        text_edit.append(f"Standard Deviation: {std_deviation}")
        text_edit.append(f"First Quartile (Q1): {q1}")
        text_edit.append(f"Third Quartile (Q3): {q3}")
        text_edit.append(f"IQR: {iqr}")
        text_edit.append(f"Skewness: {skewness}")
        text_edit.append(f"Frequency Distribution: \n{frequency_distribution}")
        text_edit.append(f"Quartiles: {quartiles}")
        text_edit.append(f"Deciles: {deciles}")
        text_edit.append(f"Percentiles (1-99): {percentiles_100}")
        text_edit.append(f"Correlation Coefficient between Year and Literacy rate: {correlation}")

# Determine the strength of the correlation
        if correlation < 0.3:
             text_edit.append("The correlation is weak.")
        elif 0.3 <= correlation < 0.7:
             text_edit.append("The correlation is moderate.")
        else:
             text_edit.append("The correlation is strong.")
        layout.addWidget(text_edit)

        # ... (Calculate and display statistics)

        self.back_button = QPushButton('Back', self)
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

    def go_back(self):
        self.close()  # Close the descriptive stats window
        self.main_window.show()  



class UniformDistributionWidget(QWidget):
     def __init__(self):
        super().__init__()
        self.setWindowTitle('Uniform Distribution Calculator')
        self.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout()

        # Create input fields and labels
        self.literacy_rate_input = QLineEdit()
        layout.addWidget(QLabel('Enter the value of Literacy Rate:'))
        layout.addWidget(self.literacy_rate_input)

        self.lower_bound_input = QLineEdit()
        layout.addWidget(QLabel('Enter the lower bound:'))
        layout.addWidget(self.lower_bound_input)

        self.upper_bound_input = QLineEdit()
        layout.addWidget(QLabel('Enter the upper bound:'))
        layout.addWidget(self.upper_bound_input)

        # Button to calculate distribution
        calculate_button = QPushButton('Calculate')
        calculate_button.clicked.connect(self.calculate_uniform_distribution)
        layout.addWidget(calculate_button)

        # Result display
        self.result_text_edit = QTextEdit()
        self.result_text_edit.setReadOnly(True)
        layout.addWidget(self.result_text_edit)

        self.setLayout(layout)

     def calculate_uniform_distribution(self):
        try:
            literacy_rate = float(self.literacy_rate_input.text())
            lower_bound = float(self.lower_bound_input.text())
            upper_bound = float(self.upper_bound_input.text())

            if lower_bound >= upper_bound:
                self.result_text_edit.clear()
                self.result_text_edit.append('Lower bound must be less than upper bound.')
            else:
                range_x = upper_bound - lower_bound
                p_x = 1 / range_x
                e_x = (upper_bound + lower_bound) / 2
                variance_x = ((upper_bound - lower_bound) ** 2) / 12
                std_x = np.sqrt(variance_x)

                self.result_text_edit.clear()
                self.result_text_edit.append(f'For Literacy Rate = {literacy_rate} '
                                            f'and the given range [{lower_bound}, {upper_bound}]:')
                self.result_text_edit.append(f'P(x): {p_x}')
                self.result_text_edit.append(f'E(x): {e_x}')
                self.result_text_edit.append(f'Range: {range_x}')
                self.result_text_edit.append(f'Variance(x): {variance_x}')
                self.result_text_edit.append(f'Std(x): {std_x}')

        except ValueError:
            self.result_text_edit.clear()
            self.result_text_edit.append('Please enter valid numerical values.')
  

        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.go_back)
        layout = QVBoxLayout()
        layout.addWidget(self.back_button)


     def go_back(self):
        self.main_window.stacked_widget.setCurrentIndex(0)



class AutocorrelationWindow(QWidget):
    def __init__(self, df):
        super().__init__()
        self.setWindowTitle('Autocorrelation Function')
        
        layout = QVBoxLayout()

        # Compute and plot autocorrelation function (ACF)
        plt.figure(figsize=(12, 6))
        plot_acf(df['Literacy rate'], lags=20)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.title('Autocorrelation Function (ACF) for Literacy Rate')
        plt.savefig('acf_plot.png')  # Save ACF plot temporarily

        # Display ACF plot in a QLabel
        acf_image = QLabel()
        pixmap = QPixmap("acf_plot.png")
        acf_image.setPixmap(pixmap)
        layout.addWidget(acf_image)

        self.setLayout(layout)

        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)


    def go_back(self):
        self.main_window.stacked_widget.setCurrentIndex(0)
class ConfidenceIntervalWidget(QWidget):
    def __init__(self, data, stacked_widget):
        super().__init__()

        self.data = data
        self.stacked_widget = stacked_widget  # Pass the stacked_widget reference from MainWindow

        self.layout = QVBoxLayout()

        self.calculate_confidence_intervals()
        self.setup_stacked_widget()

        self.setLayout(self.layout)

    def go_back(self):
        self.stacked_widget.setCurrentIndex(0)  # Use the stacked_widget reference to go back

    # rest of your class code remains unchanged


class ConfidenceIntervalWidget(QWidget):
    def __init__(self, data):
        super().__init__()

        self.data = data

        self.layout = QVBoxLayout()

        self.calculate_confidence_intervals()
        self.setup_stacked_widget()

        self.setLayout(self.layout)

    def calculate_confidence_intervals(self):
        mean_literacy = self.data['Literacy rate'].mean()
        std_dev_literacy = self.data['Literacy rate'].std()

        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)

        margin_of_error_mean = z_score * (std_dev_literacy / np.sqrt(len(self.data)))

        self.confidence_interval_mean = (
            mean_literacy - margin_of_error_mean,
            mean_literacy + margin_of_error_mean
        )

        self.confidence_interval_std = (
            std_dev_literacy - margin_of_error_mean,
            std_dev_literacy + margin_of_error_mean
        )

    def setup_stacked_widget(self):
        confidence_widget = QWidget()
        layout = QVBoxLayout()

        label_heading = QLabel('CONFIDENCE INTERVAL FOR DESCRIPTIVE MEASURES')
        layout.addWidget(label_heading)

        label_mean = QLabel(f"Mean confidence interval: {self.confidence_interval_mean}")
        label_std = QLabel(f"Standard deviation confidence interval: {self.confidence_interval_std}")

        layout.addWidget(label_mean)
        layout.addWidget(label_std)

        confidence_widget.setLayout(layout)
        self.layout.addWidget(confidence_widget)

        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)


    def go_back(self):
        self.main_window.stacked_widget.setCurrentIndex(0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        
        # Set up the main window
        self.setWindowTitle('DATA ANALYSIS')
        self.setGeometry(100, 100, 600, 400)
        self.setStyleSheet("background-color: lightgoldenrodyellow;")
        self.stacked_widget = QStackedWidget()  # Create stacked widget
        self.setCentralWidget(self.stacked_widget)
        # Create the main layout
        layout = QVBoxLayout()

        # Add buttons for different options
        button1 = QPushButton('Graphical Representation')
        button1.setFixedSize(700, 50)  
        button1.setStyleSheet("background-color: #A0522D; color: black; font-weight: bold;")
        button1.clicked.connect(self.show_graph_options)
        layout.addWidget(button1)

        button2 = QPushButton('Tabular Representation')
        button2.setFixedSize(700, 50) 
        button2.setStyleSheet("background-color: #A0522D; color: black; font-weight: bold;")
        button2.clicked.connect(self.show_tabular_data)
        layout.addWidget(button2)

        button3 = QPushButton('Descriptive Statistical Measures')
        button3.setFixedSize(700, 50) 
        button3.setStyleSheet("background-color: #A0522D; color: black; font-weight: bold;")
        button3.clicked.connect(self.show_descriptive_stats)
        layout.addWidget(button3)

        
        button4 = QPushButton('Probability Distribution Method')
        button4.clicked.connect(self.show_uniform_distribution_window)
        button4.setFixedSize(700, 50) 
        button4.setStyleSheet("background-color: #A0522D; color: black; font-weight: bold;")
        layout.addWidget(button4)


        button5 = QPushButton('Linear Regression Analysis/Auto-Corelation(Lags Values)')
        button5.setFixedSize(700, 50) 
        button5.setStyleSheet("background-color: #A0522D; color: black; font-weight: bold;")
        button5.clicked.connect(self.show_autocorrelation_screen)
        layout.addWidget(button5)

        button6 = QPushButton('Confidence Interval')
        button6.clicked.connect(self.show_confidence_interval)
        button6.setFixedSize(700, 50) 
        button6.setStyleSheet("background-color: #A0522D; color: black; font-weight: bold;")
        layout.addWidget(button6)

        # Set up the central widget
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.stacked_widget.addWidget(central_widget)

        # Load the dataset (replace 'file_path' with the path to your CSV file)
        file_path = '/Users/syedadaniya/Desktop/LR.csv'
        self.df = pd.read_csv(file_path)

    def show_graph_options(self):
        graph_window = GraphSelectionWindow(self.stacked_widget)
        self.stacked_widget.addWidget(graph_window)
        self.stacked_widget.setCurrentWidget(graph_window)

    def show_tabular_data(self):
        file_path = '/Users/syedadaniya/Desktop/LR.csv'
        df = pd.read_csv(file_path)

        tabular_window = TabularWindow(df, self.stacked_widget)
        self.stacked_widget.addWidget(tabular_window)
        self.stacked_widget.setCurrentWidget(tabular_window)
      
       
    def show_descriptive_stats(self):
        file_path = '/Users/syedadaniya/Desktop/LR.csv'
        df = pd.read_csv(file_path)

        descriptive_stats_window = DescriptiveStatsWindow(df)
        self.stacked_widget.addWidget(descriptive_stats_window)
        self.stacked_widget.setCurrentWidget(descriptive_stats_window)

    def show_uniform_distribution_window(self):
        uniform_distribution_window = UniformDistributionWidget()
        self.stacked_widget.addWidget(uniform_distribution_window)
        self.stacked_widget.setCurrentWidget(uniform_distribution_window)
     
    def show_autocorrelation_screen(self):
        acf_window = AutocorrelationWindow(self.df)
        self.stacked_widget.addWidget(acf_window)
        self.stacked_widget.setCurrentWidget(acf_window)
    

    def show_confidence_interval(self):
        confidence_interval_widget = ConfidenceIntervalWidget(self.df)
        self.stacked_widget.addWidget(confidence_interval_widget)
        self.stacked_widget.setCurrentWidget(confidence_interval_widget)



def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
