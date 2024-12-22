import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from functions import PreprocessEcg, TrainModels, PredictFile, PrepareTrainingData

# Create the main Tkinter window
root = tk.Tk()
root.title("ECG Signal Prediction")
root.geometry("1920x1280")
root.config(background='navajowhite')


# Create some GUI elements
labelWidget = tk.Label(root, text="Select ECG File:")
labelWidget.place(x=910, y=20)

entryFilePath = tk.Entry(root, width=50)
entryFilePath.place(x=770, y=60)

btnSelectFile = tk.Button(root, text="Browse", command=lambda: entryFilePath.insert(0, filedialog.askopenfilename(title="Select ECG File", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))))
btnSelectFile.place(x=915, y=100)

# Create figures (empty initially, will update later)
fig1 = plt.Figure(figsize=(4, 4), dpi=100)
ax1 = fig1.add_subplot(111)
ax1.set_title("Original ECG Signal")
ax1.set_xlabel("Sample index")
ax1.set_ylabel("Amplitude")

fig2 = plt.Figure(figsize=(4, 4), dpi=100)
ax2 = fig2.add_subplot(111)
ax2.set_title("Resampled ECG Signal")
ax2.set_xlabel("Sample index")
ax2.set_ylabel("Amplitude")

fig3 = plt.Figure(figsize=(4, 4), dpi=100)
ax3 = fig3.add_subplot(111)
ax3.set_title("Filtered ECG Signal")
ax3.set_xlabel("Sample index")
ax3.set_ylabel("Amplitude")

fig4 = plt.Figure(figsize=(4, 4), dpi=100)
ax4 = fig4.add_subplot(111)
ax4.set_title("Normalized ECG Signal")
ax4.set_xlabel("Sample index")
ax4.set_ylabel("Amplitude")

# Integrate the figures with Tkinter window using FigureCanvasTkAgg
canvas1 = FigureCanvasTkAgg(fig1, master=root)
canvas1.get_tk_widget().place(x=10, y=500)  # Positioning the first plot
canvas1.draw()

canvas2 = FigureCanvasTkAgg(fig2, master=root)
canvas2.get_tk_widget().place(x=420, y=500)  # Positioning the second plot
canvas2.draw()

canvas3 = FigureCanvasTkAgg(fig3, master=root)
canvas3.get_tk_widget().place(x=830, y=500)  # Positioning the third plot
canvas3.draw()

canvas4 = FigureCanvasTkAgg(fig4, master=root)
canvas4.get_tk_widget().place(x=1240, y=500)  # Positioning the fourth plot
canvas4.draw()

# Initialize KNN and SVM models and their accuracies
knn, svm, knnAccuracy, svmAccuracy = TrainModels()

# Function for "Start Prediction" button (performs prediction only)
def StartPrediction():
    # Get the file path from entryFilePath
    filePath = entryFilePath.get().strip()

    if not filePath:
        messagebox.showerror("Error", "Please select a valid ECG file.")
        return
    
    try:
        # Use PredictFile function for prediction
        knnPredictedLabel, svmPredictedLabel = PredictFile(filePath, knn, svm)
        
        # Predicted class labels for KNN and SVM
        predictedClassKnn = f"s{knnPredictedLabel + 1}"
        predictedClassSvm = f"s{svmPredictedLabel + 1}"
        
        # Determine the training file most similar to the test file based on prediction
        if knnPredictedLabel == 0:
            trainingFile = "s1.txt"
        elif knnPredictedLabel == 1:
            trainingFile = "s2.txt"
        else:
            trainingFile = "s3.txt"
        
        # Show message box with prediction result and training file match
        messagebox.showinfo("Prediction Results", 
                            f"Predicted by KNN: {predictedClassKnn}\nPredicted by SVM: {predictedClassSvm}\n"
                            f"Test file is most similar to: {trainingFile}")
        
        # Update plots after prediction
        UpdatePlots(filePath)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during processing: {e}")

# Function to update plots after prediction
def UpdatePlots(filePath):
    try:
        # Load the signal data from the selected file
        with open(filePath, "r") as file:
            signalData = [float(value) for value in file.read().strip().split()]
        
        # Preprocess the signal
        normalizedSignal, signals = PreprocessEcg(signalData)
        
        # Update the plots
        ax1.clear()
        ax1.plot(signalData)
        ax1.set_title("Original ECG Signal")
        ax1.set_xlabel("Sample index")
        ax1.set_ylabel("Amplitude")
        
        ax2.clear()
        ax2.plot(signals["resampled"])
        ax2.set_title("Resampled ECG Signal")
        ax2.set_xlabel("Sample index")
        ax2.set_ylabel("Amplitude")
        
        ax3.clear()
        ax3.plot(signals["filtered"])
        ax3.set_title("Filtered ECG Signal")
        ax3.set_xlabel("Sample index")
        ax3.set_ylabel("Amplitude")
        
        ax4.clear()
        ax4.plot(signals["normalized"])
        ax4.set_title("Normalized ECG Signal")
        ax4.set_xlabel("Sample index")
        ax4.set_ylabel("Amplitude")
        
        # Redraw the plots
        canvas1.draw()
        canvas2.draw()
        canvas3.draw()
        canvas4.draw()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during processing: {e}")

# Function to show training accuracy for KNN and SVM
def ShowAccuracy():
    try:
        # Show training accuracy for both models
        messagebox.showinfo("Training Accuracy", 
                            f"KNN Training Accuracy: {knnAccuracy * 100:.2f}%\n"
                            f"SVM Training Accuracy: {svmAccuracy * 100:.2f}%")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Button to start prediction (performs only the prediction)
btnPredict = tk.Button(root, text="Start Prediction", command=StartPrediction)
btnPredict.place(x=780, y=200)

# Button to show training accuracy
btnShowAccuracy = tk.Button(root, text="Show Accuracy", command=ShowAccuracy)
btnShowAccuracy.place(x=980, y=200)

# Run Tkinter window
root.mainloop()
