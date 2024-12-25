from functions import *
import numpy as np
import os
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox, Tk, Label, Button, Entry


# Add function to display predictions in Tkinter window
def DisplayPredictedFile(knnPredictions):
    # Count the most frequent prediction for each segment
    counts = np.bincount(knnPredictions)

    # Majority class from KNN predictions
    majorityClass = np.argmax(counts)

    fileMap = {0: "s1", 1: "s2", 2: "s3"}
    predictedFile = fileMap.get(majorityClass, "Unknown")

    return predictedFile


# Modify the StartPrediction function to integrate the DisplayPredictedFile with KNN only
def StartPrediction(knn, svm, filePath):
    if not filePath:
        messagebox.showerror("Error", "Please provide a valid file path.")
        return

    if not os.path.exists(filePath):
        messagebox.showerror(
            "Error", "Test file does not exist. Please provide a valid file path."
        )
        return

    try:
        # Use only KNN predictions to determine the majority class
        knnPredictions, _ = PredictFile(filePath, knn, svm)

        print("Predictions for each segment (KNN):")
        for i, knnPred in enumerate(knnPredictions, 1):
            print(f"Segment {i}: KNN -> s{knnPred + 1}")

        # Display the predicted file based on majority prediction from KNN
        predictedFile = DisplayPredictedFile(knnPredictions)
        print(f"\nThe ECG signal is most likely from {predictedFile}.\n")

        # Show the message box with the prediction result
        messagebox.showinfo(
            "Prediction Result", f"The ECG signal is most likely from {predictedFile}."
        )

        with open(filePath, "r") as file:
            signalData = np.array(
                [float(value) for value in file.read().strip().split()]
            )

        _, signals = PreprocessEcg(signalData)

        # Plot the figures with larger size
        plt.figure(figsize=(10, 6))
        plt.plot(signalData)
        plt.title("Original ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(signals["resampled"])
        plt.title("Resampled ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(signals["filtered"])
        plt.title("Filtered ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(signals["normalized"])
        plt.title("Normalized ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during processing: {e}")


# Function to show accuracy of the models
def ShowAccuracy(knnAccuracy, svmAccuracy, treeAccuracy):
    messagebox.showinfo(
        "Training Accuracy",
        f"KNN Accuracy: {knnAccuracy * 100:.2f}%\n"
        f"SVM Accuracy: {svmAccuracy * 100:.2f}%\n"
        f"Decision Tree Accuracy: {treeAccuracy * 100:.2f}%",
    )


# Function to select file using Entry
def BrowseFile(entryWidget):
    filePath = filedialog.askopenfilename(
        title="Select ECG File",
        filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")),
    )
    if filePath:
        entryWidget.delete(0, "end")  # Clear previous file path
        entryWidget.insert(
            0, filePath
        )  # Insert the new file path into the Entry widget


# GUI Initialization
def Main():
    root = Tk()
    root.title("ECG Signal Prediction")
    root.geometry("500x500")
    root.config(background="navajowhite")

    # Label for Entry Widget
    Label(root, text="Enter ECG File Path:").pack(pady=5)

    # Entry Widget for file path
    filePathEntry = Entry(root, width=50)
    filePathEntry.pack(pady=5)

    # Button for Browse File
    browseButton = Button(
        root, text="Browse File", command=lambda: BrowseFile(filePathEntry)
    )
    browseButton.pack(pady=5)

    # Button for Show Accuracy
    accuracyButton = Button(
        root,
        text="Show Accuracy",
        command=lambda: ShowAccuracy(knnAccuracy, svmAccuracy, treeAccuracy),
    )
    accuracyButton.place(x=280,y=350)

    # Button for Start Prediction
    startButton = Button(
        root,
        text="Start Prediction",
        command=lambda: StartPrediction(knn, svm, filePathEntry.get()),
    )
    startButton.place(x=400,y=350)

    # Train models and get accuracies
    knn, svm, tree, knnAccuracy, svmAccuracy, treeAccuracy = TrainModels()

    root.mainloop()


if __name__ == "__main__":
    Main()
