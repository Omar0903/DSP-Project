import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.signal import decimate, butter, filtfilt
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox

def PreprocessEcg(EcgSignal, SamplingRate=1000, DownsampleFactor=2):
    Signals = {"original": EcgSignal}

    # Down-sampling
    if DownsampleFactor > 1:
        EcgSignal = decimate(EcgSignal, DownsampleFactor)
    Signals["resampled"] = EcgSignal

    # Remove DC component (mean)
    EcgSignal = EcgSignal - np.mean(EcgSignal)

    # Bandpass filter to remove noise (1-40 Hz)
    Lowcut = 1.0
    Highcut = 40.0
    Low = Lowcut / (SamplingRate)
    High = Highcut / (SamplingRate)
    B, A = butter(4, [Low, High], btype="band")
    FilteredSignal = filtfilt(B, A, EcgSignal)
    Signals["filtered"] = FilteredSignal

    # Normalize the signal
    NormalizedSignal = (FilteredSignal - np.min(FilteredSignal)) / (np.max(FilteredSignal) - np.min(FilteredSignal))
    Signals["normalized"] = NormalizedSignal

    return NormalizedSignal, Signals

def SegmentEcgData(EcgSignal, SegmentLength=500):
    NumSegments = len(EcgSignal) // SegmentLength
    return [EcgSignal[i * SegmentLength:(i + 1) * SegmentLength] for i in range(NumSegments)]

def ExtractFeatures(EcgSegment):
    Autocorr = np.correlate(EcgSegment, EcgSegment, mode='full')
    Autocorr = Autocorr[Autocorr.size // 2:]
    DctCoeffs = dct(Autocorr, norm='ortho')
    return DctCoeffs[:10]

def PrepareTrainingData(FilePaths):
    Features = []
    Labels = []
    for Idx, FilePath in enumerate(FilePaths):
        with open(FilePath, "r") as File:
            SignalData = np.array([float(Value) for Value in File.read().strip().split()])
        PreprocessedSignal, _ = PreprocessEcg(SignalData)
        Segments = SegmentEcgData(PreprocessedSignal)

        for Segment in Segments:
            FeatureVector = ExtractFeatures(Segment)
            Features.append(FeatureVector)
            Labels.append(Idx)
    return np.array(Features), np.array(Labels)

def PredictFile(FilePath, KnnModel, SvmModel):
    with open(FilePath, "r") as File:
        SignalData = np.array([float(Value) for Value in File.read().strip().split()])
    PreprocessedSignal, _ = PreprocessEcg(SignalData)
    Segments = SegmentEcgData(PreprocessedSignal)
    TestFeatures = [ExtractFeatures(Segment) for Segment in Segments]
    KnnPredictions = KnnModel.predict(TestFeatures)
    SvmPredictions = SvmModel.predict(TestFeatures)
    KnnMajorityLabel = np.argmax(np.bincount(KnnPredictions))
    SvmMajorityLabel = np.argmax(np.bincount(SvmPredictions))
    return KnnMajorityLabel, SvmMajorityLabel

def TrainModels():
    # Define training files
    TrainingFiles = ["s1.txt", "s2.txt", "s3.txt"]

    # Prepare training data
    X, Y = PrepareTrainingData(TrainingFiles)

    # Train KNN classifier
    Knn = KNeighborsClassifier(n_neighbors=3)
    Knn.fit(X, Y)

    # Train SVM classifier
    Svm = SVC(kernel="linear", probability=True)
    Svm.fit(X, Y)

    # Calculate and display accuracy for both models
    KnnAccuracy = accuracy_score(Y, Knn.predict(X))
    SvmAccuracy = accuracy_score(Y, Svm.predict(X))

    return Knn, Svm, KnnAccuracy, SvmAccuracy

def ShowAccuracy(KnnAccuracy, SvmAccuracy):
    messagebox.showinfo("Successful", f"KNN Training Accuracy: {KnnAccuracy * 100:.2f}%\nSVM Training Accuracy: {SvmAccuracy * 100:.2f}%")

def SelectFile():
    # Open a file dialog for the user to select a test ECG file
    FilePath = filedialog.askopenfilename(title="Select ECG File", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
    return FilePath

def StartPrediction(Knn, Svm):
    TestFilePath = SelectFile()

    if not TestFilePath:
        return  # Exit if no file was selected

    if not os.path.exists(TestFilePath):
        messagebox.showerror("Error", "Test file does not exist. Please provide a valid file path.")
        return

    try:
        # Predict the file label using both models
        KnnPredictedLabel, SvmPredictedLabel = PredictFile(TestFilePath, Knn, Svm)
        
        # Display both predictions
        messagebox.showinfo("Prediction Results", f"Predicted by KNN: s{KnnPredictedLabel + 1}\nPredicted by SVM: s{SvmPredictedLabel + 1}")
        
        # Load and preprocess the test signal for plotting
        with open(TestFilePath, "r") as File:
            SignalData = np.array([float(Value) for Value in File.read().strip().split()])
        
        # Preprocess the signal and retrieve the different stages
        _, Signals = PreprocessEcg(SignalData)
        
        # Plot the signals
        plt.figure()
        plt.plot(SignalData)
        plt.title("Original ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

        plt.figure()
        plt.plot(Signals["resampled"])
        plt.title("Resampled ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

        plt.figure()
        plt.plot(Signals["filtered"])
        plt.title("Filtered ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

        plt.figure()
        plt.plot(Signals["normalized"])
        plt.title("Normalized ECG Signal")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during processing: {e}")
