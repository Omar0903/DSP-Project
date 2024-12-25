import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy.signal import resample, butter, filtfilt
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from tkinter import filedialog, messagebox

def PreprocessEcg(EcgSignal, SamplingRate=1000, DownsampleFactor=2):
    Signals = {"original": EcgSignal}

    # Down-sampling
    if DownsampleFactor > 1:
        NewLength = len(EcgSignal) // DownsampleFactor
        EcgSignal = resample(EcgSignal, NewLength)
    Signals["resampled"] = EcgSignal

    # Remove DC component (mean)
    EcgSignal = EcgSignal - np.mean(EcgSignal)

    # Bandpass filter to remove noise (1-40 Hz)
    Lowcut = 1.0
    Highcut = 40.0
    NyquistRate = SamplingRate / 2  # Nyquist frequency
    Low = Lowcut / NyquistRate
    High = Highcut / NyquistRate
    B, A = butter(4, [Low, High], btype="band")
    FilteredSignal = filtfilt(B, A, EcgSignal)
    Signals["filtered"] = FilteredSignal

    # Normalize the signal
    NormalizedSignal = (FilteredSignal - np.min(FilteredSignal)) / (np.max(FilteredSignal) - np.min(FilteredSignal))
    Signals["normalized"] = NormalizedSignal

    return NormalizedSignal, Signals


def SegmentEcgData(EcgSignal, SegmentLength=400):
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
    
    return KnnPredictions, SvmPredictions


def TrainModels():
    TrainingFiles = ["s1.txt", "s2.txt", "s3.txt"]

    X, Y = PrepareTrainingData(TrainingFiles)

    Knn = KNeighborsClassifier(n_neighbors=3)
    Knn.fit(X, Y)

    Svm = SVC(kernel="linear", probability=True)
    Svm.fit(X, Y)

    Tree = DecisionTreeClassifier()
    Tree.fit(X, Y)

    KnnAccuracy = accuracy_score(Y, Knn.predict(X))
    SvmAccuracy = accuracy_score(Y, Svm.predict(X))
    TreeAccuracy = accuracy_score(Y, Tree.predict(X))

    return Knn, Svm, Tree, KnnAccuracy, SvmAccuracy, TreeAccuracy


def ShowAccuracy(KnnAccuracy, SvmAccuracy, TreeAccuracy):
    messagebox.showinfo("Training Accuracy", f"KNN Accuracy: {KnnAccuracy * 100:.2f}%\n"
                                             f"SVM Accuracy: {SvmAccuracy * 100:.2f}%\n"
                                             f"Decision Tree Accuracy: {TreeAccuracy * 100:.2f}%")

def SelectFile():
    FilePath = filedialog.askopenfilename(title="Select ECG File", filetypes=(("Text Files", "*.txt"), ("All Files", "*.*")))
    return FilePath


def StartPrediction(Knn, Svm):
    TestFilePath = SelectFile()

    if not TestFilePath:
        return

    if not os.path.exists(TestFilePath):
        messagebox.showerror("Error", "Test file does not exist. Please provide a valid file path.")
        return

    try:
        KnnPredictions, SvmPredictions = PredictFile(TestFilePath, Knn, Svm)
        
        print("Predictions for each segment:")
        for i, (knn_pred, svm_pred) in enumerate(zip(KnnPredictions, SvmPredictions), 1):
            print(f"Segment {i}: KNN -> s{knn_pred + 1}, SVM -> s{svm_pred + 1}")
        
        KnnPredictionStr = "\n".join([f"Segment {i+1}: s{Label + 1}" for i, Label in enumerate(KnnPredictions)])
        SvmPredictionStr = "\n".join([f"Segment {i+1}: s{Label + 1}" for i, Label in enumerate(SvmPredictions)])
        
        messagebox.showinfo(
            "Prediction Results", 
            f"KNN Predictions:\n{KnnPredictionStr}\n\nSVM Predictions:\n{SvmPredictionStr}"
        )

        with open(TestFilePath, "r") as File:
            SignalData = np.array([float(Value) for Value in File.read().strip().split()])
        
        _, Signals = PreprocessEcg(SignalData)
        
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


if __name__ == "__main__":
    Knn, Svm, Tree, KnnAccuracy, SvmAccuracy, TreeAccuracy = TrainModels()
    ShowAccuracy(KnnAccuracy, SvmAccuracy, TreeAccuracy)
    StartPrediction(Knn, Svm)
