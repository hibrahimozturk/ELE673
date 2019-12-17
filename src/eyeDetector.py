from scipy.io import loadmat
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import cv2
import imutils
from PIL import Image
import scipy.linalg as la
import time


def knn_n_dimensional(eyeImage, nonImage, testEyeImage, testNonImage, K=5):

    print("\t* k-NN Classifier {} dimensional".format(eyeImage.shape[0]))
    startT = time.time()
    trainData = np.concatenate((eyeImage, nonImage), axis=1)
    trainClasses = np.concatenate((np.zeros(eyeImage.shape[1]), np.ones(nonImage.shape[1])))

    testData = np.concatenate((testEyeImage, testNonImage), axis=1)
    testClasses = np.concatenate((np.zeros(testEyeImage.shape[1]), np.ones(testNonImage.shape[1])))

    classifier = KNeighborsClassifier(n_neighbors=K)
    classifier.fit(trainData.transpose(1, 0), trainClasses)
    predictedLabels = classifier.predict(testData.transpose(1, 0))

    accuracy = np.sum(predictedLabels == testClasses)/testClasses.shape[0]
    print("\t\t* Accuracy: {}".format(accuracy))

    fpr = np.sum((predictedLabels != testClasses)[(testClasses == 0)])/np.sum(testClasses == 0)
    print("\t\t* FPR: {}".format(fpr))
    print("\t* Runtime: {}".format(time.time()-startT))
    print()

    return


def classier_reconstruction(testEyeImage, testNonImage,
                            eyeVec, eyeMean, noneyeVec, nonMean, imgSize=[25, 20]):

    print("\t* PCA Classifier based on reconstruction errors")
    startT = time.time()

    testData = np.concatenate((testEyeImage, testNonImage), axis=1)
    testClasses = np.concatenate((np.zeros(testEyeImage.shape[1]), np.ones(testNonImage.shape[1])))

    repeatedEyeMean = np.repeat(eyeMean[:, np.newaxis], testData.shape[1], axis=1)
    repeatedNonMean = np.repeat(nonMean[:, np.newaxis], testData.shape[1], axis=1)

    coeffTestEye = np.matmul((testData - repeatedEyeMean).transpose(), eyeVec.transpose()).transpose()
    coeffTestNonEye = np.matmul((testData-repeatedNonMean).transpose(), noneyeVec.transpose()).transpose()

    reconTestEye = (repeatedEyeMean.transpose() + np.matmul(coeffTestEye.transpose(), eyeVec)).transpose()
    reconTestNon = (repeatedNonMean.transpose() + np.matmul(coeffTestNonEye.transpose(), eyeVec)).transpose()

    # reconTestEyeEx = imutils.rotate_bound(reconTestEye[:, 0].reshape((imgSize[1], imgSize[0])), 90)
    # testEyeEx = imutils.rotate_bound(testData[:, 0].reshape((imgSize[1], imgSize[0])), 90)

    # cv2.imshow("reconTestEyeEx", reconTestEyeEx)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # cv2.imshow("testEyeEx", testEyeEx)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    EyeMSE = np.sum(np.square(testData - reconTestEye), axis=0)
    NonMSE = np.sum(np.square(testData - reconTestNon), axis=0)

    predictedLabels = EyeMSE > NonMSE
    accuracy = np.sum(predictedLabels == testClasses)/testClasses.shape[0]
    print("\t\t* Accuracy: {}".format(accuracy))

    fpr = np.sum((predictedLabels != testClasses)[(testClasses == 0)])/np.sum(testClasses == 0)
    print("\t\t* FPR: {}".format(fpr))
    print("\t* Runtime: {}".format(time.time()-startT))

    print()
    return


def pca_decomp(images, imgSize, pcaComp):
    # ---------- 1-A ----------
    imgAverage = np.mean(images, axis=1)

    rotatedIm = imutils.rotate_bound(imgAverage.reshape((imgSize[1], imgSize[0])), 90)
    # cv2.imshow("eyeAverage", rotatedIm)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    eyeScatter = compute_covariance_matrix(images, imgAverage)
    eigvals, eigvecs = la.eig(eyeScatter)

    # ---------- 1-C ----------
    eigvecs = eigvecs[np.argsort(eigvals)[:pcaComp]]
    for i, eigvec in enumerate(eigvecs):
        normalizedVec = np.interp(eigvec, (eigvec.min(), eigvec.max()), (0, 1))
        rotatedIm = imutils.rotate_bound(normalizedVec.reshape((imgSize[1], imgSize[0])), 90)
        # cv2.imshow("eigvec"+str(i), rotatedIm)
        # cv2.waitKey()
        # cv2.imwrite("norm_eigvec"+str(i)+".png", rotatedIm*255)

    return eigvecs, imgAverage


def compute_covariance_matrix(samples, meanVector):
    # ---------- 1-B ----------
    scatter_matrix = np.zeros((samples.shape[0], samples.shape[0]))
    for i in range(samples.shape[0]):
        scatter_matrix += (samples[:, i].reshape(samples.shape[0], 1) - meanVector).dot(
            (samples[:, i].reshape(samples.shape[0], 1) - meanVector).T)
    return scatter_matrix


if __name__ == "__main__":

    trainSet = loadmat("../data/trainSet.mat")
    testSet = loadmat("../data/testSet.mat")

    eyeImage = trainSet['eyeIm'] / 255.0
    nonImage = trainSet['nonIm'] / 255.0

    testEyeImage = testSet['testEyeIm'] / 255.0
    testNonImage = testSet['testNonIm'] / 255.0

    # ---------- 1-D ----------
    pcaComp = 50
    print("* Number of PCA Components: {}\n".format(pcaComp))
    eyeVec, eyeMean = pca_decomp(eyeImage, trainSet["sizeIm"][0], pcaComp)
    noneyeVec, nonMean = pca_decomp(nonImage, trainSet["sizeIm"][0], pcaComp)

    repeatedEyeMean = np.repeat(eyeMean[:, np.newaxis], eyeImage.shape[1], axis=1)
    repeatedNonMean = np.repeat(nonMean[:, np.newaxis], nonImage.shape[1], axis=1)

    # ---------- 1-F ----------
    coeffTrainEye = np.matmul((eyeImage-repeatedEyeMean).transpose(), eyeVec.transpose()).transpose()
    coeffTrainNonEye = np.matmul((nonImage-repeatedNonMean).transpose(), noneyeVec.transpose()).transpose()

    # ---------- 1-E ----------
    coeffTestEye = np.matmul((testEyeImage-np.repeat(eyeMean[:, np.newaxis], testEyeImage.shape[1], axis=1)).transpose(), eyeVec.transpose()).transpose()
    coeffTestNonEye = np.matmul((testNonImage-np.repeat(nonMean[:, np.newaxis], testNonImage.shape[1], axis=1)).transpose(), noneyeVec.transpose()).transpose()

    # knn_n_dimensional(eyeImage, nonImage, testEyeImage, testNonImage)
    knn_n_dimensional(coeffTrainEye, coeffTrainNonEye, coeffTestEye, coeffTestNonEye)
    classier_reconstruction(testEyeImage, testNonImage,
                            eyeVec, eyeMean, noneyeVec, nonMean)

    print("Finish")

