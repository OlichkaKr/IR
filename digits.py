from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc

def recognize():
	digits = datasets.load_digits()
	features = digits.data 
	labels = digits.target

	clf = SVC(gamma = 0.001)
	clf.fit(features, labels)

	for i in range(1,8):
		img = misc.imread('c_' + str(i) + ".png")
		# img = misc.imresize(img, (8,8))
		img = img.astype(digits.images.dtype)
		print(img.dtype)
		img = misc.bytescale(img, high=16, low=0)

		x_test = []

		for eachRow in img:
			# for eachPixel in eachRow:
				x_test.append(sum(eachRow)/3.0)



		print(clf.predict([x_test]))