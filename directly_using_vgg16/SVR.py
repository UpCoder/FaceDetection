from sklearn.svm import SVR, LinearSVR


class MySVR:
    def __init__(self, train_images, train_labels, train_scores, validation_images, validation_labels, validation_scores):
        self.train_images = train_images
        self.train_labels = train_labels
        self.train_scores = train_scores
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.validation_scores = validation_scores

    def do_linear_svr(self):
        regr = LinearSVR(random_state=0)
        regr.fit(self.train_images, self.train_scores)
        predicted = regr.predict(self.validation_images)
        for index, item in enumerate(self.validation_scores):
            print item, ' : ', predicted[index]

    def do_predict(self, test_features):
        regr = LinearSVR(random_state=0)
        regr.fit(self.train_images, self.train_scores)
        return regr.predict(test_features)
