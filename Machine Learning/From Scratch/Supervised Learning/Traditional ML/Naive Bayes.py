import numpy as np

# Naive Bayes spam classifier
# data format: 55 columns, first column is 1 for spam, 0 for ham. Other 54 are features (such as word presence), 1 for present, 0 for not present. 

class SpamClassifier:
    def __init__(self, k):
        self.k = k
        
    def train(self, data):
        y = data[:, 0]
        X = data[:, 1:]
        
        n_samples, n_features = X.shape 

        self.p_spam = np.mean(y == 1)
        self.p_ham  = np.mean(y == 0)

        X_spam = X[y == 1]
        X_ham = X[y == 0]

        k = self.k

        #  P(feature=1 | spam)
        self.p_feat_spam = ( X_spam.sum(axis=0) + k ) / (X_spam.shape[0] + 2 * k)

        #  P(feature=1 | ham)
        self.p_feat_ham = ( X_ham.sum(axis=0) + k ) / (X_ham.shape[0] + 2 * k)



        # calculate log prob to stop underflow
        self.log_spam_prior = np.log(self.p_spam)
        self.log_ham_prior  = np.log(self.p_ham)

        self.log_p_spam = np.log(self.p_feat_spam)
        self.log_p_ham  = np.log(self.p_feat_ham) 

        self.log_not_spam = np.log(1 - self.p_feat_spam)
        self.log_not_ham  = np.log(1 - self.p_feat_ham)

        
        
    def predict(self, data):

        X = data

        spam_scores = (
            self.log_spam_prior + X @ self.log_p_spam + (1- X) @ self.log_not_spam
        )

        ham_scores = (self.log_ham_prior + X @ self.log_p_ham + (1 - X) @ self.log_not_ham)

        return (spam_scores > ham_scores).astype(int)
    

def create_classifier():
    classifier = SpamClassifier(k=1)
    classifier.train(training_spam)
    return classifier

classifier = create_classifier()
