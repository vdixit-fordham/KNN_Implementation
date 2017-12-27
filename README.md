# KNN_Implementation
Supervised machine learning algorithm, KNN (K-nearest neighbour) Implementation 

Out implementation should accept two data files as input (both are posted with the assignment): a spam train.csv file and a
spam test.csv file. Both files contain exam-ples of e-mail messages, with each example having a class label of either \1" (spam) or
\0" (no-spam). Each example has 57 (numeric) features that characterize the message. Our classifier should examine each example in the spam test set and classify it as one of the two classes. The classification will be based on an unweighted vote of its k nearest examples in the spam train set. We will measure all distances using regular Euclidean distance. <br />

(a) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 without nor-malizing the features.<br />
(b) Report test accuracies when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 with z-score normalization applied to the features.<br />
(c) In the (b) case, generate an output of KNN predicted labels for the first 50 instances (i.e. t1 - t50) when k = 1; 5; 11; 21; 41; 61; 81; 101; 201; 401 (in this order).<br />
For example, if t5 is classified as class "spam" when k = 1; 5; 11; 21; 41; 61 and classified as class "no-spam" when k = 81; 101; 201; 401, then your output line for t5 should be:<br />
t5 spam, spam, spam, spam, spam, spam, no, no, no, no
