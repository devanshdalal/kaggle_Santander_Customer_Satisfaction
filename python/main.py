import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime

# DATA
train_df = pd.read_csv('../data/train.csv', header=0)        # Load the train file into a dataframe
test_df = pd.read_csv('../data/test.csv', header=0)        # Load the test file into a dataframe

# Creating submission file
def create_submission(data, prediction, filename):
    submission = pd.DataFrame({
        "ID": data["ID"],
        "TARGET": prediction
    })

    submission.to_csv(filename, index=False)

combData = train_df.append(test_df)

# -------------------- Feature extraction
print("Pre-processing ...")

SAVE_COMB = False
if SAVE_COMB:
    print("Writing combData.csv ...")
    combData.to_csv("../temp/combData.csv", index=False)

# -------------------- Training
trainData = combData[0:len(train_df)]
testData = combData[len(train_df):len(combData)]

predictColumns = combData.columns.values.tolist()
predictColumns = [ x for x in predictColumns if 'ID' not in x]
predictColumns = [ x for x in predictColumns if 'TARGET' not in x]

print("Training ...")
n_tree = 1000

rfc = RandomForestClassifier(n_estimators=n_tree, verbose=10, n_jobs=-1)
rfc.fit(trainData[predictColumns], trainData["TARGET"])
predictions = rfc.predict_proba(testData[predictColumns])
predictions = predictions[:, 1]

fileName = "../result/p_" + datetime.datetime.now().strftime("%Y_%m%d_%H%M%S") + "_n" + str(n_tree) + ".csv"
create_submission(testData, predictions, fileName)
