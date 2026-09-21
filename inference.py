import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score

SUBMISSION="submission.csv"
THRESHOLD=0.404

def main():
    df=pd.read_csv(SUBMISSION)

    assert list(df.columns)==["image_id","label"]
    assert len(df)==2000
    assert df.image_id.nunique()==2000
    assert df.label.notna().all()
    assert set(df.label.unique()).issubset({0,1})

    print("=== PAREIDOLIA PARADOX FINAL INFERENCE CHECK ===")
    print("Submission:",SUBMISSION)
    print("Rows:",len(df))
    print("Unique IDs:",df.image_id.nunique())
    print("Class 0:",int((df.label==0).sum()))
    print("Class 1:",int((df.label==1).sum()))
    print("Null labels:",int(df.label.isna().sum()))
    print("Threshold:",THRESHOLD)
    print("Status: READY FOR SUBMISSION")

if __name__=="__main__":
    main()
