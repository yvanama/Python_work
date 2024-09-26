#!/usr/bin/env python
# coding: utf-8

# In[6]:


def read_file(name):
    # Importing appropriate packages (.5 point)
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    # Reading the patients.csv file (.5 point)
    patient_df = pd.read_csv('patients.csv')
    index = patient_df['patient']

    M2_affected = [0 for i in range(len(index))]
    M2_index = 0

    for i in index:
        message_a = "Reading participant #" + str(i)
        print(message_a)

        # Reading the affected side of data for participant # (1 point)
        tmp_affected_file = patient_df[patient_df['patient'] == i]['affected'].iloc[0]
        if tmp_affected_file == "left":
            tmp_affected_file = str(i) + "/LOG_Left.csv"
        else:
            tmp_affected_file = str(i) + "/LOG_Right.csv"


        message_b = "- Reading the affected arm"
        print(message_b)

        # Concatenating strings to make the appropriate path + filename (1 point)
        
        tmp_affected_path = f'/Users/yaminivanama/Desktop/Data2/{tmp_affected_file}'   
        affected_df = pd.read_csv(tmp_affected_path, skiprows=6)

        # Extracting appropriate columns (1 point)
        affected_acc_x = affected_df["Accelerometer x"]
        affected_acc_y = affected_df["Accelerometer y"]
        affected_acc_z = affected_df["Accelerometer z"]

        # Computing M2 for left arm (1 point)
        affected_acc_mag = ((affected_acc_x**2 + affected_acc_y**2 + affected_acc_z**2)**0.5) - 9.8
        tmp_affected_M2 = np.mean(affected_acc_mag)
        M2_affected[M2_index] = tmp_affected_M2

        M2_index = M2_index + 1

    print(M2_affected)

    # Producing a scatter plot (1 point)
    therapist_assessed_MAL_scores = patient_df['MAL amount']
    plt.scatter(M2_affected, therapist_assessed_MAL_scores, label="Data points", color='blue')
    plt.xlabel('M2 values')
    plt.ylabel('Therapist-assessed MAL scores')


    # Linear fitting using the therapist-assessed MAL scores and the M2 values (1 point)
    slope, intercept = np.polyfit(M2_affected, therapist_assessed_MAL_scores, 1)


    # Add the fitted line to the produced scatter plot (1 point)
    plt.plot(M2_affected, slope * np.array(M2_affected) + intercept, color="red", label="Fitted line")
    plt.legend()
    plt.title('Scatter plot with Linear Fit')
    plt.show()



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_file('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# In[ ]:





# In[ ]:




