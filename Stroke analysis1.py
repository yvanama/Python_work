#!/usr/bin/env python
# coding: utf-8

# In[57]:


def analyze_data():
    # Use a breakpoint in the code line below to debug your script.
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # (Step 1) Selecting the one data for further data analysis. Give the path & file name below. - 1 point
    affected_filename = f'/Users/yaminivanama/Downloads/data/2966/RUE/NEO1B41100012 (2011-04-26)RAW.csv'
    affected_df = pd.read_csv(affected_filename, skiprows=10)

    acc_x = np.array(affected_df['Accelerometer X'])
    acc_y = np.array(affected_df['Accelerometer Y'])
    acc_z = np.array(affected_df['Accelerometer Z'])
    acc_mag = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z) - 1

    plt.plot(range(len(acc_mag)), acc_mag)
    plt.show()

    # (Step 3) Identifying the data for sedentary behavior (i.e., sleeping) - 1 point
    # If you identify the beginning index of the sedentary behavior, the ending index will be automatically done.
    # If you select the beginning index too loosely, the subsequent sedentary_data may include active behavior.
    # So, be careful.
    start_index = int(1.2e6)
    end_index = start_index + int(0.500e6)

    sedentary_range = range(start_index, end_index)
    sedentary_data = acc_mag[sedentary_range]

    plt.plot(range(len(sedentary_data)), sedentary_data)
    plt.show()

    # (Step 5) Computing beta - 1 point
    # Computing mu
    mu = np.mean(sedentary_data)
    print('mean = ',mu)
    # Computing std
    sigma = np.std(sedentary_data)
    print('SD = ',sigma)
    # Then, computing beta, using mu and std
    beta = mu + 1.96*sigma
    print('Beta= ',beta)

    # (Step 6) Computing M3 for the data - 1 point
    # You can apply beta to identify the data of active behavior.
    active_data = acc_mag[acc_mag > beta]
    M3 = len(active_data)/len(acc_mag)
    print('M3= ',M3)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    analyze_data()


# In[ ]:


# Time Series plots used to analyse all the data folders.


# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
affected_filename = f'/Users/yaminivanama/Downloads/data/2966/RUE/NEO1B41100012 (2011-04-26)RAW.csv'
affected_df = pd.read_csv(affected_filename, skiprows=10, dtype={0: 'str', 1: 'str', 2: 'str'})

#acc_x = pd.to_numeric(affected_df['Accelerometer X'], errors='coerce')
#acc_y = pd.to_numeric(affected_df['Accelerometer Y'], errors='coerce')
#acc_z = pd.to_numeric(affected_df['Accelerometer Z'], errors='coerce')

acc_x = pd.to_numeric(affected_df['Accelerometer X'])
acc_y = pd.to_numeric(affected_df['Accelerometer Y'])
acc_z = pd.to_numeric(affected_df['Accelerometer Z'])

acc_mag = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z) - 1

plt.plot(range(len(acc_mag)), acc_mag)
plt.show()


# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
affected_filename = f'/Users/yaminivanama/Downloads/Data/7177/LUE/NEO1D25110047 (2012-05-21)RAW.csv'
affected_df = pd.read_csv(affected_filename, skiprows=10)


acc_x = np.array(affected_df['-0.997'])
acc_y = np.array(affected_df['-0.073'])
acc_z = np.array(affected_df['0.103'])
acc_mag = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z) - 1

plt.plot(range(len(acc_mag)), acc_mag)
plt.show()


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
affected_filename = f'/Users/yaminivanama/Downloads/Data/11770/RUE/NEO1C16110292 (2011-07-07)RAW.csv'
affected_df = pd.read_csv(affected_filename, skiprows=10)

acc_x = np.array(affected_df['Accelerometer X'])
acc_y = np.array(affected_df['Accelerometer Y'])
acc_z = np.array(affected_df['Accelerometer Z'])
acc_mag = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z) - 1


plt.plot(range(len(acc_mag)), acc_mag)
plt.show()

