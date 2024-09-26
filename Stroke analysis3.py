#!/usr/bin/env python
# coding: utf-8

# ## Assignment 3- Group 7

# ## Item 1: Caluclating M2 and M3 

# In[42]:


## Claculating M3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def compute_beta(sedentary_data):
    mu = np.mean(sedentary_data)
    sigma = np.std(sedentary_data)
    beta = mu + 1.96 * sigma
    return beta

def compute_M3_for_segment(segment_data, beta):
    active_data = segment_data[segment_data > beta]
    M3 = len(active_data) / len(segment_data)
    return M3

def analyze_patient_data(patient_id, affected_side, data_directory):
    affected_filename = f"{data_directory}/{patient_id}/{affected_side}.csv"

    try:
        # Check the content of the 11th row
        eleventh_row = pd.read_csv(affected_filename, header=None, skiprows=10, nrows=1)
        is_header = not is_numeric(eleventh_row.iloc[0, 0])

        if is_header:
            affected_df = pd.read_csv(affected_filename, header=10)
        else:
            affected_df = pd.read_csv(affected_filename, header=None, skiprows=10)
            affected_df.columns = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']

        acc_x = np.array(affected_df['Accelerometer X'])
        acc_y = np.array(affected_df['Accelerometer Y'])
        acc_z = np.array(affected_df['Accelerometer Z'])
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2) - 1
        
        # Use the entire data set for beta computation
        beta = compute_beta(acc_mag)

        # Compute M3 for the entire dataset
        M3 = compute_M3_for_segment(acc_mag, beta)

        return M3

    except FileNotFoundError:
        print(f"File not found: {affected_filename}")
        return None
    except Exception as e:
        print(f"Error reading file {affected_filename}: {e}")
        return None

def process_metadata(metadata_filepath, data_directory):
    metadata_df = pd.read_csv(metadata_filepath)
    all_patient_M3_values = {}

    for _, row in metadata_df.iterrows():
        patient_id = row['Study ID']
        affected_side = 'LUE' if row['AffectedSide'].lower() == 'left' else 'RUE'

        M3_value = analyze_patient_data(patient_id, affected_side, data_directory)
        all_patient_M3_values[patient_id] = M3_value

    return all_patient_M3_values

if __name__ == '__main__':
    metadata_file_path = '/Users/yaminivanama/Desktop/Data/Metadata.csv'
    data_directory = '/Users/yaminivanama/Desktop/Data'
    patient_M3_results = process_metadata(metadata_file_path, data_directory)
    print('M3_scores= ',patient_M3_results)

    


# In[43]:


## Caluclating M2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def read_file(name):
    patient_df = pd.read_csv('Metadata.csv')
    index = patient_df['Study ID']

    M2_affected = [0 for i in range(len(index))]
    M2_index = 0

    for i in index:
        print(f"Reading participant #{i}")

        affected_side = patient_df[patient_df['Study ID'] == i]['AffectedSide'].iloc[0]
        affected_filename = "LUE.csv" if affected_side.lower() == "left" else "RUE.csv"
        affected_filepath = f'/Users/yaminivanama/Desktop/Data/{i}/{affected_filename}'

        print("- Reading the affected arm")

        # Check the content of the 11th row
        eleventh_row = pd.read_csv(affected_filepath, header=None, skiprows=10, nrows=1)
        if is_numeric(eleventh_row.iloc[0, 0]):
            # If the 11th row is numeric, use it as data
            affected_df = pd.read_csv(affected_filepath, header=None, skiprows=10)
            affected_df.columns = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']
        else:
            # If the 11th row is not numeric, use it as header
            affected_df = pd.read_csv(affected_filepath, skiprows=10)

        # Convert columns to numeric, coerce errors to NaN
        affected_df = affected_df.apply(pd.to_numeric, errors='coerce')

        # Calculate M2
        affected_acc_mag = np.sqrt(affected_df['Accelerometer X']**2 + affected_df['Accelerometer Y']**2 + affected_df['Accelerometer Z']**2) - 1
        tmp_affected_M2 = np.nanmean(affected_acc_mag)  # Use nanmean to ignore NaN values
        M2_affected[M2_index] = tmp_affected_M2

        M2_index = M2_index + 1
    print(M2_affected)
        



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    read_file('PyCharm')



# ## ITEM 2 : Drawing a plot or plots of the above M3 (plot 1) – 1.5 point

# In[44]:



import matplotlib.pyplot as plt

# Given M3 scores from the previous output

patient_M3_results = process_metadata(metadata_file_path, data_directory)

m3_scores = patient_M3_results
# Create a list of patient IDs (x values) and a list of corresponding M3 scores (y values)
patient_ids = list(m3_scores.keys())
m3_values = list(m3_scores.values())

# Determine the number of bars (i.e., number of patients)
n_bars = len(m3_scores)

# Create the bar plot
plt.figure(figsize=(15, 5)) # Adjust the figure size as needed
plt.bar(range(n_bars), m3_values, color='skyblue')

# Set the position and labels for x-ticks
plt.xticks(range(n_bars), [str(id) for id in patient_ids], rotation=45) # Rotate if there are many labels

# Add title and labels
plt.title('M3 Scores for Each Patient')
plt.xlabel('Patient ID')
plt.ylabel('M3 Score')

# Optional: Add a grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.tight_layout()
plt.show()


# ## ITEM 3 : Computing M2 and M3 for each hour

# In[45]:


#Claculation M2 for each hour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def calculate_hourly_M2(affected_df):
    data_frequency = 30  # Assuming data is recorded at 30 Hz
    rows_per_hour = 60 * 60 * data_frequency
    hourly_M2 = []

    for start in range(0, len(affected_df), rows_per_hour):
        end = start + rows_per_hour
        hourly_data = affected_df[start:end]
        if not hourly_data.empty:
            acc_mag = np.sqrt(hourly_data['Accelerometer X']**2 + hourly_data['Accelerometer Y']**2 + hourly_data['Accelerometer Z']**2) - 1
            M2_value = np.nanmean(acc_mag)  # Calculate M2 for this hour
            hourly_M2.append(M2_value)

    return hourly_M2

def read_file(name):
    patient_df = pd.read_csv('Metadata.csv')
    index = patient_df['Study ID']

    hourly_M2_results = {}

    for i in index:
        print(f"Reading participant #{i}")

        affected_side = patient_df[patient_df['Study ID'] == i]['AffectedSide'].iloc[0]
        affected_filename = "LUE.csv" if affected_side.lower() == "left" else "RUE.csv"
        affected_filepath = f'/Users/yaminivanama/Desktop/Data/{i}/{affected_filename}'

        print("- Reading the affected arm")

        # Check the content of the 11th row and read data
        eleventh_row = pd.read_csv(affected_filepath, header=None, skiprows=10, nrows=1)
        if is_numeric(eleventh_row.iloc[0, 0]):
            affected_df = pd.read_csv(affected_filepath, header=None, skiprows=10)
            affected_df.columns = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']
        else:
            affected_df = pd.read_csv(affected_filepath, skiprows=10)

        affected_df = affected_df.apply(pd.to_numeric, errors='coerce')

        # Calculate hourly M2 values
        hourly_M2_results[i] = calculate_hourly_M2(affected_df)

    return hourly_M2_results

if __name__ == '__main__':
    hourly_M2_data = read_file('PyCharm')
    print(hourly_M2_data)  # Print or process the hourly M2 results as needed



# In[47]:


#calculating M3 for each hour
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def is_numeric(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def compute_beta(sedentary_data):
    mu = np.mean(sedentary_data)
    sigma = np.std(sedentary_data)
    beta = mu + 1.96 * sigma
    return beta

def compute_M3_for_segment(segment_data, beta):
    active_data = segment_data[segment_data > beta]
    M3 = len(active_data) / len(segment_data)
    return M3

def analyze_patient_data(patient_id, affected_side, data_directory, start_indices, end_indices):
    affected_filename = f"{data_directory}/{patient_id}/{affected_side}.csv"

    try:
        # Check the content of the 11th row
        eleventh_row = pd.read_csv(affected_filename, header=None, skiprows=10, nrows=1)
        is_header = not is_numeric(eleventh_row.iloc[0, 0])

        if is_header:
            affected_df = pd.read_csv(affected_filename, header=10)
        else:
            affected_df = pd.read_csv(affected_filename, header=None, skiprows=10)
            affected_df.columns = ['Accelerometer X', 'Accelerometer Y', 'Accelerometer Z']

        acc_x = np.array(affected_df['Accelerometer X'])
        acc_y = np.array(affected_df['Accelerometer Y'])
        acc_z = np.array(affected_df['Accelerometer Z'])
        acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2) - 1
        
        beta = compute_beta(acc_mag[start_indices[0]:end_indices[0]])

        total_rows = len(affected_df)
        total_seconds = total_rows / 30  # since data is collected at 30 Hz
        total_hours = int(total_seconds / 3600)

        M3_values = []

        # Compute M3 for each full hour
        for hour in range(min(total_hours, len(start_indices))):
            start = start_indices[hour]
            end = end_indices[hour]
            segment_data = acc_mag[start:end]
            M3 = compute_M3_for_segment(segment_data, beta)
            M3_values.append(M3)

        # Check for and compute M3 for the last partial hour
        remaining_seconds = total_seconds - (total_hours * 3600)
        if remaining_seconds > 0:
            start = total_hours * 30 * 60 * 60
            end = total_rows
            segment_data = acc_mag[start:end]
            M3_partial_hour = compute_M3_for_segment(segment_data, beta)
            M3_values.append(M3_partial_hour)

        return M3_values

    except FileNotFoundError:
        print(f"File not found: {affected_filename}")
        return []
    except Exception as e:
        print(f"Error reading file {affected_filename}: {e}")
        return []

def process_metadata(metadata_filepath, data_directory):
    metadata_df = pd.read_csv(metadata_filepath)
    all_patient_M3_values = {}

    for _, row in metadata_df.iterrows():
        patient_id = row['Study ID']
        affected_side = 'LUE' if row['AffectedSide'].lower() == 'left' else 'RUE'

        start_indices = [i * 30 * 60 * 60 for i in range(24)]  # for 24 hours
        end_indices = [(i + 1) * 30 * 60 * 60 for i in range(24)]  # for 24 hours

        M3_values = analyze_patient_data(patient_id, affected_side, data_directory, start_indices, end_indices)
        all_patient_M3_values[patient_id] = M3_values

    return all_patient_M3_values
print(patient_M3_results)

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
for patient_id, M3_values in patient_M3_results.items():
    plt.plot(M3_values, label=f'Patient {patient_id}')
    plt.xlabel('Hourly Segments')
    plt.ylabel('M3 Value')
    plt.title('M3 Values Over Time for Each Patient')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    metadata_file_path = '/Users/yaminivanama/Desktop/Data/Metadata.csv'  # Update this path as needed
    data_directory = '/Users/yaminivanama/Desktop/Data'
    patient_M3_results = process_metadata(metadata_file_path, data_directory)
    


# ## Item 4 : Drawing a plot or plots of the above M2s and M3s (Plot2)
# 

# In[49]:


for patient_id in hourly_M2_data:
    plt.figure(figsize=(10, 4))
    M2_values = hourly_M2_data[patient_id]
    M3_values = patient_M3_results[patient_id]
    plt.plot(range(len(M2_values)), M2_values, label='M2')
    plt.plot(range(len(M3_values)), M3_values, label='M3')
    plt.title(f'Patient {patient_id}')
    plt.xlabel('Hour')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

