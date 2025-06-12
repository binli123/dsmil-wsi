import pandas as pd
import glob 
import os

# Read the CSV file
df = pd.read_csv("/home/karan.padariya/CLAM/dataset_csv/tcga-LUAD&LUSC.csv")

# Create dictionary
slide_label_dict = dict(zip(df["slide_id"], df["label"]))

# Print or use the dictionary
# print(slide_label_dict)

csv_list = glob.glob("/ssd_scratch/karan.p/datasets/tcga-LUAD&LUSC/karan.p/*csv")

file_list =  [os.path.basename(x) for x in csv_list]

out_dict = {}

for file in csv_list:
    if slide_label_dict[os.path.basename(file)[:-4]] == "ADENO":
        out_dict[file] = 0
    else:
        out_dict[file] = 1



dict_df = pd.DataFrame(list(out_dict.items()), columns=["0", "label"])
dict_df.to_csv("/ssd_scratch/karan.p/datasets/tcga-LUAD&LUSC/tcga-LUAD&LUSC.csv", index=False)

# import pandas as pd

# file1 = pd.read_csv("/home/karan.padariya/CLAM/dataset_csv/oral_2_class_new.csv")
# file2  = pd.read_csv("/ssd_scratch/karan.p/datasets/Oral_10_20_ORCHID/Oral_10_20.csv")

# file2['slide_id'] = file2['0'].apply(lambda x: x.split('/')[-1].split('.')[0]) # Adjust based on your path format

# print(file2.head())
# merged_file = file2.drop(columns=['label']).merge(file1, on='slide_id', how='left')

# merged_file = merged_file.drop(columns=['slide_id', 'case_id', 'quality'], errors='ignore')

# # Update the label column with 0 for 'stage_1' and 1 for all other labels
# merged_file['label'] = merged_file['label'].apply(lambda x: 0 if x == 'Stage_1' else 1)

# print(merged_file.head())
# merged_file.to_csv("/ssd_scratch/karan.p/updated_file.csv", index=False)

# df_2_class = pd.read_csv("/home/karan.padariya/CLAM/dataset_csv/oral_2_class_5X_10X.csv")
# df_5_20 = pd.read_csv("/ssd_scratch/karan.p/datasets/Oral_5_20/Oral_5_20.csv")

# # Extract the slide_ids from the "0" column of df_5_20
# # Assuming the slide_id is the part after the last underscore before .csv
# df_5_20['slide_id'] = df_5_20['0'].str.extract(r'\/([^\/]+)\.csv$')

# # Filter df_2_class by keeping rows where slide_id is in df_5_20
# filtered_df = df_2_class[df_2_class['slide_id'].isin(df_5_20['slide_id'])]

# print(filtered_df.head())
# filtered_df.to_csv("/home/karan.padariya/CLAM/dataset_csv/oral_2_class_5X_10X_252.csv", index=False)

# # Load your CSV files
# df_oral = pd.read_csv('/home/karan.padariya/CLAM/dataset_csv/oral.csv')
# df_2_class_1vs2 = pd.read_csv('/home/karan.padariya/CLAM/dataset_csv/oral_2_class_new.csv')

# # Count the number of occurrences of each stage in the 'label' column
# stage_counts = df_oral['label'].value_counts()

# # Print the counts for each stage
# print(stage_counts)

# # Identify slide IDs in oral.csv with label 'Stage_3'
# stage_3_slides = df_oral[df_oral['label'] == 'Stage_3']['slide_id']

# # Filter out rows in oral_2_class_1vs2.csv where slide_id is in stage_3_slides
# filtered_df = df_2_class_1vs2[~df_2_class_1vs2['slide_id'].isin(stage_3_slides)]


# stage_counts = filtered_df['label'].value_counts()

# # Print the counts for each stage
# print(stage_counts)
# # Save the filtered data if needed
# filtered_df.to_csv('/home/karan.padariya/CLAM/dataset_csv/filtered_oral_2_class_1vs2.csv', index=False)

# print(filtered_df)
