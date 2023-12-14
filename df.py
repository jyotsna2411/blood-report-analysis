import pandas as pd

df_columns=['White Blood Cell Count (WBC)', 'Lymphocyte Percentage',
       'Monocyte Percentage', 'Neutrophil Percentage', 'Eosinophil Percentage',
       'Basophil Percentage', 'Lymphocyte Count', 'Monocyte Count',
       'Neutrophil Count', 'Eosinophil Count', 'Basophil Count',
       'Red Blood Cell Count (RBC)', 'Hemoglobin Level', 'Hematocrit Level',
       'Mean Corpuscular Volume (MCV)', 'Mean Corpuscular Hemoglobin (MCH)',
       'Mean Cell Hemoglobin Concentration',
       'Red Cell Distribution Width (RDW)', 'Platelet Count',
       'Mean Platelet Volume (MPV)']

df= pd.DataFrame(columns=df_columns)
df.to_csv('df')
print(df)