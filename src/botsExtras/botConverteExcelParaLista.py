import pandas as pd

def excel_to_list(file_path, sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str)
    
    data_list = df.values.tolist()
    
    return data_list

def save_list_to_py_file(data_list, output_file):
    with open(output_file, 'w') as file:
        file.write('cameras = [\n')
        for row in data_list:
            for item in row:
                file.write(f"    '{item}',\n")
        file.write(']\n')

file_path = 'src/assets/Cameras.xlsx'

output_file = 'src/botsExtras/ListaCameras.py'

data_list = excel_to_list(file_path)

save_list_to_py_file(data_list, output_file)

print(f"Lista salva em {output_file}")
