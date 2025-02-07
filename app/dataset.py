import os 
import pandas as pd
import re 


def decode_label(filename:str) -> str:
    match = re.search(".*vertebrae_?([A-Z][0-9]).*.stl", filename)
    if match: 
        return match.group(1)
                 
def load_mesh_metadata(dir:str) -> pd.DataFrame: 
    metadata = pd.DataFrame(columns=['filename', 'label'])
    for i, file in enumerate(os.listdir(dir)): 
        if file.endswith('stl'): 
            metadata.loc[i] = {'filename': file, 'label': decode_label(file)}

    return metadata

def sample_meshes(df, n=10):
    
    samples = pd.DataFrame()
    for _class in df['label'].unique(): 
        class_samples = df[df['label'] == _class].sample(n=n)
        
        samples = pd.concat([samples, class_samples])

    return samples
