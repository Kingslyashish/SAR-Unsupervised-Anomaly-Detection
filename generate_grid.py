import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

print("--- Compiling Official DARTIS Visual Grid ---")

tab_file = 'DARTIS_2019.tab'
img_dir = 'dataset_DARTIS_2019/subset_images'

if os.path.exists(tab_file) and os.path.exists(img_dir):
    df = pd.read_csv(tab_file, sep='\t', skiprows=49)
    subset_col = df.columns[0]
    img_col = df.columns[1]
    
    subsets = {
        'Open Water: True Oil Spill (ow)': df[df[subset_col] == 'ow'].iloc[0][img_col],
        'Open Water: Natural Look-Alike (nw)': df[df[subset_col] == 'nw'].iloc[0][img_col],
        'Coastal: True Oil Spill (oc)': df[df[subset_col] == 'oc'].iloc[0][img_col],
        'Coastal: Natural Look-Alike (nc)': df[df[subset_col] == 'nc'].iloc[0][img_col]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()
    
    for idx, (title, filename) in enumerate(subsets.items()):
        ax = axes[idx]
        target_path = os.path.join(img_dir, filename)
        
        if os.path.exists(target_path):
            img = Image.open(target_path)
            ax.imshow(img, cmap='gray')
        else:
            ax.text(0.5, 0.5, f"Missing:\n{filename}", ha='center', va='center', fontsize=8)
            
        ax.set_title(title, fontsize=8, fontweight='bold')
        ax.axis('off')
        
    plt.tight_layout()
    output_grid = 'DARTIS_Visual_Grid.png'
    plt.savefig(output_grid, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Successfully generated '{output_grid}'!")
else:
    print("⚠️ Metadata file or image directory not found. Please verify paths.")