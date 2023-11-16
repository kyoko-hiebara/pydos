import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import numpy as np


Efermi = -1.7958333333
x_display_range = (-3, 5)
y_display_range = (-120, 120)
pdos1_scale_factor = 5
color_settings = {
    'Total DOS': '#FFCCCC',
    'PDOS 1': '#98514B',
}

legend_names = {
    'Total DOS (Spin 1)': 'Total',
    'PDOS 1 (Spin 1)': 'CO$_2$ x5'
}

csv_file_path = './dos_data.csv' 

df = pd.read_csv(csv_file_path)


energy = df['Energy']

def plot_custom_color_pdos(df, Efermi, pdos1_scale_factor, x_range, y_range, color_settings, legend_names):
    plt.figure(figsize=(14, 8))

    # Shift energy by Efermi
    shifted_energy = df['Energy'] - Efermi


    for col in df.columns[1:]: 
        base_name = col.split(' (')[0]

        if base_name in color_settings:
            color = color_settings[base_name]
            label_name = legend_names.get(col, col)  
            if 'PDOS 1' in col:
                plt.fill_between(shifted_energy.to_numpy(), df[col].to_numpy() * pdos1_scale_factor,
                 label=label_name if '(Spin 2)' not in col else '', color=color, alpha=0.8)
            else:
                plt.fill_between(shifted_energy.to_numpy(), df[col].to_numpy(),
                 label=label_name if '(Spin 2)' not in col else '', color=color, alpha=0.8)
                

    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.yticks([])
    plt.xticks(fontsize=15)
    plt.grid(True, which='major', axis='x', linestyle=':', linewidth=1.0)
    plt.xlabel('E-E$_{Fermi}$ (eV)', fontsize=20)
    plt.ylabel('DOS (a.u.)', fontsize=20)
    plt.title('')
    plt.legend(loc="upper right", fontsize=20)
    plt.savefig('pdos.png')
    plt.show()

plot_custom_color_pdos(df, Efermi, pdos1_scale_factor, x_display_range, y_display_range, color_settings, legend_names)
