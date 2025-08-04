import gradio as gr
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime

def clean_excel(file):
    """
    Clean Excel file using pandas - more robust than xlwings
    Logic:
    - Pick the sheet with the most rows.
    - Find the header row: first row with > 3 non-empty cells.
    """
    try:
        all_sheets = pd.read_excel(file.name, sheet_name=None, header=None)

        # Find the sheet with the most rows
        target_sheet_name = max(all_sheets.items(), key=lambda x: len(x[1]))[0]
        target_df = all_sheets[target_sheet_name]

        if target_df is None or target_df.empty:
            return None, "❌ Error: No data found in the largest sheet.", None

        # Identify the header row: first row with more than 3 non-empty cells
        header_row_idx = None
        first_non_empty_col = None

        for i in range(len(target_df)):
            row_values = target_df.iloc[i].values
            non_empty_indices = [idx for idx, v in enumerate(row_values) if pd.notna(v) and str(v).strip() != ""]
            if len(non_empty_indices) > 3:
                header_row_idx = i
                first_non_empty_col = non_empty_indices[0]
                break

        if header_row_idx is None:
            return None, "❌ Error: Could not find a row with more than 3 filled cells.", None

        # Clean the dataframe
        cleaned_df = target_df.iloc[header_row_idx:].copy()

        # Remove columns before the first filled column
        if first_non_empty_col > 0:
            cleaned_df = cleaned_df.iloc[:, first_non_empty_col:].copy()

        # Reset index and set first row as header
        cleaned_df.reset_index(drop=True, inplace=True)
        cleaned_df.columns = cleaned_df.iloc[0]
        cleaned_df = cleaned_df.iloc[1:].reset_index(drop=True)

        # Drop fully empty rows/columns
        cleaned_df = cleaned_df.dropna(how='all')
        cleaned_df = cleaned_df.loc[:, ~cleaned_df.isnull().all()]

        # Save to temp file
        tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
        tmp_output.close()

        with pd.ExcelWriter(tmp_output.name, engine='openpyxl') as writer:
            cleaned_df.to_excel(writer, sheet_name='Data', index=False)

        return tmp_output.name, "✅ Excel cleaned successfully.", None

    except Exception as e:
        return None, f"❌ Error: {str(e)}", None


def process_cleaned_excel(file_path, master_file):
    if file_path is None:
        return pd.DataFrame(), "❌ Error: No cleaned file to process.", None
    
    if master_file is None:
        return pd.DataFrame(), "❌ Error: No master file uploaded.", None

    df = pd.read_excel(file_path, sheet_name='Data', header=0)

    # Filter for department 25902
    filtered_dept = df[df['Loaning Department ID'] == 25902]
    filtered_dept = filtered_dept.copy()

    # ⭐ EXACT column replacement logic from your original script
    filtered_dept.columns = [col.replace('PAG - PCB Code mapping', 'PAG').replace('PAG as per mapping file', 'PAG') for col in filtered_dept.columns]
    filtered_dept.columns = [col.replace('PCB Code', 'Code') if 'PCB Code' in col else col for col in filtered_dept.columns]
    filtered_dept['PAG'] = filtered_dept['PAG'].astype(str).str.strip().str.upper()
    
    # ⭐ EXACT processing logic from your original script
    selected_df = filtered_dept[['Member Supervisor', 'Code', 'PAG']].sort_values(by='Code')
    
    # Group by key columns to identify the manager with max partners and map it to Unspecified 
    grouped_df_gcc = selected_df.groupby(['Code','PAG','Member Supervisor']) \
                                .size() \
                                .reset_index(name='Count')
    
    # Fetch supervisor with highest count for each Code and PAG and remove duplicates
    grouped_counts_sorted = (
        grouped_df_gcc
        .sort_values(by=['Code','PAG','Count'], ascending=[True, True, False])
        .drop_duplicates(subset=['PAG', 'Code'], keep='first').rename(columns={'Member Supervisor': 'Member Supervisor_New'})
    )
    
    unspecified_df = selected_df[selected_df['Member Supervisor'] == 'Unspecified Unspecified']
    joined_df = pd.merge(grouped_counts_sorted, unspecified_df, on=['PAG', 'Code'], how='inner').drop_duplicates()

    df_unspecified_joined = pd.merge(
        filtered_dept,
        joined_df,
        on=['PAG', 'Code', 'Member Supervisor'],
        how='left'
    )

    df_unspecified_joined['Member Supervisor'] = np.where(
        df_unspecified_joined['Member Supervisor'] == 'Unspecified Unspecified',
        df_unspecified_joined['Member Supervisor_New'],
        df_unspecified_joined['Member Supervisor']
    )

    b_test = df_unspecified_joined.drop(columns=['Member Supervisor_New', 'Count'])

    # ⭐ EXACT fallback logic using uploaded master file
    df_leaders = pd.read_excel(master_file.name, sheet_name='Sheet1', header=0)
    df_leaders = df_leaders.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    df_leaders['PAG'] = df_leaders['PAG'].astype(str).str.strip().str.upper()
    
    b_test1 = b_test[b_test['Member Supervisor'] == 'Unspecified Unspecified']
    
    df_accural_manager_code = df_leaders.drop_duplicates(subset=['PAG'], keep='first')
    
    df_leaders_UN = b_test1.merge(df_accural_manager_code, on=['PAG'], how='inner')
    
    df_leaders_select = df_leaders_UN[['PAG', 'Code','Member Supervisor_y']]	
    
    df_leaders_final = pd.merge(
        b_test,
        df_leaders_select,
        on=['PAG', 'Code'],
        how='left'
    )

    df_leaders_final['Member Supervisor'] = np.where(
        df_leaders_final['Member Supervisor'] == 'Unspecified Unspecified',
        df_leaders_final['Member Supervisor_y'],
        df_leaders_final['Member Supervisor']
    )

    df_leaders_final = df_leaders_final.drop(columns=['Member Supervisor_y'])
    
    # Generate a Master file from LnB with director mapping
    df_leaders_select = df_leaders_final[['Member Supervisor','PAG']].drop_duplicates()
    df_leaders_select = df_leaders_select.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
    df_leaders_select['PAG'] = df_leaders_select['PAG'].astype(str).str.strip().str.upper()

    df_leaders_LNB = df_leaders_select.copy()

    df_leaders_director = df_leaders.copy()
    df_leaders_monthly = pd.merge(
        df_leaders_LNB,
        df_leaders_director,
         on=['PAG', 'Member Supervisor'],
        how='left'
    ).drop_duplicates()

    # ✅ Sort so rows with null Director are at the top (GUI feature)
    df_leaders_monthly["__SortNull__"] = df_leaders_monthly["DIRECTOR"].isnull() | (df_leaders_monthly["DIRECTOR"] == "")
    df_leaders_monthly = df_leaders_monthly.sort_values(
        by="__SortNull__",
        ascending=False
    ).drop(columns="__SortNull__")

    return df_leaders_monthly, df_leaders_monthly, df_leaders_final


def export_edited_master(df_edited):
    if df_edited is None or len(df_edited) == 0:
        return None

    tmp_export = Path(tempfile.gettempdir()) / "Master_Data_Leaders_Monthly.xlsx"
    df_edited.to_excel(tmp_export, index=False)

    return str(tmp_export)

def process_master_data_update(df_edited, df_leaders_final):
    """
    Process the approved edited table to create Master_Data_Leaders and LnB_Mapped_Monthly files
    Following the exact logic from the original script
    """
    if df_edited is None or len(df_edited) == 0:
        return None, None, None
    
    try:
        # Create temporary file for the monthly data
        tmp_monthly = Path(tempfile.gettempdir()) / "Master_Data_Leaders_Monthly.xlsx"
        df_edited.to_excel(tmp_monthly, index=False)
        
        # Read the updated master (monthly) file - exactly as in original script
        df_master_mapping = pd.read_excel(tmp_monthly, sheet_name='Sheet1', header=0) 
        df_manager_director = df_master_mapping[['PAG','Member Supervisor','DIRECTOR']].drop_duplicates()
        
        # Merge with leaders_final data
        df_director_map = pd.merge(
            df_leaders_final,
            df_manager_director,
            on=['PAG', 'Member Supervisor'],
            how='left'
        ).drop_duplicates()
        
        # Create master data for directors
        df_director_master = df_director_map[['PAG','Member Supervisor','DIRECTOR']].drop_duplicates()
        
        # Create output files
        master_data_path = Path(tempfile.gettempdir()) / "Master_Data_Leaders.xlsx"
        lnb_mapped_path = Path(tempfile.gettempdir()) / "LnB_Mapped_Month.xlsx"
        
        df_director_master.to_excel(master_data_path, index=False)
        df_director_map.to_excel(lnb_mapped_path, index=False)
        
        return str(master_data_path), str(lnb_mapped_path), df_director_map
        
    except Exception as e:
        return None, None, None

def process_accrual_mapping(accrual_file, df_director_map):
    """
    Process accrual file and map it with director data
    """
    if accrual_file is None:
        return None
    
    if df_director_map is None or len(df_director_map) == 0:
        return None
    
    try:
        # First, clean the accrual file using the same cleaning logic
        cleaned_accrual_path, clean_msg, clean_error = clean_excel(accrual_file)
        if clean_error:
            return None
        
        # Read the cleaned accrual file
        df_accrual = pd.read_excel(cleaned_accrual_path, sheet_name='Data', header=0)
        
        # Filter accrual data
        filtered_df_accrual = df_accrual[
            (df_accrual['Trx Fin Dept'] == 25902) &
            (df_accrual['Trx OU'] == 1062) &
            (df_accrual['Exclusion'] == 'N') &
            (df_accrual['Account 425%'] == 425000) &
            (df_accrual['Accruals type'] == 'Sharing')
        ].copy()
        
        # Create MEMBER ID column
        filtered_df_accrual['MEMBER ID'] = filtered_df_accrual['Empl ID'].astype(str)
        
        # Select relevant columns
        filtered_df_select = filtered_df_accrual[['MEMBER ID', 'Project']].drop_duplicates()
        
        # Process director mapping data
        df_leaders_accural = df_director_map.copy()
        df_leaders_accural['MEMBER ID'] = df_leaders_accural['Member Name & ID'].str.split().str[-1]
        df_leaders_CODE = df_leaders_accural.rename(columns={'Code': 'Project'})
        df_leaders_CODE_select = df_leaders_CODE[['MEMBER ID', 'Project', 'Member Supervisor', 'DIRECTOR']].drop_duplicates()
        
        # Merge accrual data with leader codes
        df_accural_manager_code = pd.merge(
            filtered_df_select,
            df_leaders_CODE_select,
            on=['MEMBER ID', 'Project'],
            how='left'
        )
        
        df_accural_manager_code2 = df_accural_manager_code.drop_duplicates(subset=['MEMBER ID', 'Project'], keep='first')
        
        # Final merge
        df_accural_final = filtered_df_accrual.merge(
            df_accural_manager_code2[['MEMBER ID', 'Project', 'Member Supervisor', 'DIRECTOR']],
            on=['MEMBER ID', 'Project'],
            how='left'
        )
        
        # Save to temporary file
        accrual_output_path = Path(tempfile.gettempdir()) / "Accrual_Mapped.xlsx"
        df_accural_final.to_excel(accrual_output_path, index=False)
        
        return str(accrual_output_path)
        
    except Exception as e:
        return None

def get_director_choices(df):
    """Get unique DIRECTOR values for dropdown, excluding null/empty values"""
    if df is None or len(df) == 0 or 'DIRECTOR' not in df.columns:
        return []
    
    # Get unique non-null directors
    directors = df['DIRECTOR'].dropna().unique()
    directors = [d for d in directors if str(d).strip() != '']
    return sorted(list(directors))

def create_director_info_display(df):
    """Create an info display showing available director options"""
    if df is None or len(df) == 0:
        return "No data available"
    
    choices = get_director_choices(df)
    if not choices:
        return "No director options found in data"
    
    # Create a simple, copy-friendly list
    directors_list = " | ".join([f"**{director}**" for director in choices])
    
    return f"""**Available Director Options:**

{directors_list}

*Click on any director name above to select and copy it, then paste into the DIRECTOR column cells.*"""

with gr.Blocks(title="Excel Cleaner + Master Mapper") as demo:
    # Header with logo and center-aligned title
    with gr.Row(equal_height=True):
        with gr.Column(scale=0, min_width=100):
            gr.Image(
                value="logo.png",  # Replace with your image file or path
                show_label=False,
                height=60,
                width=60,
                elem_id="logo"
            )
        with gr.Column(scale=1):
            gr.Markdown(
                """
                <h2 style="text-align:center; font-size:36px; font-weight:bold; color:#1a237e; margin:0; padding-top:16px;position:relative; left: -55px;">
                    Realization Margin - Computation
                </h2>
                """,
                elem_id="title"
            )

    # Side-by-side file inputs
    with gr.Row():
        with gr.Column():
            master_file_input = gr.File(
                label="Upload Master Leaders File (.xlsx)",
                file_types=[".xlsx"],
                type="filepath"
            )
        with gr.Column():
            file_input = gr.File(
                label="Upload LnB File (.xlsx)",
                file_types=[".xlsx"],
                type="filepath"
            )

    with gr.Row():
        process_button = gr.Button("Clean and Process Files", variant="primary", elem_classes="btn-blue")

    full_data = gr.State()
    leaders_final_data = gr.State()
    director_map_data = gr.State()
    director_choices = gr.State()

    director_info = gr.Markdown(
        label="Director Options",
        value="Upload and process files to see available director options.",
        elem_id="director-info"
    )

    dataframe_output = gr.Dataframe(
        label="Editable Final Mapping (null Directors on top)",
        interactive=True,
        wrap=True,
        column_widths=["flex", "flex", "200px"]
    )

    with gr.Row():
        export_button = gr.Button("Approve Monthly File", variant="primary", elem_classes="btn-blue")
    
    export_file_output = gr.File(label="Download Monthly Mapping", visible=True)
    
    # New section for master data processing
    gr.Markdown("### Master Data Processing")
    gr.Markdown("After approving the edited table above, click below to generate the master files:")
    
    with gr.Row():
        process_master_button = gr.Button("Generate Master Files", variant="secondary", elem_classes="btn-narrow")
    
    with gr.Row():
        master_data_output = gr.File(label="Download Master_Data_Leaders.xlsx", visible=True)
        lnb_mapped_output = gr.File(label="Download LnB_Mapped_Month.xlsx", visible=True)
    
    # New section for accrual processing
    gr.Markdown("### Accrual Data Processing")
    gr.Markdown("After generating master files, upload an accrual file to map with director data:")
    
    accrual_file_input = gr.File(
        label="Upload Accrual File (.xlsx/.xlsb)",
        file_types=[".xlsx", ".xlsb"],
        type="filepath"
    )
    
    with gr.Row():
        process_accrual_button = gr.Button("Process Accrual Mapping", variant="secondary", elem_classes="btn-narrow")
    
    accrual_output = gr.File(label="Download Accrual_Mapped.xlsx", visible=True)

    def clean_and_process(file, master_file):
        if file is None or master_file is None:
            return gr.update(), gr.update(), gr.update(), gr.update(), None, None, None
            
        cleaned_path, clean_msg, clean_error = clean_excel(file)
        if clean_error:
            return gr.update(), gr.update(), gr.update(), gr.update(), None, None, None
            
        df_final, df_for_state, df_leaders_final = process_cleaned_excel(cleaned_path, master_file)
        
        # Create director info display
        director_info_text = create_director_info_display(df_final)
        
        return df_final, director_info_text, gr.update(), gr.update(), df_for_state, df_leaders_final, None

    process_button.click(
        clean_and_process,
        inputs=[file_input, master_file_input],
        outputs=[dataframe_output, director_info, export_file_output, master_data_output, full_data, leaders_final_data, director_map_data]
    )

    export_button.click(
        export_edited_master,
        inputs=dataframe_output,
        outputs=export_file_output
    )
    
    process_master_button.click(
        process_master_data_update,
        inputs=[dataframe_output, leaders_final_data],
        outputs=[master_data_output, lnb_mapped_output, director_map_data]
    )
    
    process_accrual_button.click(
        process_accrual_mapping,
        inputs=[accrual_file_input, director_map_data],
        outputs=accrual_output
    )

if __name__ == "__main__":
    demo.css = """
    button[title="Download"], button[title="Fullscreen"] {
        display: none !important;
    }
    #logo {
        padding-right: 10px;
    }
    .btn-narrow {
        background: linear-gradient(135deg, #2196f3, #1976d2) !important;
        color: white !important;
        border: none !important;
        max-width: 300px !important;
        margin: 0 auto !important;
        display: block !important;
    }
    .btn-narrow:hover {
        background: linear-gradient(135deg, #1976d2, #1565c0) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3) !important;
    }
    .btn-blue {
        background: linear-gradient(135deg, #2196f3, #1976d2) !important;
        color: white !important;
        border: none !important;
        max-width: 300px !important;
        margin: 0 auto !important;
        display: block !important;
    }
    .btn-blue:hover {
        background: linear-gradient(135deg, #1976d2, #1565c0) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.3) !important;
    }
    .gradio-container .gradio-button {
        max-width: 300px;
        margin: 0 auto;
        display: block;
    }
    #director-info {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
    }
    #director-info strong {
        background-color: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 6px;
        padding: 2px 8px;
        margin: 0 2px;
        color: #1976d2;
        user-select: all;
        cursor: pointer;
    }
    #director-info strong:hover {
        background-color: #bbdefb;
    }
    """
    demo.launch(
        inbrowser=True,
        share=False,
        show_error=True
    )